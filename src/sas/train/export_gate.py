"""Export the trained gate router as an sglang-blocksparse *AttnGates* directory.

SAS training is router-only: the backbone is frozen and only the gate router
learns. Instead of dumping the full model (huge, backbone unchanged) and running
a separate conversion step, we build the small AttnGates dir directly at the end
of training, from the gathered rank-0 state dict.

The sglang ``seer_attn`` backend expects a directory containing:
  * ``attn_gate_weights.pth`` — gate router tensors, renamed
    ``...self_attn.router.attngate_*``  ->  ``...self_attn.attn_gate.attngate_*``
  * ``config.json`` — base Qwen3 config + ``seerattn_*`` fields + ``base_model``
  * tokenizer files (so the dir is self-contained), with the base model's
    tools-capable ``chat_template.jinja``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import torch

# router param prefix in the SAS model  ->  gate prefix expected by sglang seer_attn
_SAS_GATE_SEG = ".self_attn.router."
_SGLANG_GATE_SEG = ".self_attn.attn_gate."

# tokenizer-ish files to copy for a self-contained output dir
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def derive_seerattn_fields(sparse_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map the SAS sparse config -> sglang ``seerattn_*`` fields, with validation.

    sglang's seer_attn backend only supports a specific router configuration, so
    we validate up front and fail loudly rather than emit a config the server
    would silently mis-handle.
    """
    qpool = sparse_cfg.get("q_head_pooling_type")
    kpool = sparse_cfg.get("k_pooling_names")
    if qpool != "Qproj":
        raise ValueError(
            f"sglang seer_attn only supports q_head_pooling_type='Qproj', got {qpool!r}"
        )
    if list(kpool or []) != ["max", "min", "avg"]:
        raise ValueError(
            f"sglang seer_attn only supports k_pooling_names=['max','min','avg'] "
            f"(=> Kmaxminavg), got {kpool!r}"
        )
    if not sparse_cfg.get("use_qk_norm", False):
        raise ValueError("sglang seer_attn requires use_qk_norm=True")
    if not sparse_cfg.get("use_rope_for_gate", False):
        raise ValueError("sglang seer_attn requires use_rope_for_gate=True")

    return {
        "seerattn_gate_block_size": int(sparse_cfg["block_size"]),
        "seerattn_gate_hidden_size": int(sparse_cfg["gate_hidden_size"]),
        "seerattn_q_head_pooling_type": "Qproj",
        "seerattn_k_seq_pooling_type": "Kmaxminavg",
        "seerattn_use_qk_norm": True,
        "seerattn_use_rope": True,
    }


def _collect_gate_tensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pull only gate-router tensors from a full state dict and rename them."""
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if _SAS_GATE_SEG in key:
            new_key = key.replace(_SAS_GATE_SEG, _SGLANG_GATE_SEG)
            t = tensor
            if hasattr(t, "full_tensor"):          # DTensor -> gather full
                t = t.full_tensor()
            out[new_key] = t.detach().cpu()
    if not out:
        raise ValueError(
            f"No gate tensors ('{_SAS_GATE_SEG}') found in state dict "
            f"({len(state_dict)} keys). Is this a router-trained SAS ckpt?"
        )
    return out


def _fix_chat_template(output: Path, base_model: str) -> None:
    """Overwrite the copied chat_template with the BASE model's tools-capable one.

    SAS training assets may ship a minimal chat_template (role+content only, no
    tools rendering). sglang serves that verbatim, so tool schemas get dropped and
    the served model can't do function-calling. The base Qwen3 template supports
    tools; we copy it over, backing up the original to chat_template.jinja.bak.

    The base template lives in EITHER a standalone ``chat_template.jinja`` file or
    the ``chat_template`` field of ``tokenizer_config.json``. If neither exists
    (e.g. base_model is a remote repo id), warn and leave the template as-is.
    """
    base_dir = Path(base_model)
    base_ct = None
    src_desc = None

    base_jinja = base_dir / "chat_template.jinja"
    if base_jinja.is_file():
        base_ct = base_jinja.read_text()
        src_desc = "chat_template.jinja"
    else:
        base_tok_cfg = base_dir / "tokenizer_config.json"
        if base_tok_cfg.is_file():
            base_ct = json.loads(base_tok_cfg.read_text()).get("chat_template")
            src_desc = "tokenizer_config.json['chat_template']"

    if not base_ct:
        print(f"[export]   WARN: base_model '{base_model}' has no chat_template "
              f"(neither chat_template.jinja nor tokenizer_config.json['chat_template']); "
              f"served model may NOT support function-calling.")
        return
    if "tools" not in base_ct:
        print(f"[export]   WARN: base chat_template has no 'tools' logic "
              f"(len={len(base_ct)}); writing it anyway.")

    dst = output / "chat_template.jinja"
    if dst.exists():
        shutil.copy2(dst, output / "chat_template.jinja.bak")
        print("[export]   backed up original template -> chat_template.jinja.bak")
    dst.write_text(base_ct)
    print(f"[export]   wrote base chat_template ({len(base_ct)} bytes, "
          f"tools={'tools' in base_ct}, src={src_desc}) -> chat_template.jinja")


def export_attn_gates(
    state_dict: Dict[str, torch.Tensor],
    model_config: Any,
    base_model: str,
    output_dir: str,
    tokenizer_dir: str | None = None,
    copy_tokenizer: bool = True,
) -> None:
    """Write an sglang AttnGates dir from a gathered (rank-0) full state dict.

    Args:
        state_dict:    full model state dict (backbone + router) on rank 0.
        model_config:  the live HF config carrying ``sas_sparse_config``.
        base_model:    path to the frozen base Qwen3 (written to config.json and
                       used to source the tools-capable chat_template). Resolved
                       to an absolute path if it is a local dir.
        output_dir:    destination AttnGates dir.
        tokenizer_dir: dir holding tokenizer files to copy (defaults to base_model).
        copy_tokenizer: copy tokenizer files + fix chat_template into the output.
    """
    output = Path(output_dir)

    # Resolve base_model to an absolute path so config.json['base_model'] does not
    # depend on the serving process's cwd. A remote repo id is left untouched.
    base_path = Path(base_model)
    if base_path.exists():
        base_model = str(base_path.resolve())

    cfg = model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config)
    sparse_cfg = cfg.get("sas_sparse_config")
    if sparse_cfg is None:
        raise ValueError(
            "model_config has no 'sas_sparse_config' — not a seer-trained SAS model."
        )

    gate = _collect_gate_tensors(state_dict)
    n_layers = cfg.get("num_hidden_layers")
    per_layer = len(gate) // n_layers if isinstance(n_layers, int) and n_layers else "?"
    print(f"[export] {len(gate)} gate tensors ({per_layer}/layer) -> {output}")

    seerattn = derive_seerattn_fields(sparse_cfg)

    # base Qwen3 config, minus SAS-specific keys, plus seerattn_* + base_model
    out_cfg = {k: v for k, v in cfg.items() if k not in ("sas_sparse_config", "sas_sparse_mod")}
    out_cfg["architectures"] = ["Qwen3ForCausalLM"]  # sglang dispatches via --attention-backend seer_attn
    out_cfg["base_model"] = base_model
    out_cfg.update(seerattn)

    output.mkdir(parents=True, exist_ok=True)
    torch.save(gate, output / "attn_gate_weights.pth")
    (output / "config.json").write_text(json.dumps(out_cfg, indent=2))

    if copy_tokenizer:
        src_dir = Path(tokenizer_dir) if tokenizer_dir else base_path
        copied = []
        for fn in _TOKENIZER_FILES:
            src = src_dir / fn
            if src.exists():
                shutil.copy2(src, output / fn)
                copied.append(fn)
        print(f"[export]   copied tokenizer files from {src_dir}: {copied}")
        _fix_chat_template(output, base_model)

    print(f"[export]   done. base_model={base_model} "
          f"block_size={seerattn['seerattn_gate_block_size']} "
          f"gate_hidden={seerattn['seerattn_gate_hidden_size']}")
