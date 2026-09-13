import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Dict, Any, Optional, Tuple, List
from transformers.cache_utils import Cache

from ._base import SparseAttention
from ._registry import register_sparse_attention
from ..kernels.block_scoring import fused_causal_matmul_logsoftmax
from ..kernels.sparsex_attention import sparsex_attn


# Fixed seed so every rank generates the identical full tensor before sharding.
_ROUTER_INIT_SEED = 42


def _seeded_xavier_uniform_(param: nn.Parameter) -> None:
    """xavier_uniform_ init that is rank-consistent and DTensor-aware.

    Under FSDP2 the parameter is a sharded DTensor. We build the full tensor on
    CPU under a fixed, shape-derived seed (so all ranks agree), then distribute
    it to match the parameter's mesh/placements.
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    def _fill(dst: torch.Tensor) -> None:
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(_ROUTER_INIT_SEED + hash(tuple(dst.shape)) % (2**31))
        init.xavier_uniform_(dst)
        torch.random.set_rng_state(rng_state)

    if isinstance(param, DTensor):
        full = torch.empty(param.shape, device="cpu", dtype=param.dtype)
        _fill(full)
        sharded = distribute_tensor(full.to(param.device), param.device_mesh, param.placements)
        with torch.no_grad():
            param.copy_(sharded)
    else:
        with torch.no_grad():
            _fill(param)


def _fill_ones_(param: nn.Parameter) -> None:
    """Fill a parameter with ones, DTensor-aware."""
    from torch.distributed.tensor import DTensor, distribute_tensor

    if isinstance(param, DTensor):
        full = torch.ones(param.shape, device=param.device, dtype=param.dtype)
        sharded = distribute_tensor(full, param.device_mesh, param.placements)
        with torch.no_grad():
            param.copy_(sharded)
    else:
        with torch.no_grad():
            param.fill_(1.0)


def init_router_weights(model: nn.Module) -> None:
    """Re-initialize router weights across a (possibly sharded) model.

    Walks the module tree and calls ``reset_sparse_parameters`` on every router
    submodule that defines it. Safe to call after ``build_parallelize_model``:
    the per-module methods are DTensor-aware and rank-consistent.
    """
    for module in model.modules():
        reset_fn = getattr(module, "reset_sparse_parameters", None)
        if callable(reset_fn):
            reset_fn()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    if x.shape[-1] != rotary_dim:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_embed = torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)
    else:
        x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


class RMSNorm(nn.Module):
    """RMS Normalization layer."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def reset_sparse_parameters(self) -> None:
        _fill_ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HeadPoolingLinear(nn.Module):
    """Linear layer with head pooling for GQA."""
    def __init__(self, num_k_head: int, gqa_group_size: int, model_hidden_size: int, gate_hidden_size: int):
        super(HeadPoolingLinear, self).__init__()
        self.num_k_head = num_k_head
        self.gqa_group_size = gqa_group_size
        self.model_hidden_size = model_hidden_size
        self.gate_hidden_size = gate_hidden_size
        self.weight = nn.Parameter(torch.Tensor(self.num_k_head, gqa_group_size, self.model_hidden_size, self.gate_hidden_size))
        init.xavier_uniform_(self.weight)

    def reset_sparse_parameters(self) -> None:
        _seeded_xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, num_q_head, L, model_hidden_size = x.shape
        x = x.view(B, self.num_k_head, self.gqa_group_size, L, model_hidden_size)
        return torch.einsum('bkgli,kgio->bklo', x, self.weight)


class MultiHeadLinear(nn.Module):
    """Multi-head linear layer."""
    def __init__(self, in_channel_size: int, hidden_size: int, num_head: int):
        super(MultiHeadLinear, self).__init__()
        self.in_channel = in_channel_size
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.weight = nn.Parameter(torch.Tensor(self.num_head, self.in_channel, self.hidden_size))
        init.xavier_uniform_(self.weight)

    def reset_sparse_parameters(self) -> None:
        _seeded_xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bhli,hio->bhlo', x, self.weight)


# Pooling functions for batch format
def maxpool_batch(x: torch.Tensor, block_size: int, kv_len: int) -> torch.Tensor:
    B, H, L, D = x.shape
    device = x.device
    dtype = x.dtype

    num_blocks = (kv_len + block_size - 1) // block_size
    L_pad = num_blocks * block_size

    if L < L_pad:
        x_padded = F.pad(x, (0, 0, 0, L_pad - L))
    else:
        x_padded = x

    x_blocks = x_padded.view(B, H, num_blocks, block_size, D)

    token_indices = torch.arange(L_pad, device=device)
    valid_mask = (token_indices < kv_len).view(1, 1, num_blocks, block_size, 1).to(dtype)

    x_blocks_masked = x_blocks * valid_mask + (1 - valid_mask) * torch.finfo(dtype).min
    k_summary = x_blocks_masked.max(dim=3)[0]

    return k_summary


def avgpool_batch(x: torch.Tensor, block_size: int, kv_len: int) -> torch.Tensor:
    B, H, L, D = x.shape
    device = x.device
    dtype = x.dtype

    num_blocks = (kv_len + block_size - 1) // block_size
    L_pad = num_blocks * block_size

    if L < L_pad:
        x_padded = F.pad(x, (0, 0, 0, L_pad - L))
    else:
        x_padded = x

    x_blocks = x_padded.view(B, H, num_blocks, block_size, D)

    token_indices = torch.arange(L_pad, device=device)
    valid_mask = (token_indices < kv_len).view(1, 1, num_blocks, block_size, 1).to(dtype)

    block_sums = (x_blocks * valid_mask).sum(dim=3)
    block_counts = valid_mask.sum(dim=3).clamp_min(1.0)
    k_summary = block_sums / block_counts

    return k_summary


def minpool_batch(x: torch.Tensor, block_size: int, kv_len: int) -> torch.Tensor:
    return -maxpool_batch(-x, block_size, kv_len)


POOL_FUNCS = {
    'max': maxpool_batch,
    'min': minpool_batch,
    'avg': avgpool_batch,
}


class AttnGateRouter(nn.Module):
    """
    Attention Gate Router following SeerAttention implementation.
    Supports Q head pooling and K block pooling.
    """
    def __init__(
        self,
        block_size: int,
        model_hidden_size: int,
        gate_hidden_size: int,
        num_k_head: int,
        num_q_head: int,
        q_head_pooling_type: str = "Qproj",
        k_pooling_names: List[str] = ["max", "min", "avg"],
        use_qk_norm: bool = True,
    ):
        super(AttnGateRouter, self).__init__()
        self.block_size = block_size
        self.model_hidden_size = model_hidden_size
        self.gate_hidden_size = gate_hidden_size
        self.num_k_head = num_k_head
        self.num_q_head = num_q_head
        self.gqa_group_size = int(num_q_head // num_k_head)
        self.k_pooling_funcs = [POOL_FUNCS[name] for name in k_pooling_names]
        self.use_qk_norm = use_qk_norm
        self.q_head_pooling_type = q_head_pooling_type

        self.k_dup_size = len(self.k_pooling_funcs)
        k_in_channel_size = model_hidden_size * self.k_dup_size

        if self.q_head_pooling_type == "Qproj":
            self.attngate_linear_q = HeadPoolingLinear(self.num_k_head, self.gqa_group_size, self.model_hidden_size, self.gate_hidden_size)
        elif self.q_head_pooling_type == "Qavgproj":
            self.attngate_linear_q = MultiHeadLinear(self.model_hidden_size, self.gate_hidden_size, self.num_k_head)
        else:
            self.attngate_linear_q = None

        self.attngate_linear_k = MultiHeadLinear(k_in_channel_size, self.gate_hidden_size, self.num_k_head)

        if self.use_qk_norm:
            self.attngate_qnorm = RMSNorm(self.gate_hidden_size, eps=1e-06)
            self.attngate_knorm = RMSNorm(self.gate_hidden_size, eps=1e-06)

    def forward_q(
        self,
        q_states: torch.Tensor,
        position_embeddings_gate_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qavg":
            q = F.avg_pool2d(q_states, kernel_size=[self.gqa_group_size, 1], stride=[self.gqa_group_size, 1])
        else:
            q = q_states

        if self.q_head_pooling_type == "Qavgproj" or self.q_head_pooling_type == "Qproj":
            q = self.attngate_linear_q(q)

        if self.use_qk_norm:
            q = self.attngate_qnorm(q)

        if position_embeddings_gate_q is not None:
            cos, sin = position_embeddings_gate_q
            q = apply_rotary_pos_emb_single(q, cos, sin, unsqueeze_dim=1)

        return q

    def forward_k_blocks(
        self,
        key_states: torch.Tensor,
        kv_len: int,
        block_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        k_pooled = [pool_func(key_states, self.block_size, kv_len) for pool_func in self.k_pooling_funcs]
        k = torch.cat(k_pooled, dim=-1)

        k = self.attngate_linear_k(k)

        if self.use_qk_norm:
            k = self.attngate_knorm(k)

        if block_position_embeddings is not None:
            cos, sin = block_position_embeddings
            k = apply_rotary_pos_emb_single(k, cos, sin, unsqueeze_dim=1)

        return k


@register_sparse_attention("simple_sparse_attention")
class SimpleSparseAttention(SparseAttention):
    """
    Soft-gate attention with a decoupled router and Top-K block pruning (training).

    Pipeline:
    1. Router pools Q per token and K per block into a low-dim gate space.
    2. Block scoring via the `fused_causal_matmul_logsoftmax` kernel produces
       causal logsoftmax-normalized block scores.
    3. Top-K block selection via a per-query threshold (kthvalue).
    4. Sparse attention via the `sparsex_attn` kernel, restricted to selected
       blocks with soft gating.

    Math:
        block_logsm = logsoftmax(causal_mask(q_route @ k_blocks^T * gate_scale))
        threshold   = kthvalue(block_logsm, k=num_blocks - topk + 1)
        o_t         = softmax(s_ti + log g_{t, floor(i/b)}) @ V   (selected blocks only)

    where g = exp(block_logsm) for selected blocks (above threshold), and the
    self-block always has gate=1 (logsm=0) and is always included. Gradients flow
    through block_logsm, so the router is differentiable.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.block_size = config.get("block_size", 64)
        self.topk = config.get("topk", 4)

        # Router configuration
        self.gate_hidden_size = config.get("gate_hidden_size", 128)
        self.q_head_pooling_type = config.get("q_head_pooling_type", "Qproj")
        self.k_pooling_names = config.get("k_pooling_names", ["max", "min", "avg"])
        self.use_qk_norm = config.get("use_qk_norm", True)
        self.use_rope_for_gate = config.get("use_rope_for_gate", True)

        self.layer_idx = config.get("layer_idx", 0)
        self.head_dim = None
        self.num_attention_heads = None
        self.num_key_value_heads = None
        self.num_key_value_groups = None
        self.hidden_size = None
        self.scaling = None
        self.is_causal = True

        self.q_proj: nn.Linear = None
        self.k_proj: nn.Linear = None
        self.v_proj: nn.Linear = None
        self.o_proj: nn.Linear = None

        self.q_norm: nn.Module = None
        self.k_norm: nn.Module = None
        self._has_qk_norm = False

        self._apply_rope = None
        self.router: AttnGateRouter = None

    @classmethod
    def from_dense(cls, dense_module: nn.Module, config: Dict[str, Any]) -> "SimpleSparseAttention":
        instance = cls(config)

        for name, child in dense_module.named_children():
            setattr(instance, name, child)

        module_obj = inspect.getmodule(dense_module)
        rope_func = getattr(module_obj, "apply_rotary_pos_emb", None)
        if rope_func is None:
            raise AttributeError(
                "SimpleSparseAttention.from_dense expects the dense attention module to define "
                "`apply_rotary_pos_emb` in the same module."
            )
        instance._apply_rope = rope_func

        instance._has_qk_norm = (
            getattr(instance, 'q_norm', None) is not None and
            getattr(instance, 'k_norm', None) is not None
        )

        instance.head_dim = dense_module.head_dim
        instance.layer_idx = dense_module.layer_idx
        instance.scaling = dense_module.scaling
        instance.is_causal = dense_module.is_causal

        instance.num_key_value_groups = dense_module.num_key_value_groups

        if hasattr(dense_module, 'config'):
            instance.num_attention_heads = dense_module.config.num_attention_heads
            instance.num_key_value_heads = dense_module.config.num_key_value_heads
            instance.hidden_size = dense_module.config.hidden_size
        else:
            instance.num_attention_heads = getattr(dense_module, 'num_heads',
                                                    getattr(dense_module, 'num_attention_heads', None))
            instance.num_key_value_heads = getattr(dense_module, 'num_key_value_heads', None)
            instance.hidden_size = getattr(dense_module, 'hidden_size', None)

        instance.router = AttnGateRouter(
            block_size=instance.block_size,
            model_hidden_size=instance.head_dim,
            gate_hidden_size=instance.gate_hidden_size,
            num_k_head=instance.num_key_value_heads,
            num_q_head=instance.num_attention_heads,
            q_head_pooling_type=instance.q_head_pooling_type,
            k_pooling_names=instance.k_pooling_names,
            use_qk_norm=instance.use_qk_norm,
        ).to(dense_module.q_proj.weight.device).to(dense_module.q_proj.weight.dtype)

        return instance

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self._has_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # [B, H, T, D] for RoPE and router
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Save pre-RoPE states for router (detached)
        q_unrope = query_states.detach()
        k_unrope = key_states.detach()

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        batch_size, q_heads, q_len, head_dim = query_states.shape
        kv_heads = key_states.shape[1]
        kv_len = key_states.shape[2]
        device = query_states.device

        # Token- and block-level cumulative sequence lengths (dense / non-packed path)
        cu_seqlens_q  = torch.arange(0, (batch_size + 1) * q_len,  q_len,  dtype=torch.int32, device=device)
        cu_seqlens_kv = torch.arange(0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device=device)
        max_seqlen_q  = q_len
        max_seqlen_kv = kv_len

        seq_lens       = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()          # [batch_size]
        blocks_per_seq = (seq_lens + self.block_size - 1) // self.block_size     # [batch_size]
        cu_seqlens_kb  = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            blocks_per_seq.cumsum(0).to(torch.int32),
        ])                                                                        # [batch_size+1]
        total_blocks   = int(cu_seqlens_kb[-1].item())
        max_seqlen_kb  = int(blocks_per_seq.max().item())

        # Gate RoPE embeddings (optional)
        position_embeddings_gate_q = None
        block_position_embeddings  = None
        if self.use_rope_for_gate:
            position_embeddings_gate_q, block_position_embeddings = \
                self._generate_rope_embeddings_for_gate(
                    cos=cos, sin=sin, kv_len=kv_len, device=device,
                )

        # Router: pool Q per token and K per block, then flatten to varlen layout
        q_route = self.router.forward_q(q_unrope, position_embeddings_gate_q)     # [B, H_kv, q_len, gate_dim]
        nheads_score = q_route.shape[1]  # == kv_heads
        k_blocks = self.router.forward_k_blocks(k_unrope, kv_len, block_position_embeddings)  # [B, H_kv, num_blocks, gate_dim]

        q_route_varlen = q_route.permute(0, 2, 1, 3).reshape(
            batch_size * q_len, nheads_score, self.gate_hidden_size
        ).contiguous()
        k_blocks_varlen = k_blocks.permute(0, 2, 1, 3).reshape(
            total_blocks, nheads_score, self.gate_hidden_size
        ).contiguous()

        gate_scale = 1.0 / math.sqrt(self.gate_hidden_size)

        # 1. Block scoring: logsoftmax-normalized scores, [nheads_score, total_q, max_seqlen_kb]
        block_logsm = fused_causal_matmul_logsoftmax(
            q_route_varlen, k_blocks_varlen,
            cu_seqlens_q, cu_seqlens_kb,
            max_seqlen_q, max_seqlen_kb,
            self.block_size, gate_scale,
        )
        if nheads_score != q_heads:  # expand kv_heads -> q_heads for GQA
            block_logsm = block_logsm.repeat_interleave(q_heads // nheads_score, dim=0)

        # 2. Top-K block selection via threshold
        max_cols    = block_logsm.size(-1)
        actual_topk = min(self.topk, max_cols)
        logsm_th    = torch.kthvalue(block_logsm, k=max_cols - actual_topk + 1, dim=-1).values
        logsm_th    = torch.where(
            logsm_th == float('-inf'),
            torch.finfo(block_logsm.dtype).min,
            logsm_th,
        )

        # 3. Sparse attention via sparsex_attn kernel
        q_varlen = query_states.permute(0, 2, 1, 3).reshape(
            batch_size * q_len,  q_heads,  head_dim
        ).contiguous()
        k_varlen = key_states.permute(0, 2, 1, 3).reshape(
            batch_size * kv_len, kv_heads, head_dim
        ).contiguous()
        v_varlen = value_states.permute(0, 2, 1, 3).reshape(
            batch_size * kv_len, kv_heads, head_dim
        ).contiguous()

        scale = self.scaling if self.scaling is not None else head_dim ** -0.5

        o, lse = sparsex_attn(
            q_varlen, k_varlen, v_varlen,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            self.block_size, block_logsm, logsm_th,
            sm_scale=scale,
        )

        # [total_q, q_heads, head_dim] -> [B, T, H*D]
        attn_output = o.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _generate_rope_embeddings_for_gate(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_len: int,
        device: torch.device,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Generate RoPE embeddings for the router gate (training/prefill only).

        Q positions: [0, 1, ..., kv_len-1]
        Block positions: [0, block_size, 2*block_size, ...]
        """
        position_ids_q = torch.arange(0, kv_len, device=device, dtype=torch.long)

        max_seqlen_round = math.ceil(kv_len / self.block_size) * self.block_size
        block_position_ids = torch.arange(
            0, max_seqlen_round, self.block_size, device=device, dtype=torch.long
        )
        block_position_ids = torch.clamp(block_position_ids, max=kv_len - 1)

        cos_gate_q = cos[:, position_ids_q, :]
        sin_gate_q = sin[:, position_ids_q, :]
        position_embeddings_gate_q = (cos_gate_q, sin_gate_q)

        cos_block = cos[:, block_position_ids, :]
        sin_block = sin[:, block_position_ids, :]
        block_position_embeddings = (cos_block, sin_block)

        return position_embeddings_gate_q, block_position_embeddings
