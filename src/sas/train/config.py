from dataclasses import dataclass, field
from typing import Dict, Any

from veomni.arguments import DataArguments, ModelArguments, TrainingArguments

@dataclass
class SparseArguments:
    """Arguments for sparse attention module"""
    sparse_mod: str = field(
        default="",
        metadata={"help": "Sparse attention module type (e.g., 'simple_sparse_attention')"}
    )
    block_size: int = field(
        default=64,
        metadata={"help": "Block size for block sparse attention"}
    )
    topk: int = field(
        default=4,
        metadata={"help": "Number of topk blocks to attend to"}
    )
    # Router configuration (aligned with SeerAttn)
    gate_hidden_size: int = field(
        default=128,
        metadata={"help": "Gate hidden size for router"}
    )
    q_head_pooling_type: str = field(
        default="Qproj",
        metadata={"help": "Q head pooling type: Qproj, Qavgproj, Qavg, Qorig"}
    )
    k_pooling_names: str = field(
        default="avg",
        metadata={"help": "K pooling names: SeerAttn format like 'Kmaxminavg' or comma-separated 'max,min,avg' or single 'avg'"}
    )
    use_qk_norm: bool = field(
        default=False,
        metadata={"help": "Whether to use QK norm in router"}
    )
    use_rope_for_gate: bool = field(
        default=False,
        metadata={"help": "Whether to use RoPE for gate"}
    )
    # Independent router LR schedule configuration
    router_lr: float = field(
        default=1e-3,
        metadata={"help": "Router learning rate (independent of train.lr)"}
    )
    router_lr_decay_style: str = field(
        default="cosine",
        metadata={"help": "Router LR decay style: 'constant', 'linear', or 'cosine'"}
    )
    router_lr_warmup_ratio: float = field(
        default=0.01,
        metadata={"help": "Router warmup ratio (fraction of total training steps)"}
    )
    router_lr_min: float = field(
        default=1e-5,
        metadata={"help": "Router minimum learning rate"}
    )
    router_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Router weight decay"}
    )

    def to_sparse_config(self) -> Dict[str, Any]:
        """Convert to sparse attention module configuration dictionary"""
        # Parse k_pooling_names from SeerAttn format (e.g., "Kmaxminavg") or comma-separated string
        if isinstance(self.k_pooling_names, str):
            # Handle SeerAttn format: "Kmaxminavg" -> ["max", "min", "avg"]
            if self.k_pooling_names.startswith("K"):
                # Extract pooling types after "K"
                pool_str = self.k_pooling_names[1:]
                # Split by known pooling types
                k_pooling_list = []
                for pool_type in ["max", "min", "avg"]:
                    if pool_type in pool_str.lower():
                        k_pooling_list.append(pool_type)
                if not k_pooling_list:
                    k_pooling_list = ["avg"]  # default
            else:
                # Comma-separated format: "max,min,avg" or single "avg"
                k_pooling_list = [name.strip() for name in self.k_pooling_names.split(",")]
        else:
            k_pooling_list = self.k_pooling_names
        
        config = {
            "block_size": self.block_size,
            "topk": self.topk,
            # Router configuration
            "gate_hidden_size": self.gate_hidden_size,
            "q_head_pooling_type": self.q_head_pooling_type,
            "k_pooling_names": k_pooling_list,
            "use_qk_norm": self.use_qk_norm,
            "use_rope_for_gate": self.use_rope_for_gate,
        }
        return config


@dataclass
class SasArguments:
    """
    SAS training arguments, extending VeOmni's arguments system to support sparse attention configuration
    """
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    train: TrainingArguments = field(default_factory=TrainingArguments)
    sparse: SparseArguments = field(default_factory=SparseArguments)