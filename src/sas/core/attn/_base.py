import torch.nn as nn
from typing import Dict, Any

class SparseAttention(nn.Module):
    """
    Base class for sparse attention modules.
    """
    def __init__(self, config: Dict[str, Any], layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    @classmethod
    def from_dense(cls, dense_module: nn.Module, config: Dict[str, Any]) -> "SparseAttention":
        """Create a sparse attention module from a dense one. Subclasses must implement."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError
