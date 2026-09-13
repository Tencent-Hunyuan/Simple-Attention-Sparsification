from typing import Dict, Any, Type
import torch.nn as nn

SPARSE_ATTENTION_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_sparse_attention(name: str):
    def decorator(cls: Type[nn.Module]):
        SPARSE_ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator
