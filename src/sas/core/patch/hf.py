from typing import Dict, Any, Optional
from transformers import PreTrainedModel
from ..attn._registry import SPARSE_ATTENTION_REGISTRY

def hf_convert(
    model: PreTrainedModel,
    sparse_mod: str,
    sparse_config: Dict[str, Any],
    full_config: Optional[Dict[str, Any]] = None
) -> PreTrainedModel:
    """
    Converts a dense model to a sparse model by patching attention modules.
    
    Args:
        model: The HuggingFace model to convert.
        sparse_mod: The type of sparse attention to use (e.g., 'sparsex').
        sparse_config: Configuration for the sparse attention module.
        
    Returns:
        The patched model.
    """
    # Ensure modules are registered by importing them
    from .. import attn 

    if sparse_mod not in SPARSE_ATTENTION_REGISTRY:
        raise ValueError(f"Sparse module '{sparse_mod}' not found in registry. "
                         f"Available modules: {list(SPARSE_ATTENTION_REGISTRY.keys())}")

    sparse_cls = SPARSE_ATTENTION_REGISTRY[sparse_mod]
    
    print(f"Converting model to {sparse_mod} with config: {sparse_config}")
    
    for name, module in model.named_modules():
        # Common names for attention modules in HF models
        if any(attn_name in name for attn_name in ["self_attn", "attention"]):
            # Check if it's a leaf attention module
            if hasattr(module, "q_proj") or hasattr(module, "query"):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Use from_dense if available to inherit weights/parameters, 
                # otherwise try to instantiate from scratch (random initialization)
                # TODO: make the logic more clear
                if hasattr(sparse_cls, "from_dense"):
                    sparse_module = sparse_cls.from_dense(module, sparse_config)
                else:
                    sparse_module = sparse_cls(full_config)
                    
                setattr(parent, child_name, sparse_module)
                print(f"Patched {name} with config: {sparse_config}.")
            
    return model
