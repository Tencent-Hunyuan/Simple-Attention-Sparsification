import torch


def ceildiv(a, b):
    return (a + b - 1) // b


def maybe_contiguous(x: torch.Tensor): # make sure the last dimension (usually head dim) is contiguous
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x
