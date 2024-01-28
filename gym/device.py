# gym/device_utils.py

import torch
from collections.abc import Mapping

# Determine the default device based on availability
def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def to_device(x, device=def_device):
    """
    Move the input `x` to the specified device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(to_device(o, device) for o in x)

def to_cpu(x):
    """
    Move the input `x` to the CPU and convert to float if necessary.
    """
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    
    res = x.detach().cpu()
    return res.float() if res.dtype == torch.float16 else res
