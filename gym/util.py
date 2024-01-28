import torch, random
import numpy as np
from operator import itemgetter
from torch.utils.data import default_collate
from .device import def_device

def set_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f

def collate_dict(ds):
    get = itemgetter(*ds.features)
    def _f(b): return get(default_collate(b))
    return _f

def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f

def conv(ni, nf, ks=3, stride=2, act=torch.nn.ReLU, norm=None, bias=None):
    is_batch_norm = isinstance(norm, (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d
    ))
    # the bias is redundant when doing batch normalization as this is captured by the 'shift'
    if bias is None: bias = not is_batch_norm
    layers = [torch.nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2, bias=bias)]
    if norm: layers.append(norm(nf))
    if act: layers.append(act())
    return torch.nn.Sequential(*layers)

def get_model(act=torch.nn.ReLU, nfs=None, norm=None):
    if nfs is None: nfs = [1,8,16,32,64]
    layers = [conv(nfs[i], nfs[i+1], act=act, norm=norm) for i in range(len(nfs)-1)]
    return torch.nn.Sequential(*layers, conv(nfs[-1],10, act=None, norm=False, bias=True),
                         torch.nn.Flatten()).to(def_device)