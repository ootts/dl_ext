from collections import OrderedDict
from enum import IntEnum
from typing import List, Collection

from torch import nn
from torch.nn import ModuleList

from .dist import get_rank


def to_cuda(x):
    if hasattr(x, 'cuda'):
        return x.cuda(device=get_rank())
    elif isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}


def to_cpu(x):
    if hasattr(x, 'cpu'):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}


def batch_gpu(batch):
    x, y = batch
    return to_cuda(x), to_cuda(y)


def format_time(t):
    t = int(t)
    h, m, s = t // 3600, (t // 60) % 60, t % 60
    if h != 0:
        return f'{h}:{m:02d}:{s:02d}'
    else:
        return f'{m:02d}:{s:02d}'


class TrainerState(IntEnum):
    BASE = 1
    PARALLEL = 2
    DISTRIBUTEDPARALLEL = 3


def split_list(vals, skip_start: int, skip_end: int):
    return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]


class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."

    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x):
        return x


def children_and_parameters(m: nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()], [])
    for p in m.parameters():
        if id(p) not in children_p:
            children.append(ParameterModule(p))
    return children


def children(m: nn.Module) -> ModuleList:
    "Get children of `m`."
    return list(m.children())


def num_children(m: nn.Module) -> int:
    "Get number of children modules in `m`."
    return len(children(m))


flatten_model = lambda m: sum(map(flatten_model, children_and_parameters(m)), []) if num_children(m) else [m]

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
no_wd_types = bn_types + (nn.LayerNorm,)
bias_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def trainable_params(m: nn.Module) -> Collection[nn.Parameter]:
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


def split_no_wd_params(layer_groups: Collection[nn.Module]) -> List[List[nn.Parameter]]:
    "Separate the parameters in `layer_groups` between `no_wd_types` and  bias (`bias_types`) from the rest."
    split_params = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in l.children():
            if isinstance(c, no_wd_types):
                l2 += list(trainable_params(c))
            elif isinstance(c, bias_types):
                bias = c.bias if hasattr(c, 'bias') else None
                l1 += [p for p in trainable_params(c) if not (p is bias)]
                if bias is not None: l2.append(bias)
            else:
                l1 += list(trainable_params(c))
        # Since we scan the children separately, we might get duplicates (tied weights). We need to preserve the order
        # for the optimizer load of state_dict
        l1, l2 = list(OrderedDict.fromkeys(l1).keys()), list(OrderedDict.fromkeys(l2).keys())
        split_params += [l1, l2]
    return split_params
