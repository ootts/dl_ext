import numpy as np
import torch
from torch import Tensor


def pad(a, max_len, value):
    if isinstance(a, list):
        if max_len > len(a):
            return a + [value] * (max_len - len(a))
        else:
            return a[:max_len]
    elif isinstance(a, np.ndarray):
        return np.array(pad(a.tolist(), max_len, value))
    elif isinstance(a, Tensor):
        if a.is_cuda:
            ndarr = a.detach().cpu().numpy()
        else:
            ndarr = a.numpy()
        return torch.from_numpy(pad(ndarr, max_len, value))
    else:
        raise RuntimeError('can not pad')
