import numpy as np


def pad(a, max_len, value):
    if isinstance(a, list):
        if max_len > len(a):
            return a + [value] * (max_len - len(a))
        else:
            return a[:max_len]
    elif isinstance(a, np.ndarray):
        return np.array(pad(a.tolist(), max_len, value))
    else:
        raise RuntimeError('can not pad')
