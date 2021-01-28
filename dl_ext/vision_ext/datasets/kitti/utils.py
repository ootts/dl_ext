import numpy as np
import torch


def inverse_rigid_trans(Tr):
    """
     Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def check_type(arr):
    if isinstance(arr, np.ndarray):
        pass
    elif isinstance(arr, torch.Tensor):
        assert arr.dtype == torch.float32
    else:
        raise TypeError('arr must be a numpy array or a pytorch float tensor.')
