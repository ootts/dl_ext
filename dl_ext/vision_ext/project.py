from numbers import Number
from warnings import warn

import numpy as np
import torch
from torch import Tensor
from multipledispatch import dispatch


def project_K(pts3d, K, pose):
    """
    :param pts3d: shape(n,3)
    :param K: pose: shape(3,3)
    :param pose: shape(3,4)
    :return:
    """
    p = pts3d.t()
    n = p.shape[1]
    p = K @ pose @ torch.cat([p, torch.ones(1, n)])
    p[0] = p[0] / p[2]
    p[1] = p[1] / p[2]
    pts2d = p[0:2]
    pts2d = pts2d.t()
    return pts2d


def project_P2(pts3d, P2, pose):
    """
    project using P2, mainly for kitti projection.
    :param pts3d: shape(n,3)
    :param P2: shape(3,4)
    :param pose: shape(3,4)
    :return:
    """
    p = pts3d.t()  # 3,n
    n = pts3d.shape[0]
    p = torch.cat([p, torch.ones(1, n)])
    p = pose @ p  # 3,14
    p = torch.cat([p, torch.ones(1, n)])
    p = P2 @ p  # 3,14
    p[0] = p[0] / p[2]
    p[1] = p[1] / p[2]
    pts2d = p[0:2]
    pts2d = pts2d.t()
    return pts2d


def pts3d_induction(pts3d):
    pts3d_shape_error_msg = f'pts3d must have shape (3,), (*,3) or (3,*), but got {pts3d.shape}'
    if pts3d.ndimension() == 1:
        assert pts3d.shape == (3,), pts3d_shape_error_msg
    elif pts3d.ndimension() == 2:
        if pts3d.shape == (3, 3):
            warn('pts3d got shape (3,3), cannot perform auto induction, using each row as a point.')
        elif pts3d.shape[0] == 3:
            pts3d = pts3d.t()
        elif pts3d.shape[1] == 3:
            pass
        else:
            raise ValueError(pts3d_shape_error_msg)
    else:
        raise ValueError(pts3d_shape_error_msg)
    return pts3d


@dispatch((np.ndarray, Tensor), (np.ndarray, Tensor), (np.ndarray, Tensor))
def project(pts3d, intrinsic_matrix, pose, return_type=torch.Tensor):
    """
    :param pts3d:
    :param intrinsic_matrix:
    :param pose:
    :param return_type: torch.Tensor or np.ndarray
    :return:
    """
    pts3d = torch.as_tensor(pts3d).float()
    intrinsic_matrix = torch.as_tensor(intrinsic_matrix).float()
    pose = torch.as_tensor(pose).float()
    pts3d = pts3d_induction(pts3d)
    assert pose.shape == (3, 4), f'pose must have shape (3,4), got {pose.shape}'
    if intrinsic_matrix.shape == (3, 3):
        result = project_K(pts3d, intrinsic_matrix, pose)
    elif intrinsic_matrix.shape == (3, 4):
        result = project_P2(pts3d, intrinsic_matrix, pose)
    else:
        raise ValueError(f'intrinsic matrix must have shape (3,3) or (3,4), got {intrinsic_matrix.shape}')
    if return_type == torch.Tensor:
        return torch.Tensor(result)
    elif return_type == np.ndarray:
        return np.array(result)
    else:
        return ValueError(f'return_type must be torch.Tensor or numpy.ndarray, got {return_type}')


@dispatch((np.ndarray, Tensor), (np.ndarray, Tensor), (np.ndarray, Tensor), (np.ndarray, Tensor))
def project(pts3d, intrinsic_matrix, R, t, return_type=torch.Tensor):
    """
    :param return_type:
    :param pts3d:
    :param intrinsic_matrix:
    :param pose:
    :return:
    """
    R = torch.as_tensor(R).float()
    t = torch.as_tensor(t).float()
    assert R.shape == (3, 3), f'R.shape {R.shape} != (3,3)'
    if t.shape == (3,):
        t = t.unsqueeze(-1)
    assert t.shape == (3, 1), f't.shape {t.shape} != (3,1) or (3,)'
    pose = torch.cat([R, t], dim=1)
    assert pose.shape == (3, 4)
    return project(pts3d, intrinsic_matrix, pose, return_type=return_type)


@dispatch((np.ndarray, Tensor), (np.ndarray, Tensor), Number, Number, Number, (np.ndarray, Tensor))
def project(pts3d, intrinsic_matrix, rx, ry, rz, t, return_type=torch.Tensor):
    Rx = torch.as_tensor(np.array([[1, 0, 0],
                                   [0, np.cos(rx), -np.sin(rx)],
                                   [0, np.sin(rx), np.cos(rx)]]))
    Ry = torch.as_tensor(np.array([[np.cos(ry), 0, np.sin(ry)],
                                   [0, 1, 0],
                                   [-np.sin(ry), 0, np.cos(ry)]]))
    Rz = torch.as_tensor(np.array([[np.cos(rz), -np.sin(rz), 0],
                                   [np.sin(rz), np.cos(rz), 0],
                                   [0, 0, 1]]))
    R = Rx @ Ry @ Rz
    return project(pts3d, intrinsic_matrix, R, t, return_type=return_type)


@dispatch((np.ndarray, Tensor), (np.ndarray, Tensor), Number, Number, Number, Number, Number, Number)
def project(pts3d, intrinsic_matrix, rx, ry, rz, tx, ty, tz, return_type=torch.Tensor):
    """
    :param return_type:
    :param pts3d:
    :param intrinsic_matrix:
    :param pose:
    :return:
    """
    return project(pts3d, intrinsic_matrix, rx, ry, rz, torch.Tensor([[tx, ty, tz]]).t(), return_type=return_type)
