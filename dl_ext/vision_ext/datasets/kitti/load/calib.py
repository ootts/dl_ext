import os
from multipledispatch import dispatch
import numpy as np


@dispatch(str)
def load_calib(absolute_path):
    with open(absolute_path) as f:
        lines = {line.strip().split(':')[0]: list(map(float, line.strip().split(':')[1].split())) for line in
                 f.readlines()[:-1]}
    calibs = {}
    calibs['P0'] = np.array(lines['P0']).reshape((3, 4))
    calibs['P1'] = np.array(lines['P1']).reshape((3, 4))
    calibs['P2'] = np.array(lines['P2']).reshape((3, 4))
    R0_rect = np.eye(4)
    R0_rect[0:3, 0:3] = np.array(lines['R0_rect']).reshape((3, 3))
    calibs['R0_rect'] = R0_rect
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[0:3, :] = np.array(lines['Tr_velo_to_cam']).reshape((3, 4))
    calibs['Tr_velo_to_cam'] = Tr_velo_to_cam
    Tr_imu_to_velo = np.eye(4)
    Tr_imu_to_velo[0:3, :] = np.array(lines['Tr_imu_to_velo']).reshape((3, 4))
    calibs['Tr_imu_to_velo'] = Tr_imu_to_velo
    return calibs


@dispatch(str, str)
def load_calib(calib_dir, imgid):
    absolute_path = os.path.join(calib_dir, imgid + '.txt')
    return load_calib(absolute_path)


@dispatch(str, int)
def load_calib(calib_dir, imgid):
    imgid = '%06d' % imgid
    return load_calib(calib_dir, imgid)


@dispatch(str, str, str)
def load_calib(kitti_root, split, imgid):
    calib_dir = os.path.join(kitti_root, 'object', split, 'calib')
    return load_calib(calib_dir, imgid)


@dispatch(str, str, int)
def load_calib(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    return load_calib(kitti_root, split, imgid)
