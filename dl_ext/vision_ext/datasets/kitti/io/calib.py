import os.path as osp
import os
from typing import Union

import numpy as np
from ..structures.calib import Calibration
from .image_info import load_image_info


def load_calib(kitti_root: str, split: str, imgid: Union[str, int]):
    if not isinstance(imgid, str):
        imgid = '%06d' % imgid
    calib_dir = os.path.join(kitti_root, 'object', split, 'calib')
    absolute_path = osp.join(calib_dir, imgid + '.txt')
    with open(absolute_path) as f:
        lines = {line.strip().split(':')[0]: list(map(float, line.strip().split(':')[1].split())) for line in
                 f.readlines()[:-1]}
    calibs = {'P0': np.array(lines['P0']).reshape((3, 4)),
              'P1': np.array(lines['P1']).reshape((3, 4)),
              'P2': np.array(lines['P2']).reshape((3, 4)),
              'P3': np.array(lines['P3']).reshape((3, 4)),
              'R0_rect': np.array(lines['R0_rect']).reshape((3, 3)),
              'Tr_velo_to_cam': np.array(lines['Tr_velo_to_cam']).reshape((3, 4)),
              'Tr_imu_to_velo': np.array(lines['Tr_imu_to_velo']).reshape((3, 4))}
    H, W, _ = load_image_info(kitti_root, split, imgid)
    image_size = (W, H)
    return Calibration(calibs, image_size)
