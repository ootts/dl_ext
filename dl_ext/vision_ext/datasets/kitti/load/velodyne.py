import os
from multipledispatch import dispatch
import numpy as np


@dispatch(str)
def load_velodyne(absolute_path) -> np.ndarray:
    return np.fromfile(absolute_path, dtype=np.float32).reshape(-1, 4)


@dispatch(str, str)
def load_velodyne(velodyne_dir, imgid):
    absolute_path = os.path.join(velodyne_dir, imgid + '.bin')
    return load_velodyne(absolute_path)


@dispatch(str, int)
def load_velodyne(velodyne_dir, imgid):
    imgid = '%06d' % imgid
    return load_velodyne(velodyne_dir, imgid)


@dispatch(str, str, str)
def load_velodyne(kitti_root, split, imgid):
    velodyne_dir = os.path.join(kitti_root, 'object', split, 'velodyne')
    return load_velodyne(velodyne_dir, imgid)


@dispatch(str, str, int)
def load_velodyne(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    return load_velodyne(kitti_root, split, imgid)
