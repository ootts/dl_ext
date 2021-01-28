import os.path as osp
import os
from multipledispatch import dispatch
import numpy as np


@dispatch(str, str, str)
def load_velodyne(kitti_root, split, imgid):
    velodyne_dir = os.path.join(kitti_root, 'object', split, 'velodyne')
    velodyne = np.fromfile(osp.join(velodyne_dir, imgid + '.bin'), dtype=np.float32).reshape(-1, 4)
    return velodyne


@dispatch(str, str, int)
def load_velodyne(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    return load_velodyne(kitti_root, split, imgid)
