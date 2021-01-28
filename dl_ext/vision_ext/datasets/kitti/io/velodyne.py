import os.path as osp
import os
import numpy as np


def load_velodyne(kitti_root, split, imgid):
    if not isinstance(imgid, str):
        imgid = '%06d' % imgid
    velodyne_dir = os.path.join(kitti_root, 'object', split, 'velodyne')
    velodyne = np.fromfile(osp.join(velodyne_dir, imgid + '.bin'), dtype=np.float32).reshape(-1, 4)
    return velodyne
