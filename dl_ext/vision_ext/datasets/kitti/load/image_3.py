import os
from multipledispatch import dispatch
import numpy as np
import imageio


@dispatch(str)
def load_image_3(absolute_path):
    return np.array(imageio.imread(absolute_path))


@dispatch(str, str)
def load_image_3(image_3_dir, imgid):
    absolute_path = os.path.join(image_3_dir, imgid + '.png')
    return load_image_3(absolute_path)


@dispatch(str, int)
def load_image_3(image_3_dir, imgid):
    imgid = '%06d' % imgid
    return load_image_3(image_3_dir, imgid)


@dispatch(str, str, str)
def load_image_3(kitti_root, split, imgid):
    calib_dir = os.path.join(kitti_root, 'object', split, 'image_3')
    return load_image_3(calib_dir, imgid)


@dispatch(str, str, int)
def load_image_3(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    return load_image_3(kitti_root, split, imgid)
