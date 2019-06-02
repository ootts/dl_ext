import os
from multipledispatch import dispatch
import numpy as np
import imageio


@dispatch(str)
def load_image_2(absolute_path):
    """
    :param absolute_path:
    :return: rgb
    """
    rgb = np.array(imageio.imread(absolute_path))
    return rgb


@dispatch(str, str)
def load_image_2(image_2_dir, imgid):
    absolute_path = os.path.join(image_2_dir, imgid + '.png')
    return load_image_2(absolute_path)


@dispatch(str, int)
def load_image_2(image_2_dir, imgid):
    imgid = '%06d' % imgid
    return load_image_2(image_2_dir, imgid)


@dispatch(str, str, str)
def load_image_2(kitti_root, split, imgid):
    calib_dir = os.path.join(kitti_root, 'object', split, 'image_2')
    return load_image_2(calib_dir, imgid)


@dispatch(str, str, int)
def load_image_2(kitti_root, split, imgid):
    """
    :param kitti_root:
    :param split: training or testing
    :param imgid: int
    :return: rgb
    """
    imgid = '%06d' % imgid
    return load_image_2(kitti_root, split, imgid)
