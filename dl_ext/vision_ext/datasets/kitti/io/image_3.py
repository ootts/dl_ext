import os.path as osp
import os
from multipledispatch import dispatch
from PIL import Image


@dispatch(str, str, str)
def load_image_3(kitti_root, split, imgid):
    image_dir = os.path.join(kitti_root, 'object', split, 'image_3')
    img = Image.open(osp.join(image_dir, imgid + '.png'))
    return img


@dispatch(str, str, int)
def load_image_3(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    return load_image_3(kitti_root, split, imgid)
