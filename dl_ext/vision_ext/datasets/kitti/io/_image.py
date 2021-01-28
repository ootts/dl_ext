import os.path as osp
import os
from typing import Union

from PIL import Image


def _load_image_i(kitti_root: str, split: str, imgid: Union[str, int], view: int) -> Image.Image:
    assert view in (0, 1, 2, 3)
    if not isinstance(imgid, str):
        imgid = '%06d' % imgid
    image_dir = os.path.join(kitti_root, 'object', split, f'image_{view}')
    img = Image.open(osp.join(image_dir, imgid + '.png'))
    return img
