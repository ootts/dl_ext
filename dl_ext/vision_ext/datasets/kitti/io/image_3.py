from typing import Union

from PIL import Image
from ._image import _load_image_i


def load_image_3(kitti_root: str, split: str, imgid: Union[str, int]) -> Image.Image:
    return _load_image_i(kitti_root, split, imgid, 3)
