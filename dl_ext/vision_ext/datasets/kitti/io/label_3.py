from typing import Union

from ._label import _load_label_i


def load_label_3(kitti_root: str, split: str, imgid: Union[str, int], interest_classes=None):
    return _load_label_i(kitti_root, split, imgid, 3, interest_classes)
