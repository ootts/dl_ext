import os
from typing import Union

from ..structures.kitti_object_3d import KITTIObjectClass, KITTIObject3D

default_interest_classes = list(map(int, KITTIObjectClass))


def _load_label_i(kitti_root: str, split: str, imgid: Union[str, int], view: int, interest_classes=None):
    """
    Load label_view
    :param kitti_root:
    :param split:
    :param imgid:
    :param view:
    :param interest_classes: default all classes, including dontcare
    :return:
    """
    assert view in (2, 3)
    label_i_dir = os.path.join(kitti_root, 'object', split, f'label_{view}')
    if not isinstance(imgid, str):
        imgid = '%06d' % imgid
    absolute_path = os.path.join(label_i_dir, imgid + '.txt')
    with open(absolute_path) as f:
        lines = f.read().splitlines()
    labels = []
    for l in lines:
        items = l.split()
        cls = items[0]
        truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry = map(float, items[1:])
        label = KITTIObject3D(KITTIObjectClass[cls], truncated, occluded, alpha,
                              x1, y1, x2, y2, h, w, l, x, y, z, ry)
        labels.append(label)
    if interest_classes is not None:
        for i, ic in enumerate(interest_classes):
            if isinstance(ic, str):
                interest_classes[i] = KITTIObjectClass[ic]

        labels = list(filter(lambda x: x.cls in interest_classes, labels))
    return labels
