import os
from multipledispatch import dispatch
from ..kitti_object_3d import KITTIObject3D, KITTIObjectClass

default_interest_classes = list(map(int, KITTIObjectClass))


@dispatch(str, list)
def load_label_3(absolute_path, interest_classes=default_interest_classes):
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


@dispatch(str, str, list)
def load_label_3(label_3_dir, imgid, interest_classes=default_interest_classes):
    absolute_path = os.path.join(label_3_dir, imgid + '.txt')
    return load_label_3(absolute_path, interest_classes)


@dispatch(str, int, list)
def load_label_3(label_3_dir, imgid, interest_classes=None):
    imgid = '%06d' % imgid
    return load_label_3(label_3_dir, imgid, interest_classes)


@dispatch(str, str, str, list)
def load_label_3(kitti_root, split, imgid, interest_classes=default_interest_classes):
    calib_dir = os.path.join(kitti_root, 'object', split, 'label_3')
    return load_label_3(calib_dir, imgid, interest_classes)


@dispatch(str, str, int, list)
def load_label_3(kitti_root, split, imgid, interest_classes=default_interest_classes):
    imgid = '%06d' % imgid
    return load_label_3(kitti_root, split, imgid, interest_classes)


@dispatch(str)
def load_label_3(absolute_path):
    return load_label_3(absolute_path, default_interest_classes)


@dispatch(str, str)
def load_label_3(label_3_dir, imgid):
    return load_label_3(label_3_dir, imgid, default_interest_classes)


@dispatch(str, int)
def load_label_3(label_3_dir, imgid):
    return load_label_3(label_3_dir, imgid, default_interest_classes)


@dispatch(str, str, str)
def load_label_3(kitti_root, split, imgid):
    return load_label_3(kitti_root, split, imgid, default_interest_classes)


@dispatch(str, str, int)
def load_label_3(kitti_root, split, imgid):
    return load_label_3(kitti_root, split, imgid, default_interest_classes)
