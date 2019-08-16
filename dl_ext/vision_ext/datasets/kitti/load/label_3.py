import os
from multipledispatch import dispatch

@dispatch(str)
def load_label_3(absolute_path):
    with open(absolute_path) as f:
        lines = f.read().splitlines()
    labels = []
    for l in lines:
        items = l.split()
        cls = items[0]
        truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry = map(float, items[1:])
        label = {'cls': cls,
                 'truncated': truncated, 'occluded': occluded, 'alpha': alpha,
                 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                 'h': h, 'w': w, 'l': l,
                 'x': x, 'y': y, 'z': z, 'ry': ry}
        labels.append(label)
    return labels


@dispatch(str, str)
def load_label_3(label_3_dir, imgid):
    absolute_path = os.path.join(label_3_dir, imgid + '.txt')
    return load_label_3(absolute_path)


@dispatch(str, int)
def load_label_3(label_3_dir, imgid):
    imgid = '%06d' % imgid
    return load_label_3(label_3_dir, imgid)


@dispatch(str, str, str)
def load_label_3(kitti_root, split, imgid):
    calib_dir = os.path.join(kitti_root, 'object', split, 'label_3')
    return load_label_3(calib_dir, imgid)


@dispatch(str, str, int)
def load_label_3(kitti_root, split, imgid):
    imgid = '%06d' % imgid
    return load_label_3(kitti_root, split, imgid)
