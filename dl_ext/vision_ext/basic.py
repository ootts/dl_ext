from warnings import warn
import numpy as np
import torch
from torch import Tensor


def area(box):
    TO_REMOVE = 1
    area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
    return area


def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      boxlist1: bounding boxes, sized [N,4].
      boxlist2: bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if isinstance(boxlist1, np.ndarray):
        return boxlist_iou(torch.from_numpy(boxlist1), boxlist2)
    if isinstance(boxlist2, np.ndarray):
        return boxlist_iou(boxlist1, torch.from_numpy(boxlist2))
    assert boxlist1.shape[1] == 4 and boxlist2.shape[1] == 4, 'shape[1] must be 4'
    if boxlist1.dtype != torch.float:
        warn(f'cast boxlist1 from {boxlist1.dtype} to float')
        boxlist1 = boxlist1.type(torch.float)
    if boxlist2.dtype != torch.float:
        warn(f'cast boxlist2 from {boxlist2.dtype} to float')
        boxlist2 = boxlist2.type(torch.float)

    area1 = area(boxlist1)
    area2 = area(boxlist2)

    box1, box2 = boxlist1, boxlist2

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
