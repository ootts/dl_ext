from warnings import warn

import math
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms


def conv_size_out(size, kernel_size=5, stride=2, pad=0):
    return int(math.floor((size + 2 * pad - kernel_size) / stride + 1))


imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])


def imagnet_revert_normalize(rgb):
    '''

     :param rgb: [b,3,h,w]
     :return:
     '''
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    if isinstance(rgb, Tensor):
        if rgb.ndimension() == 3:
            rgb = rgb.unsqueeze(0)
        std = Tensor(std).type_as(rgb)
        mean = Tensor(mean).type_as(rgb)
        rgb = rgb * std[None, :, None, None] + mean[None, :, None, None]
        rgb = rgb.squeeze(0)
        return rgb
    elif isinstance(rgb, np.ndarray):
        if rgb.ndim == 3:
            rgb = rgb[np.newaxis, :, :, :]
        std = np.asarray(std)
        mean = np.asarray(mean)
        rgb = rgb * std[None, None, None, :] + mean[None, None, None, :]
        rgb = rgb.squeeze(0)
        return rgb
    else:
        raise TypeError('expect type of arg0 Tensor or ndarray, but found', type(rgb))


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
