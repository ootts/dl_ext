from warnings import warn
import numpy as np
import torch
from torch import Tensor


def area(box):
    TO_REMOVE = 1
    area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
    return area
