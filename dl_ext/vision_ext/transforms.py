from torch import Tensor
from torchvision import transforms
import numpy as np

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
