import torch

from clh_utils.cv_utils import *
from torchvision.transforms import transforms

# print(conv_size_out(28, 2, 2))
img = np.random.rand(2, 2, 3)
transformed = imagenet_normalize(torch.from_numpy(img).clone().permute(2, 0, 1))
reverted = imagnet_revert_normalize(transformed)
assert np.allclose(img,
                   reverted.permute(1, 2, 0).numpy())
