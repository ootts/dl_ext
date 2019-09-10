import torch

from dl_ext.vision_ext.models.resnet import resnet18

net = resnet18(pretrained=True)
input = torch.randn(2, 3, 448, 448)
print(net(input).shape)
