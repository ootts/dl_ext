import torch
from torch.optim import Adam
from torchvision.models.resnet import resnet18

net = resnet18()
optim = Adam(net.parameters(), 1e-2)
