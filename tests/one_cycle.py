import torch
from torch import nn
from torch.optim import SGD
from torchvision.models import resnet18
from tqdm import trange

from dl_ext.pytorch_ext.optim import OneCycleScheduler
from tensorboardX import SummaryWriter

tb_writer = SummaryWriter('./tests/log')
net = resnet18().cuda()
optim = SGD(net.parameters(), lr=0.01, momentum=0.95)
total_steps = 1000
scheduler = OneCycleScheduler(optim, max_lr=1e-2, total_steps=total_steps)
loss_fn = nn.CrossEntropyLoss()
for it in trange(total_steps):
    scheduler.step()
    input = torch.randn(2, 3, 224, 224).cuda()
    target = torch.zeros(2).long().cuda()
    output = net(input)
    loss = loss_fn(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    tb_writer.add_scalar('test/lr', optim.param_groups[0]['lr'], it)
    tb_writer.add_scalar('test/mom', optim.param_groups[0]['momentum'], it)
