import argparse

import torch
from torch.distributed import get_rank
from torchvision.models import resnet18

parser = argparse.ArgumentParser()
from torch.nn.parallel import DistributedDataParallel

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()
net = resnet18().to(torch.device(args.local_rank))

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(
    backend="nccl", init_method="env://"
)
local_rank = get_rank()
net = DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, )
print(local_rank, net.device_ids, next(net.parameters()).device)
