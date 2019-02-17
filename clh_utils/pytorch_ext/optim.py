from torch.optim.optimizer import Optimizer


def get_lr(optimizer):
    assert isinstance(optimizer, Optimizer), 'optimizer must be an instance of Optimizer!'
    for param_group in optimizer.param_groups:
        return param_group['lr']
