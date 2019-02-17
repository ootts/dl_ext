import os

import torch

from myutils.history import History


def load_model(model, optim, scheduler, model_dir, for_train, load_optim, load_scheduler, epoch=-1,
               history: History = None):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if pth.endswith('.pth')]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('Loading from epoch', pth)
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    if hasattr(model, 'module'):
        print('Loading module...')
        model.module.load_state_dict(pretrained_model['net'])
    else:
        print('Loading model...')
        model.load_state_dict(pretrained_model['net'])
    if for_train and load_optim:
        print('Loading optimizer...')
        optim.load_state_dict(pretrained_model['optim'])
    if for_train and load_scheduler:
        print('Loading scheduler...')
        scheduler.load_state_dict(pretrained_model['scheduler'])
    if history is not None:
        print('Loading history...')
        history.load_state_dict(pretrained_model['history'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, epoch, model_dir, cfg, history=None):
    os.system('mkdir -p {}'.format(model_dir))
    obj = {
        'net': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    if history is not None:
        print('Saving history...')
        obj['history'] = history.state_dict()
    torch.save(obj, os.path.join(model_dir, '{}.pth'.format(epoch)))
