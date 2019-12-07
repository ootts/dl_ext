import os

import torch
from torch.optim import Adam
from tqdm import tqdm
from .model import save_model
from .optim import OneCycleScheduler
from ..average_meter import AverageMeter
from tensorboardX import SummaryWriter


def to_cuda(x):
    if hasattr(x, 'cuda'):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}


def batch_gpu(batch):
    x, y = batch
    return to_cuda(x), to_cuda(y)


class BaseTrainer:

    def __init__(self, model, dataloaders, num_epochs,
                 loss_function, optimizer=None, scheduler=None, begin_epoch=0,
                 output_dir='models', max_lr=1e-2, save_every=False,
                 metric_functions=None):
        self.loss_function = loss_function
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        if optimizer is None:
            optimizer = Adam(model.parameters(), lr=max_lr)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.begin_epoch = begin_epoch
        self.max_lr = max_lr
        self.save_every = save_every
        if scheduler is None:
            scheduler = OneCycleScheduler(self.optimizer, self.max_lr,
                                          total_steps=len(dataloaders['train']) * num_epochs)
        self.scheduler = scheduler
        self.model = model
        if metric_functions is None:
            metric_functions = {}
        self.metric_functions = metric_functions
        self.tb_writer = SummaryWriter(output_dir, flush_secs=20)
        self.global_steps = 0

    def train(self, epoch):
        loss_meter = AverageMeter()
        metric_ams = {}
        for metric in self.metric_functions.keys():
            metric_ams[metric] = AverageMeter()
        self.model.train()
        bar = tqdm(self.dataloaders['train'])
        for batch in bar:
            self.optimizer.zero_grad()
            x, y = batch_gpu(batch)
            output = self.model(x)
            loss = self.loss_function(output, y)
            loss = loss.mean()
            loss_meter.update(loss.item())
            lr = self.optimizer.param_groups[0]['lr']
            self.tb_writer.add_scalar('train/loss', loss.item(), self.global_steps)
            self.tb_writer.add_scalar('train/lr', lr, self.global_steps)
            metrics = {}
            for metric, f in self.metric_functions.items():
                s = f(output, y).mean().item()
                metric_ams[metric].update(s)
                metrics[metric] = metric_ams[metric].avg
                self.tb_writer.add_scalar(f'train/{metric}', s, self.global_steps)
            loss.backward()
            self.optimizer.step()
            if isinstance(self.scheduler, OneCycleScheduler):
                self.scheduler.step()
            bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg, 'lr': lr}
            for k, v in metrics.items():
                bar_vals[k] = v
            bar.set_postfix(bar_vals)
            self.global_steps += 1
        if not isinstance(self.scheduler, OneCycleScheduler):
            self.scheduler.step()

    def val(self, epoch):
        loss_meter = AverageMeter()
        metric_ams = {}
        for metric in self.metric_functions.keys():
            metric_ams[metric] = AverageMeter()
        self.model.eval()
        bar = tqdm(self.dataloaders['val'])
        with torch.no_grad():
            for batch in bar:
                x, y = batch_gpu(batch)
                output = self.model(x)
                loss = self.loss_function(output, y)
                loss = loss.mean()
                loss_meter.update(loss.item())
                metrics = {}
                for metric, f in self.metric_functions.items():
                    s = f(output, y).mean().item()
                    metric_ams[metric].update(s)
                    metrics[metric] = metric_ams[metric].avg
                bar_vals = {'epoch': epoch, 'phase': 'val', 'loss': loss_meter.avg}
                for k, v in metrics.items():
                    bar_vals[k] = v
                bar.set_postfix(bar_vals)
            self.tb_writer.add_scalar('val/loss', loss_meter.avg, epoch)
            for metric, s in metric_ams.items():
                self.tb_writer.add_scalar(f'val/{metric}', s.avg, epoch)
            return loss_meter.avg

    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        best_val_loss = 1000
        num_epochs = self.num_epochs
        for epoch in range(self.begin_epoch, num_epochs):
            self.train(epoch)
            val_loss = self.val(epoch)
            if self.save_every:
                save_model(self.model, self.optimizer, self.scheduler,
                           epoch, self.output_dir)
            elif val_loss < best_val_loss:
                print('Epoch %d, save better model.' % epoch)
                best_val_loss = val_loss
                save_model(self.model, self.optimizer, self.scheduler,
                           epoch, self.output_dir)
        print('Training finished.')
