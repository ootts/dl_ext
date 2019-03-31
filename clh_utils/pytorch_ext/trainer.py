from __future__ import absolute_import
from __future__ import print_function

import os
from os import path as osp

import matplotlib.pyplot as plt
import progressbar
import torch
from ..history import History
from ..average_meter import AverageMeter
from .model import save_model
from .optim import get_lr


class BaseTrainer:

    def __init__(self, model, optimizer, dataloaders, metrics_functions=None, scheduler=None, num_epochs=-1,
                 begin_epoch=0,
                 save_model_dir='data/models', history=None,
                 min_lr=1e-9, save_every=True,
                 allow_keyboard_interrupt=False, log_dir='data/log') -> None:
        super().__init__()
        if metrics_functions is None:
            metrics_functions = {}
        self.metrics_functions = metrics_functions
        self.dataloaders = dataloaders
        self.history = history
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.begin_epoch = begin_epoch
        self.save_model_dir = save_model_dir
        self.min_lr = min_lr
        self.allow_keyboard_interrupt = allow_keyboard_interrupt
        self.save_every = save_every
        self.scheduler = scheduler
        self.model = model

    def train(self, epoch, history):
        meters = {'loss': AverageMeter()}
        for k in self.metrics_functions.keys():
            meters[k] = AverageMeter()
        self.model.train()
        widgets = [
            progressbar.Bar(),
            progressbar.Percentage(),
            progressbar.ETA(),
            progressbar.DynamicMessage('loss'),
        ]
        widgets += [progressbar.DynamicMessage(k)
                    for k in self.metrics_functions.keys()]
        with progressbar.ProgressBar(max_value=len(self.dataloaders['train']), widgets=widgets) as bar:
            for it, batch in enumerate(self.dataloaders['train']):
                self.optimizer.zero_grad()
                loss, scores = self.model(*batch)
                loss = loss.mean()
                meters['loss'].update(loss.item())
                history.records['loss']['train'].append(loss.item())
                for k, f in self.metrics_functions.items():
                    result = f.apply(batch, scores)
                    meters[k].update(result)
                    history.records[k]['train'].append(result)
                loss.backward()
                self.optimizer.step()
                bar_values = {'loss': meters['loss'].avg}
                for k in self.metrics_functions.keys():
                    bar_values[k] = meters[k].avg

                bar.update(it, **bar_values)
            s = 'Epoch {}, train, loss = {:.4f}'.format(
                epoch + 1, meters['loss'].avg)
            for k in self.metrics_functions.keys():
                s += ', {} = {:.4f}'.format(k, meters[k].avg)
            print(s)

    def val(self, epoch, history):
        meters = {'loss': AverageMeter()}
        for k in self.metrics_functions.keys():
            meters[k] = AverageMeter()
        self.model.eval()
        widgets = [
            progressbar.Bar(),
            progressbar.Percentage(),
            progressbar.ETA(),
            progressbar.DynamicMessage('loss'),
        ]
        widgets += [progressbar.DynamicMessage(k)
                    for k in self.metrics_functions.keys()]
        with progressbar.ProgressBar(max_value=len(self.dataloaders['val']), widgets=widgets) as bar:
            with torch.no_grad():
                for it, batch in enumerate(self.dataloaders['val']):
                    self.optimizer.zero_grad()
                    loss, scores = self.model(*batch)
                    loss = loss.mean()
                    meters['loss'].update(loss.item())
                    history.records['loss']['val'].append(loss.item())
                    for k, f in self.metrics_functions.items():
                        result = f.apply(batch, scores)
                        meters[k].update(result)
                        history.records[k]['val'].append(result)
                    bar_values = {'loss': meters['loss'].avg}
                    for k in self.metrics_functions.keys():
                        bar_values[k] = meters[k].avg
                    bar.update(it, **bar_values)
                s = 'Epoch {}, val, loss = {:.4f}'.format(
                    epoch + 1, meters['loss'].avg)
                for k in self.metrics_functions.keys():
                    s += ', {} = {:.4f}'.format(k, meters[k].avg)
                print(s)
                self.scheduler.step(meters['loss'].avg)
                return meters['loss'].avg

    def fit(self):
        os.makedirs(self.save_model_dir, exist_ok=True)
        if self.history is None:
            history = History(['loss', *self.metrics_functions.keys()])
        else:
            history = self.history
        try:
            best_val_loss = 1000
            num_epochs = self.num_epochs if self.num_epochs >= 0 else 100000
            for epoch in range(self.begin_epoch, num_epochs):
                print('Starting epoch {}, lr = {}'.format(
                    epoch + 1, get_lr(self.optimizer)))
                self.train(epoch, history)
                val_loss = self.val(epoch, history)
                lr = get_lr(self.optimizer)
                if lr < self.min_lr:
                    break
                if self.save_every:
                    save_model(self.model, self.optimizer, self.scheduler,
                               epoch, self.save_model_dir, history)
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(self.model, self.optimizer, self.scheduler,
                               epoch, self.save_model_dir, history)
        except KeyboardInterrupt as e:
            if not self.allow_keyboard_interrupt:
                raise e
        print('Training finished.')
        history.plot_loss(save_path=osp.join(self.log_dir, 'loss.jpg'))
        history.plot_metrics(save_dir=self.log_dir)
