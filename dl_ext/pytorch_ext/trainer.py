from __future__ import absolute_import
from __future__ import print_function

import os

import progressbar
import torch

from .model import save_model
from .optim import OneCycleScheduler
from .optim import get_lr
from ..average_meter import AverageMeter


class BaseTrainer:

    def __init__(self, model, optimizer, dataloaders,
                 num_epochs, loss_function, scheduler=None,
                 begin_epoch=0, output_dir='models',
                 max_lr=1e-9, save_every=True) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.dataloaders = dataloaders
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
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

    def train(self, epoch):
        meters = {'loss': AverageMeter()}
        self.model.train()
        widgets = [
            progressbar.Bar(),
            progressbar.Percentage(),
            progressbar.ETA(),
            progressbar.DynamicMessage('loss'),
        ]
        with progressbar.ProgressBar(max_value=len(self.dataloaders['train']), widgets=widgets) as bar:
            for it, batch in enumerate(self.dataloaders['train']):
                self.optimizer.zero_grad()
                loss, scores = self.model(*batch)
                loss = loss.mean()
                meters['loss'].update(loss.item())
                loss.backward()
                self.optimizer.step()
                bar_values = {'loss': meters['loss'].avg}

                bar.update(it, **bar_values)
            s = 'Epoch {}, train, loss = {:.4f}'.format(
                epoch + 1, meters['loss'].avg)
            print(s)

    def val(self, epoch):
        meters = {'loss': AverageMeter()}
        self.model.eval()
        widgets = [
            progressbar.Bar(),
            progressbar.Percentage(),
            progressbar.ETA(),
            progressbar.DynamicMessage('loss'),
        ]
        with progressbar.ProgressBar(max_value=len(self.dataloaders['val']), widgets=widgets) as bar:
            with torch.no_grad():
                for it, batch in enumerate(self.dataloaders['val']):
                    self.optimizer.zero_grad()
                    loss, scores = self.model(*batch)
                    loss = loss.mean()
                    meters['loss'].update(loss.item())
                    bar_values = {'loss': meters['loss'].avg}
                    bar.update(it, **bar_values)
                s = 'Epoch {}, val, loss = {:.4f}'.format(
                    epoch + 1, meters['loss'].avg)
                print(s)
                self.scheduler.step(meters['loss'].avg)
                return meters['loss'].avg

    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        best_val_loss = 1000
        num_epochs = self.num_epochs if self.num_epochs >= 0 else 100000
        for epoch in range(self.begin_epoch, num_epochs):
            print('Starting epoch {}, lr = {}'.format(
                epoch + 1, get_lr(self.optimizer)))
            self.train(epoch)
            val_loss = self.val(epoch)
            if self.save_every:
                save_model(self.model, self.optimizer, self.scheduler,
                           epoch, self.output_dir)
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(self.model, self.optimizer, self.scheduler,
                           epoch, self.output_dir)
        print('Training finished.')
