# from fastai.torch_core import *
from typing import Any
from warnings import warn

import math
import torch.distributed as dist
from fastai.torch_core import rank_distrib, is_listy, ifnone

from ..average_meter import AverageMeter
from fastai.basic_train import LearnerCallback, Learner
from fastai.callbacks import LossMetrics as LM


class PrintOnIterCallback(LearnerCallback):
    learn: Learner

    def __init__(self, learn, print_interval=1, has_loss_metric=False, print_funcs=None):
        super().__init__(learn)
        if print_funcs is None:
            print_funcs = [print]
        self.print_funcs = print_funcs
        self.print_interval = print_interval
        self.has_loss_metric = has_loss_metric
        if has_loss_metric:
            keys = self.learn.loss_func.metric_names
            if LM in self.learn.callback_fns:
                warn('bulitin LossMetrics has been deprecated! Use clh_utils.LossMetrics instead!')
            self.meters = {k: AverageMeter() for k in keys}
        self.eval_iters = 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.total_train_iters = math.ceil(len(self.learn.data.train_ds) /
                                           world_size /
                                           self.learn.data.train_dl.batch_size)
        self.total_val_iters = math.ceil(len(self.learn.data.valid_ds) /
                                         world_size /
                                         self.learn.data.valid_dl.batch_size)

    def on_epoch_begin(self, **kwargs: Any) -> None:
        for k, v in self.meters.items():
            v.reset()
        self.eval_iters = 0

    def on_batch_end(self, **kwargs) -> None:
        if dist.is_initialized() and rank_distrib() != 0: return
        if self.has_loss_metric:
            for k, v in self.learn.loss_func.metrics.items():
                self.meters[k].update(v.item())

        if kwargs['train']:
            phase = 'Train'
            tit = self.total_train_iters
            it = kwargs['iteration'] % tit
        else:
            phase = 'Validate'
            tit = self.total_val_iters
            it = self.eval_iters % tit
            self.eval_iters += 1
        if it % self.print_interval == 0:
            s = f"Epoch: {kwargs['epoch']} Phase: {phase}"
            s = s + f" iter: [{it}/{tit}]"
            if phase == 'Train':
                s += f" last_loss: {kwargs['last_loss'].item():.4f}"
                s += f" smooth_loss: {kwargs['smooth_loss'].item():.4f}"
                if self.has_loss_metric:
                    for k, v in self.meters.items():
                        s += ' ' + k + ': {:.4f}'.format(v.avg)
            for pf in self.print_funcs:
                pf(s)


class LossMetrics(LearnerCallback):
    "Add `loss_func.metrics` to metrics named by `loss_func.metric_names`"
    _order = -20  # Needs to run before the recorder

    def on_train_begin(self, **kwargs):
        "Add the metrics names to the `Recorder`."
        self.names = ifnone(self.learn.loss_func.metric_names, [])
        if not self.names: warn('LossMetrics requested but no loss_func.metric_names provided')
        self.learn.recorder.add_metric_names(self.names)

    def on_epoch_begin(self, **kwargs):
        "Initialize the metrics for this epoch."
        self.metrics = {name: 0. for name in self.names}
        self.nums = 0

    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        if train: return
        if not is_listy(last_target): last_target = [last_target]
        bs = last_target[0].size(0)
        for name in self.names:
            self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

    def on_epoch_end(self, last_metrics, **kwargs):
        "Finish the computation and sends the result to the Recorder."
        if not self.nums: return
        metrics = [self.metrics[name] / self.nums for name in self.names]
        return {'last_metrics': last_metrics + metrics}
