from ..average_meter import AverageMeter
from fastai.basic_train import LearnerCallback, Learner


class PrintOnIterCallback(LearnerCallback):
    learn: Learner

    def __init__(self, learn, print_interval=1, has_loss_metric=False, print_func=print):
        super().__init__(learn)
        self.print_interval = print_interval
        self.has_loss_metric = has_loss_metric
        self.print_func = print_func
        if has_loss_metric:
            keys = self.learn.loss_func.metric_names
            self.meters = dict(zip(keys, [AverageMeter()] * len(keys)))

    def on_batch_end(self, **kwargs) -> None:
        if self.has_loss_metric:
            for k, v in self.learn.loss_func.metrics.items():
                self.meters[k].update(v.item())
        if kwargs['iteration'] % self.print_interval == 0:
            s = 'Epoch: {}'.format(kwargs['epoch']) + \
                ' iter: {}'.format(kwargs['iteration']) + \
                ' last_loss: {:.4f}'.format(kwargs['last_loss'].item()) + \
                ' smooth_loss: {:.4f}'.format(kwargs['smooth_loss'].item())
            if self.has_loss_metric:
                for k, v in self.meters.items():
                    s += ' ' + k + ': {:.4f}'.format(v.avg)
            self.print_func(s)
