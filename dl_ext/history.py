import matplotlib.pyplot as plt
import numpy as np
from visdom import Visdom
from os import path as osp
from .plot_utils import update_vis_plot


class History:
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = ['loss']
        metrics = list(set(metrics))
        self.records = {}
        for k in metrics:
            self.records[k] = {'train': [], 'val': []}
        cmap = plt.get_cmap('gnuplot')
        self.colors = [cmap(i) for i in np.linspace(0, 1, 2 * (len(metrics) + 1))]

    def load_state_dict(self, other):
        self.records = other

    def state_dict(self):
        return self.records

    def update_vis_plot(self, viz: Visdom, epoch, epoch_plot):
        values = []
        for phase in ['train', 'val']:
            for k in self.records.keys():
                values.append(self.records[k][phase][-1])
        update_vis_plot(viz, epoch, epoch_plot, 'append', values)

    def plot_loss(self, save_path=None):
        plt.figure()
        plt.plot(range(len(self.records['loss']['train'])),
                 self.records['loss']['train'],
                 label='loss_train')
        plt.plot(range(len(self.records['loss']['val'])),
                 self.records['loss']['val'],
                 label='loss_val')
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
            plt.show()

    def plot_metrics(self, save_dir=None):
        for key in self.records.keys():
            if key == 'loss': continue
            plt.figure()
            plt.plot(range(len(self.records[key]['train'])),
                     self.records[key]['train'],
                     label=key + '_train')
            plt.plot(range(len(self.records[key]['val'])),
                     self.records[key]['val'],
                     label=key + '_val')
            plt.legend()
            if save_dir is not None:
                plt.savefig(osp.join(save_dir, key + '.jpg'))
                plt.show()
