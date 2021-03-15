import datetime
import time

import torch


class Timer(object):
    def __init__(self, ignore_first_n=0):
        self.reset(ignore_first_n)

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self, synchronize=False):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        if synchronize:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def toc(self, average=True, synchronize=False):
        if synchronize:
            torch.cuda.synchronize()
        self.add(time.time() - self.start_time)
        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff):
        if self.ignore_first_n > 0:
            self.ignore_first_n -= 1
        else:
            self.calls += 1
            self.diff = time_diff
            self.total_time += self.diff

    def reset(self, ignore_first_n):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.ignore_first_n = ignore_first_n

    def avg_time_str(self):
        time_str = str(datetime.timedelta(seconds=self.average_time))
        return time_str


def get_time_str(time_diff):
    time_str = str(datetime.timedelta(seconds=time_diff))
    return time_str
