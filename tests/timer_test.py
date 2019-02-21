import time

from clh_utils.timer import timer


def f1():
    timer.tic('inner for loop')
    for i in range(10):
        time.sleep(0.01)
    timer.toc()


if __name__ == '__main__':
    timer.tic('outer for loop')
    for i in range(10):
        f1()
    timer.toc()
    timer.summarize()