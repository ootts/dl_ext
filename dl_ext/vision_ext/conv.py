import math


def conv_size_out(size, kernel_size=5, stride=2, pad=0):
    return int(math.floor((size + 2 * pad - kernel_size) / stride + 1))

