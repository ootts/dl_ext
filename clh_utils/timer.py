import time


class Timer:
    def __init__(self, print_all=False, unit='ms'):
        valid_units = ['ms', 's']
        _unit_functions = {'s': lambda x: x, 'ms': lambda x: x * 1000}
        if unit not in valid_units:
            raise Exception('expect unit in {}, but found {}'.format(valid_units, unit))
        self.unit = unit
        self.unit_function = _unit_functions[unit]
        self._begin = 0
        self._msg = ''
        self.first_time = 0
        self.totals = {}
        self.times = {}
        self.print_all = print_all

    def tic(self, msg=''):
        self._msg = msg
        if msg not in self.totals:
            self.times[msg] = 0
            self.totals[msg] = 0
        if self.print_all:
            print(self._msg, 'begins')
        self._begin = time.time()
        if self.first_time == 0:
            self.first_time = self._begin

    def toc(self):
        end = time.time()
        if self.print_all:
            print(self._msg, 'ends', end - self._begin)
        self.totals[self._msg] += end - self._begin
        self.times[self._msg] += 1
        return end - self._begin

    def summarize(self):

        total_avg = 0
        for k, v in self.totals.items():
            try:
                print('{} total {} times takes {:.2f} {}, average {:.2f} {}.'.format(
                    k, self.times[k], self.unit_function(v), self.unit, self.unit_function(v / self.times[k]),
                    self.unit))
                total_avg += v / self.times[k]

            except ZeroDivisionError as e:
                print(k)
        print('total average {:.2f} {}, {:.2f}fps'.format(self.unit_function(total_avg), self.unit, 1 / total_avg))
