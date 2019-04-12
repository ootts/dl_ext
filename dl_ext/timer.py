import time


class Timer:
    def __init__(self, unit='ms'):
        valid_units = ['ms', 's']
        _unit_functions = {'s': lambda x: x, 'ms': lambda x: x * 1000}
        if unit not in valid_units:
            raise Exception('expect unit in {}, but found {}'.format(valid_units, unit))
        self.unit = unit
        self.unit_function = _unit_functions[unit]
        self._msgs = []
        self.begins = {}
        self.totals = {}
        self.times = {}

    def tic(self, msg=''):
        self._msgs.append(msg)
        if msg not in self.totals:
            self.times[msg] = 0
            self.totals[msg] = 0
        self.begins[msg] = time.time()

    def toc(self):
        end = time.time()
        last_msg = self._msgs.pop()
        self.totals[last_msg] += end - self.begins[last_msg]
        self.times[last_msg] += 1
        return end - self.begins[last_msg]

    def summarize(self):

        for k, v in self.totals.items():
            print('{} total {} times takes {:.2f} {}, average {:.2f} {}.'.format(
                k, self.times[k], self.unit_function(v), self.unit, self.unit_function(v / self.times[k]),
                self.unit))


timer = Timer()
