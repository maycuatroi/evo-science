import numpy as np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.total = 0
        self.average = 0

    def update(self, value, n=1):
        if not np.isnan(value):
            self.count += n
            self.total += value * n
            self.average = self.total / self.count

    @property
    def avg(self):
        return self.average
