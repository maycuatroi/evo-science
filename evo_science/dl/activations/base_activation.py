import abc

import numpy as np


class BaseActivation:
    def activate(self, x: np.array) -> np.array:
        return self._activate(x)

    def derivative(self, x: np.array) -> np.array:
        return self._derivative(x)

    @abc.abstractmethod
    def _activate(self, x: np.array) -> np.array:
        return x

    @abc.abstractmethod
    def _derivative(self, **kwargs):
        raise NotImplementedError
