import numpy as np

from evo_science.dl.activations.base_activation import BaseActivation


class Relu(BaseActivation):
    def _derivative(self, x: np.array) -> np.array:
        return (x > 0).astype(x.dtype)

    def _activate(self, x: np.array) -> np.array:
        return np.maximum(0, x)
