import numpy as np

from evo_science.dl.activations.base_activation import BaseActivation


class Softmax(BaseActivation):
    def _activate(self, x: np.array) -> np.array:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _derivative(self, y_pred, y_true) -> np.array:
        m = y_true.shape[0]
        grad = y_pred
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad
