import numpy as np

from evo_science.entities.metrics.base_binary_classify_metric import (
    BaseBinaryClassifyMetric,
)


class Accuracy(BaseBinaryClassifyMetric):
    name = "Accuracy"

    def _calculate_np(self, y_true: np.array, y_pred: np.array):
        y_pred = self._binary_threshold(y_pred)
        return (y_true == y_pred).mean()

    def _on_init(self, threshold=0.5, **kwargs):
        self.threshold = threshold

    def _on_call(self, threshold=0.5, **kwargs):
        self.threshold = threshold
