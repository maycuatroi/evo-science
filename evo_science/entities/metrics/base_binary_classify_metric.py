import numpy as np

from evo_science.entities.metrics import BaseMetric


class BaseBinaryClassifyMetric(BaseMetric):
    def _on_init(self, threshold=0.5, **kwargs):
        self.threshold = threshold

    def _on_call(self, threshold=0.5, **kwargs):
        self.threshold = threshold

    def _binary_threshold(self, y_pred):
        y_pred[y_pred > self.threshold] = 1
        y_pred[y_pred <= self.threshold] = 0
        return y_pred
