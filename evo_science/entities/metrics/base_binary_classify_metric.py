import numpy as np

from evo_science.entities.metrics.base_metric import BaseMetric


class BaseBinaryClassifyMetric(BaseMetric):
    def _on_init(self, threshold=0.5, **kwargs):
        self.threshold = threshold

    def _on_call(self, threshold=0.5, **kwargs):
        self.threshold = threshold

    def _binary_threshold(self, y_pred):
        y_pred[y_pred > self.threshold] = 1
        y_pred[y_pred <= self.threshold] = 0
        return y_pred

    def calculate_tp(self, y_true, y_pred):
        """
        Calculate the number of true positives.
        """
        return np.sum((y_true == 1) & (y_pred == 1))

    def calculate_fp(self, y_true, y_pred):
        """
        Calculate the number of false positives.
        """
        return np.sum((y_true == 0) & (y_pred == 1))

    def calculate_tn(self, y_true, y_pred):
        """
        Calculate the number of true negatives.
        """
        return np.sum((y_true == 0) & (y_pred == 0))

    def calculate_fn(self, y_true, y_pred):
        """
        Calculate the number of false negatives.
        """
        return np.sum((y_true == 1) & (y_pred == 0))
