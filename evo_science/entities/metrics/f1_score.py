import numpy as np

from .base_binary_classify_metric import BaseBinaryClassifyMetric


class F1Score(BaseBinaryClassifyMetric):
    name = "F1 Score"

    def _calculate_np(self, y_true: np.array, y_pred: np.array):
        y_pred = self._binary_threshold(y_pred)
        p = self.calculate_precision(y_true, y_pred)
        r = self.calculate_recall(y_true, y_pred)
        return 2 * (p * r) / (p + r)

    def calculate_recall(self, y_true, y_pred):
        tp = self.calculate_tp(y_true, y_pred)
        fn = self.calculate_fn(y_true, y_pred)
        return tp / (tp + fn)

    def calculate_precision(self, y_true, y_pred):
        tp = self.calculate_tp(y_true, y_pred)
        fp = self.calculate_fp(y_true, y_pred)
        return tp / (tp + fp)
