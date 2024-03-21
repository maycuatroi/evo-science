import numpy as np

from evo_science.entities.metrics.base_binary_classify_metric import BaseBinaryClassifyMetric


class Recall(BaseBinaryClassifyMetric):

    def _calculate_np(self, y_true: np.array, y_pred: np.array):
        y_pred = self._binary_threshold(y_pred)
        tp = self.calculate_tp(y_true, y_pred)
        fn = self.calculate_fn(y_true, y_pred)
        return tp / (tp + fn)