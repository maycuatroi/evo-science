import numpy as np

from evo_science.entities.metrics import BaseMetric


class WAPE(BaseMetric):
    """
    WAPE (Weighted Absolute Percentage Error) is a metric used to measure the accuracy of a forecasting model.
    It is calculated as the ratio of the sum of absolute errors to the sum of actual values, multiplied by 100.

    Formula:
    WAPE = (sum(|y_true - y_pred|) / sum(y_true)) * 100
    """
    name = "WAPE"

    def _calculate_np(self, y_true: np.array, y_pred: np.array):
        """
        Calculate WAPE using numpy arrays.
        """
        absolute_errors = np.abs(y_true - y_pred)
        sum_absolute_errors = np.sum(absolute_errors)
        sum_actual_values = np.sum(y_true)
        wape = (sum_absolute_errors / sum_actual_values) * 100
        return wape



