import numpy as np

from . import BaseMetric


class MAE(BaseMetric):
    name = "MAE"

    def _calculate_np(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
