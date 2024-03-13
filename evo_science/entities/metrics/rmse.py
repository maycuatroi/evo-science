from . import BaseMetric


class RMSE(BaseMetric):
    name = "RMSE"

    def _calculate_np(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean() ** 0.5
