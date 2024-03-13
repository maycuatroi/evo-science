from src.entities.metrics.base_metric import BaseMetric


class MSE(BaseMetric):
    name = "MSE"

    def _calculate_np(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
