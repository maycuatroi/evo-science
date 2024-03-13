from evo_science.entities.metrics.base_metric import BaseMetric


class ErrorAve(BaseMetric):
    name = "Error Ave"

    def _calculate_np(self, y_true, y_pred):
        return (y_true - y_pred).mean()
