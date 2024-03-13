from src.entities.metrics.base_metric import BaseMetric


class Intercept(BaseMetric):
    name = "Intercept"

    def _calculate_np(self, y_true, y_pred):
        return self.model.model.intercept_[0]
