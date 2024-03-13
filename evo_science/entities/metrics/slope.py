from evo_science.entities.metrics.base_metric import BaseMetric


class Slope(BaseMetric):
    name = "Slope"

    def _calculate_np(self, y_true, y_pred):
        return self.model.model.coef_[0][0]
