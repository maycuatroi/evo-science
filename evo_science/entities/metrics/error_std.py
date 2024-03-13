from evo_science.entities.metrics import BaseMetric


class ErrorStd(BaseMetric):
    name = "Error STD"

    def _calculate_np(self, y_true, y_pred):
        return (y_true - y_pred).std()
