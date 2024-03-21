from evo_science.entities.metrics import BaseMetric


class Accuracy(BaseMetric):
    name = "Accuracy"

    def _calculate_np(self, y_true, y_pred):
        y_pred[y_true > self.threshold] = 1
        y_pred[y_true <= self.threshold] = 0
        return (y_true == y_pred).mean()

    def _on_init(self, threshold=0.5, **kwargs):
        self.threshold = threshold

    def _on_call(self, threshold=0.5, **kwargs):
        self.threshold = threshold
