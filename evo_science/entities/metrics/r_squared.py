from . import BaseMetric


class RSquared(BaseMetric):
    name = "R Squared"

    def _calculate_np(self, y_true, y_pred):
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()

        return 1 - (ss_res / ss_tot)
