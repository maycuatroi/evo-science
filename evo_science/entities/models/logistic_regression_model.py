from sklearn.linear_model import LogisticRegression

from evo_science.entities.features import FeatureSet
from evo_science.entities.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression(**kwargs)

    def fit(self, x: FeatureSet, y: FeatureSet):
        x_data = x.to_numpy()
        y_data = y.to_numpy().flatten()
        self.model.fit(x_data, y_data)
