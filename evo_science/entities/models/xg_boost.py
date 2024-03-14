import pandas as pd
import xgboost
from dtreeviz import dtreeviz
from matplotlib import pyplot as plt

from evo_science.entities.features import FeatureSet


class BaseModel:
    pass


class XGBoost(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = xgboost.XGBRegressor(**kwargs)

    def fit(self, x, y):
        self.model.fit(x.to_numpy(), y.to_numpy())

    def plot(self, x: FeatureSet, y: FeatureSet):
        raise NotImplementedError("XGBoost does not support plotting")

    def _calculate_coefficients(self, x: FeatureSet):
        coef = self.model.feature_importances_
        feature_names = x.column_names
        coef_dict = dict(zip(feature_names, coef))
        df = pd.DataFrame(coef_dict.items(), columns=["Feature", "Coefficient"])

        return df
