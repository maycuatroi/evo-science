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
        x_data = x.to_numpy(is_train=False)
        y_data = y.to_numpy(is_train=False)

        viz = dtreeviz(
            self.model,
            x_data,
            y_data,
            tree_index=0,
            target_name=y.column_names[0],
            orientation="LR",  # left-right orientation
            feature_names=x.column_names,
            class_names=y.column_names,
            X=x_data,
        )
        plt.show()

    def _calculate_coefficients(self, x: FeatureSet):
        coef = self.model.feature_importances_
        feature_names = x.column_names
        coef_dict = dict(zip(feature_names, coef))
        df = pd.DataFrame(coef_dict.items(), columns=["Feature", "Coefficient"])

        return df
