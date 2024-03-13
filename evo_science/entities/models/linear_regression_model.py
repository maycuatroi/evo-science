import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from evo_science import FeatureSet, BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression(**kwargs)

    def plot(self, x: FeatureSet, y: FeatureSet):
        x_data = x.to_numpy()
        y_data = y.to_numpy()
        y_pred = self.predict(x_data)
        dimension = x_data.shape[1]
        if dimension == 1:
            plt.scatter(x_data, y_data, color="blue")
            plt.plot(x_data, y_pred, color="green", linewidth=3)
            plt.show()
        else:
            for i in range(dimension):
                x_dim = x_data[:, i]
                x_dim = x_dim.reshape(-1, 1)
                plt.scatter(x_dim, y_data, color="blue")
                plt.scatter(x_dim, y_pred, color="red")
                # prepare line function y=mx+c
                m = self.model.coef_[0][i]
                c = self.model.intercept_[0]
                y_line = m * x_dim + c
                plt.plot(x_dim, y_line, color="green", linewidth=3)

                plt.xlabel(x.column_names[i])
                plt.show()

    def _calculate_coefficients(self, x: FeatureSet):
        coef = self.model.coef_[0]
        feature_names = x.column_names
        coef_dict = dict(zip(feature_names, coef))
        df = pd.DataFrame(coef_dict.items(), columns=["Feature", "Coefficient"])
        return df
