from evo_science import FEATURE_TYPES, FeatureSet
from evo_science.entities.features.base_feature import BaseFeature
from evo_science.entities.metrics import *
from evo_science.entities.models.linear_regression_model import LinearRegressionModel


class PClass(BaseFeature):
    column_name = "Pclass"  # column_name is optional
    column_type = int


class Sex(BaseFeature):
    column_type = FEATURE_TYPES.CATEGORY
    categories = {"male": 1, "female": 0}


class Age(BaseFeature):
    column_type = float


class SibSp(BaseFeature):
    column_name = "Siblings/Spouses Aboard"
    column_type = int


class Parch(BaseFeature):
    column_name = "Parents/Children Aboard"
    column_type = int


class Fare(BaseFeature):
    column_type = float


class Survived(BaseFeature):
    column_type = int


def example_lr_model():

    model = LinearRegressionModel()

    x = FeatureSet(features=[PClass, Sex, Age, SibSp, Parch, Fare])
    y = FeatureSet(features=[Survived])

    (x + y).build(
        csv_path="https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )

    model.fit(x=x, y=y)
    model.evaluate(
        x=x,
        y=y,
        metrics=[
            Slope,
            ErrorStd,
            Accuracy(threshold=0.5),
            Precision,
            Recall,
            F1Score,
            RocCurve,
            AUC,
        ],
    )
    model.calculate_coefficients(x=x)

    # model.plot(x=x, y=y)


if __name__ == "__main__":
    example_lr_model()
