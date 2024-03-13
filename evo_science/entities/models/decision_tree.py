from evo_science.entities.models.base_model import BaseModel


class DecisionTree(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decision_tree = None
        raise NotImplementedError("This class is not implemented yet.")

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass

    def plot(self, x, y):
        pass
