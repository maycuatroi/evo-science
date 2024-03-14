class MetricLib:
    metrics = {
        "RMSE": "from src.entities.metrics.rmse import RMSE",
        "Slope": "from evo_science.entities.metrics.slope import Slope",
        "MAE": "from src.entities.metrics.mae import MAE",
    }

    def __init__(self):

        raise NotImplementedError("This class is not implemented yet")

    @classmethod
    def get_metric(cls, name):
        if name not in cls.metrics:
            raise ValueError(f"Metric {name} is not supported")
        exec(cls.metrics[name])
