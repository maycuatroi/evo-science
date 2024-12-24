import numpy as np


class BoxTransformer:
    @staticmethod
    def wh2xy(x: np.ndarray, w: float = 640, h: float = 640, pad_w: float = 0, pad_h: float = 0) -> np.ndarray:
        y = np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
        return y

    @staticmethod
    def xy2wh(x: np.ndarray, w: int, h: int) -> np.ndarray:
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)

        y = np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
        y[:, 2] = (x[:, 2] - x[:, 0]) / w
        y[:, 3] = (x[:, 3] - x[:, 1]) / h
        return y
