import math
import random
from typing import Tuple, Optional

import cv2
import numpy as np


class ImageTransformer:
    @staticmethod
    def resize(
        image: np.ndarray, input_size: int, augment: bool
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        shape = image.shape[:2]
        r = min(input_size / shape[0], input_size / shape[1])
        if not augment:
            r = min(r, 1.0)

        pad = int(round(shape[1] * r)), int(round(shape[0] * r))
        w = (input_size - pad[0]) / 2
        h = (input_size - pad[1]) / 2

        if shape[::-1] != pad:
            interpolation = (
                random.choice(
                    [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
                )
                if augment
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(image, dsize=pad, interpolation=interpolation)

        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return image, (r, r), (w, h)

    @staticmethod
    def random_perspective(
        image: np.ndarray, label: np.ndarray, border: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if border is None:
            border = (0, 0)

        h = image.shape[0] + border[0] * 2
        w = image.shape[1] + border[1] * 2

        center = np.eye(3)
        center[0, 2] = -image.shape[1] / 2
        center[1, 2] = -image.shape[0] / 2

        perspective = np.eye(3)
        rotate = np.eye(3)
        a = random.uniform(-0, 0)
        s = random.uniform(1 - 0.5, 1 + 0.5)
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        shear = np.eye(3)
        shear[0, 1] = math.tan(random.uniform(-0, 0) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-0, 0) * math.pi / 180)

        translate = np.eye(3)
        translate[0, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * w
        translate[1, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * h

        matrix = translate @ shear @ rotate @ perspective @ center
        if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

        n = len(label)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ matrix.T
            xy = xy[:, :2].reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
            box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)

            w1, h1 = label[:, 2] - label[:, 0], label[:, 3] - label[:, 1]
            w2, h2 = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
            aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
            indices = (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)

            label = label[indices]
            label[:, 1:5] = box[indices]

        return image, label
