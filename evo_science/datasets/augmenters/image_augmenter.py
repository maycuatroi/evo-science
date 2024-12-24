from typing import Tuple

import cv2
import numpy as np


class ImageAugmenter:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A

            transforms = [
                A.Blur(p=0.01),
                A.CLAHE(p=0.01),
                A.ToGray(p=0.01),
                A.MedianBlur(p=0.01),
            ]
            self.transform = A.Compose(transforms, A.BboxParams("yolo", ["class_labels"]))
        except ImportError:
            pass

    def __call__(
        self, image: np.ndarray, box: np.ndarray, cls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.transform:
            x = self.transform(image=image, bboxes=box, class_labels=cls)
            image = x["image"]
            box = np.array(x["bboxes"])
            cls = np.array(x["class_labels"])
        return image, box, cls

    @staticmethod
    def augment_hsv(image: np.ndarray) -> None:
        h, s, v = 0.015, 0.7, 0.4
        r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype("uint8")
        lut_s = np.clip(x * r[1], 0, 255).astype("uint8")
        lut_v = np.clip(x * r[2], 0, 255).astype("uint8")

        hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)

    @staticmethod
    def mix_up(
        image1: np.ndarray, box1: np.ndarray, image2: np.ndarray, box2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha = np.random.beta(32.0, 32.0)
        image = (image1 * alpha + image2 * (1 - alpha)).astype(np.uint8)
        box = np.concatenate((box1, box2), 0)
        return image, box
