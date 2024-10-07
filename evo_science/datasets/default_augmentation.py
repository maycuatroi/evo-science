import albumentations
import numpy as np


class DefaultAugmentation:
    def __init__(self):
        transforms = [
            albumentations.Blur(p=0.01),
            albumentations.CLAHE(p=0.01),
            albumentations.ToGray(p=0.01),
            albumentations.MedianBlur(p=0.01),
        ]
        self.transform = albumentations.Compose(transforms, albumentations.BboxParams("yolo", ["class_labels"]))

    def __call__(self, image, box, cls):
        if self.transform:
            x = self.transform(image=image, bboxes=box, class_labels=cls)
            image = x["image"]
            box = np.array(x["bboxes"])
            cls = np.array(x["class_labels"])
        return image, box, cls
