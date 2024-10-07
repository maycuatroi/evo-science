from typing import List
from evo_science.datasets.lables.abstract_label import AbstractLabel
from evo_science.datasets.lables.bbox_label import BBoxLabel


class Image:
    def __init__(self, image_path: str, labels: List[AbstractLabel]):
        self.image_path = image_path
        self.labels = labels

        self.bboxes: List[BBoxLabel] = [label for label in labels if label.label_type == "bbox"]

    def plot(self):
        import matplotlib.pyplot as plt
        import cv2

        img = cv2.imread(self.image_path)
        for bbox in self.bboxes:
            xmin, ymin, xmax, ymax = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        plt.imshow(img)
        plt.show()
