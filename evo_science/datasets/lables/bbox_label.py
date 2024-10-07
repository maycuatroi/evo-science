from evo_science.datasets.lables.abstract_label import AbstractLabel


class BBoxLabel(AbstractLabel):
    label_type = "bbox"

    def __init__(self, xmin, ymin, xmax, ymax, label):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
