from evo_science.packages.yolo.layers.backbone import Backbone
from evo_science.packages.yolo.layers.head import Head
from evo_science.packages.yolo.layers.neck import Neck
from evo_science.packages.yolo.yolo import Yolo
from evo_science.packages.yolo.yolo_config import YoloConfig


class YoloV8(Yolo):
    @staticmethod
    def yolo_v8_n(num_classes: int = 80):
        depth = [1, 2, 2]
        width = [3, 16, 32, 64, 128, 256]
        return YoloV8(width=width, depth=depth, num_classes=num_classes)


if __name__ == "__main__":
    model = YoloV8.yolo_v8_n()
    print(model)