from dataclasses import dataclass


@dataclass
class ValidatorConfig:
    """Configuration for YOLO model validator"""

    data_dir: str
    input_size: int = 640
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    iou_min: float = 0.5
    iou_max: float = 0.95
    iou_steps: int = 10
    conf_thres: float = 0.001
    iou_thres: float = 0.7
