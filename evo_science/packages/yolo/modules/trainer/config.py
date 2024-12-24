from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Configuration for YOLO trainer"""

    data_dir: str
    batch_size: int
    epochs: int
    world_size: int
    input_size: int = 640
    distributed: bool = False
    lrf: float = 0.2
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    local_rank: int = 0
    warmup_epochs: int = 5
    warmup_bias_lr: float = 0.1
    warmup_momentum: float = 0.8
