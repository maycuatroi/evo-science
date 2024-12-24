import os
import warnings
import torch

from evo_science.entities.utils import setup_multi_processes, setup_seed
from evo_science.packages.yolo.modules.handle_distributed import handle_distributed
from evo_science.packages.yolo.yolo_v8 import YoloV8
from evo_science.packages.yolo.modules.profiler import Profiler
from evo_science.packages.yolo.modules.trainer import Trainer, TrainerConfig

warnings.filterwarnings("ignore")

coco_dir = "./data/COCO"
config = TrainerConfig(
    data_dir=coco_dir,
    batch_size=32,
    epochs=300,
    world_size=1,
    input_size=640,
    distributed=False,
    lrf=0.2,
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    local_rank=0,
    warmup_epochs=5,
    warmup_bias_lr=0.1,
    warmup_momentum=0.8,
)


def main():
    handle_distributed()  # Handle distributed training if necessary

    setup_seed()
    setup_multi_processes()

    model = YoloV8.yolo_v8_n()

    profiler = Profiler(model=model, input_size=config.input_size, batch_size=config.batch_size, num_classes=2)
    profiler.profile()

    trainer = Trainer(model, config)
    trainer.train()


if __name__ == "__main__":
    main()
