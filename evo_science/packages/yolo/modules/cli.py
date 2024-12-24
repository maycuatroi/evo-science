import click
import os
import warnings
import torch

from evo_science.entities.utils import setup_multi_processes, setup_seed
from evo_science.packages.yolo.yolo_v8 import YoloV8
from evo_science.packages.yolo.modules.validator import validate
from evo_science.packages.yolo.modules.demo import demo
from evo_science.packages.yolo.modules.profiler import Profiler
from evo_science.packages.yolo.modules.trainer import Trainer, TrainerConfig

warnings.filterwarnings("ignore")
coco_dir = f"./data/COCO"

DEFAULT_TRAIN_CONFIG = TrainerConfig(
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


def train(
    model,
    data_dir: str = DEFAULT_TRAIN_CONFIG.data_dir,
    batch_size: int = DEFAULT_TRAIN_CONFIG.batch_size,
    epochs: int = DEFAULT_TRAIN_CONFIG.epochs,
    world_size: int = DEFAULT_TRAIN_CONFIG.world_size,
    input_size: int = DEFAULT_TRAIN_CONFIG.input_size,
    distributed: bool = DEFAULT_TRAIN_CONFIG.distributed,
    lrf: float = DEFAULT_TRAIN_CONFIG.lrf,
    lr0: float = DEFAULT_TRAIN_CONFIG.lr0,
    momentum: float = DEFAULT_TRAIN_CONFIG.momentum,
    weight_decay: float = DEFAULT_TRAIN_CONFIG.weight_decay,
    local_rank: int = DEFAULT_TRAIN_CONFIG.local_rank,
    warmup_epochs: int = DEFAULT_TRAIN_CONFIG.warmup_epochs,
    warmup_bias_lr: float = DEFAULT_TRAIN_CONFIG.warmup_bias_lr,
    warmup_momentum: float = DEFAULT_TRAIN_CONFIG.warmup_momentum,
):
    """Train a YOLO model with the given configuration"""
    config = TrainerConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        epochs=epochs,
        world_size=world_size,
        input_size=input_size,
        distributed=distributed,
        lrf=lrf,
        lr0=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        local_rank=local_rank,
        warmup_epochs=warmup_epochs,
        warmup_bias_lr=warmup_bias_lr,
        warmup_momentum=warmup_momentum,
    )
    trainer = Trainer(model, config)
    trainer.train()


@click.command()
@click.option("--input-size", default=640, type=int, help="Input size")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--epochs", default=300, type=int, help="Number of epochs")
@click.option("--is_train", is_flag=True, help="Run training", default=True)
@click.option("--is_test", is_flag=True, help="Run testing", default=False)
@click.option("--is_demo", is_flag=True, help="Run demo", default=False)
@click.option("--local-rank", default=0, type=int, help="Local rank")
@click.option("--world-size", default=1, type=int, help="World size")
@click.option("--data-dir", default=coco_dir, type=str, help="COCO formatted dataset directory")
@click.option("--distributed", is_flag=True, help="Distributed training")
def main(input_size, batch_size, epochs, is_train, is_test, is_demo, local_rank, world_size, distributed, data_dir):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(device=local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if local_rank == 0:
        if not os.path.exists("weights"):
            os.makedirs("weights")

    setup_seed()
    setup_multi_processes()

    model = YoloV8.yolo_v8_n()
    profiler = Profiler(model=model, input_size=input_size, batch_size=batch_size, num_classes=2)
    profiler.profile()

    if is_train:
        train(
            model=model,
            data_dir=data_dir,
            batch_size=batch_size,
            epochs=epochs,
            input_size=input_size,
            world_size=world_size,
            distributed=distributed,
            lrf=0.2,
            lr0=0.01,
            warmup_bias_lr=0.1,
            warmup_momentum=0.8,
            weight_decay=0.0005,
            local_rank=local_rank,
            warmup_epochs=5,
        )

    if is_test:
        model = YoloV8.yolo_v8_n()
        model.load_weight(checkpoint_path="./weights/best.pt")
        model.half()
        model.eval()
        validate(model, data_dir=data_dir)
    if is_demo:
        demo(input_size=input_size, model=model)


if __name__ == "__main__":
    main()
