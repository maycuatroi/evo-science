import copy
import csv
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.sgd import SGD
from torch.utils import data

from evo_science.datasets.coco_dataset import CocoDataset
from evo_science.entities.optimizers.ema import ExponentialMovingAverage
from evo_science.entities.utils.average_meter import AverageMeter
from evo_science.packages.yolo.modules.trainer.config import TrainerConfig
from evo_science.packages.yolo.modules.trainer.primary_process import PrimaryProcessHandler
from evo_science.packages.yolo.modules.validator import validate


class Trainer:
    """YOLO model trainer with distributed training support"""

    def __init__(self, model: nn.Module, config: TrainerConfig):
        self.model = model
        self.config = config
        self.primary = PrimaryProcessHandler(config.local_rank)
        self.best_map = 0

        self.setup()

    def setup(self):
        """Initialize training components"""
        self.model.cuda()
        self._setup_dataset()
        self._setup_optimizer()
        self._setup_ema()
        self._setup_distributed()
        self._setup_training_tools()
        self._setup_csv_writer()

    def _setup_ema(self):
        """Setup Exponential Moving Average model"""
        self.ema = self.primary.run(lambda: ExponentialMovingAverage(self.model))

    @staticmethod
    def strip_optimizer(path: str):
        """Strip optimizer states from checkpoint"""
        x = torch.load(path, map_location=torch.device("cpu"))
        x["model"].half()
        for p in x["model"].parameters():
            p.requires_grad = False
        torch.save(x, path)

    def _collect_parameters(self) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
        """Collect model parameters grouped by type"""
        weights, biases, bn_weights = [], [], []
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_weights.append(module.weight)
            elif isinstance(module, torch.nn.modules.conv._ConvNd):
                weights.append(module.weight)
                if module.bias is not None:
                    biases.append(module.bias)
            elif isinstance(module, torch.nn.Linear):
                weights.append(module.weight)
                if module.bias is not None:
                    biases.append(module.bias)
        return weights, biases, bn_weights

    def _setup_dataset(self):
        """Setup training dataset"""
        self.dataset = CocoDataset(
            data_dir=self.config.data_dir, input_size=self.config.input_size, is_augment=True, data_type="train"
        )

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        accumulate = max(round(64 / (self.config.batch_size * self.config.world_size)), 1)
        weight_decay = 0.0005 * self.config.batch_size * self.config.world_size * accumulate / 64
        weights, biases, bn_weights = self._collect_parameters()

        self.optimizer = SGD(
            [
                {"params": biases, "lr": self.config.lr0},
                {"params": weights, "lr": self.config.lr0, "weight_decay": weight_decay},
                {"params": bn_weights},
            ],
            lr=self.config.lr0,
            momentum=self.config.momentum,
            nesterov=True,
        )

        lr_lambda = lambda x: ((1 - x / self.config.epochs) * (1.0 - self.config.lrf) + self.config.lrf)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)

    def _setup_distributed(self):
        """Setup distributed training if enabled"""
        sampler = None
        if self.config.distributed:
            sampler = data.distributed.DistributedSampler(self.dataset)

        self.loader = data.DataLoader(
            self.dataset,
            self.config.batch_size,
            sampler is None,
            sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn,
        )

        if self.config.distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                module=self.model, device_ids=[self.config.local_rank], output_device=self.config.local_rank
            )

    def _setup_training_tools(self):
        """Setup training tools like gradient scaler and loss criterion"""
        self.amp_scale = torch.cuda.amp.GradScaler()
        self.criterion = self.model.get_criterion()

    def _setup_csv_writer(self):
        """Setup CSV writer for logging training metrics"""
        if self.primary.is_primary:
            with open("weights/step.csv", "w") as f:
                self.csv_writer = csv.DictWriter(
                    f, fieldnames=["epoch", "box", "dfl", "cls", "Recall", "Precision", "mAP@50", "mAP"]
                )
                self.csv_writer.writeheader()

    def _update_warmup(self, x: int, epoch: int, accumulate: int) -> int:
        """Update warmup parameters"""
        num_warmup = max(round(self.config.warmup_epochs * len(self.loader)), 1000)
        if x <= num_warmup:
            xp = [0, num_warmup]
            fp = [1, 64 / (self.config.batch_size * self.config.world_size)]
            accumulate = max(1, np.interp(x, xp, fp).round())
            for j, y in enumerate(self.optimizer.param_groups):
                if j == 0:
                    fp = [
                        self.config.warmup_bias_lr,
                        y["initial_lr"]
                        * ((1 - epoch / self.config.epochs) * (1.0 - self.config.lrf) + self.config.lrf),
                    ]
                else:
                    fp = [
                        0.0,
                        y["initial_lr"]
                        * ((1 - epoch / self.config.epochs) * (1.0 - self.config.lrf) + self.config.lrf),
                    ]
                y["lr"] = np.interp(x, xp, fp)
                if "momentum" in y:
                    fp = [self.config.warmup_momentum, self.config.momentum]
                    y["momentum"] = np.interp(x, xp, fp)
        return accumulate

    def _train_step(
        self, samples: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute single training step"""
        samples = samples.cuda().float() / 255
        with torch.amp.autocast(device_type="cuda"):
            outputs = self.model(samples)
            return self.criterion(outputs, targets)

    def _update_model(self, x: int, accumulate: int, loss_sum: torch.Tensor):
        """Update model parameters"""
        if x % accumulate == 0:
            self.amp_scale.unscale_(self.optimizer)
            self.model.clip_gradients()
            self.amp_scale.step(self.optimizer)
            self.amp_scale.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    def _save_checkpoint(self, is_best: bool):
        """Save model checkpoint"""
        save = {"model": copy.deepcopy(self.ema.ema_model).half()}
        torch.save(save, "./weights/last.pt")
        if is_best:
            torch.save(save, "./weights/best.pt")
        del save

    def _handle_validation(self, epoch: int, avg_losses: Tuple[AverageMeter, AverageMeter, AverageMeter]):
        """Handle validation and logging"""

        def validate_and_log():
            last = validate(self.model, data_dir=self.config.data_dir)
            with open("weights/step.csv", "a") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["epoch", "box", "dfl", "cls", "Recall", "Precision", "mAP@50", "mAP"]
                )
                writer.writerow(
                    {
                        "epoch": str(epoch + 1).zfill(3),
                        "box": f"{avg_losses[0].avg:.3f}",
                        "cls": f"{avg_losses[1].avg:.3f}",
                        "dfl": f"{avg_losses[2].avg:.3f}",
                        "mAP": f"{last[0]:.3f}",
                        "mAP@50": f"{last[1]:.3f}",
                        "Recall": f"{last[2]:.3f}",
                        "Precision": f"{last[2]:.3f}",
                    }
                )

            if last[0] > self.best_map:
                self.best_map = last[0]
                self._save_checkpoint(is_best=True)
            else:
                self._save_checkpoint(is_best=False)

        self.primary.run(validate_and_log)

    def train(self):
        """Main training loop"""
        accumulate = max(round(64 / (self.config.batch_size * self.config.world_size)), 1)

        for epoch in range(self.config.epochs):
            self.model.train()
            if self.config.distributed:
                self.loader.sampler.set_epoch(epoch)
            if self.config.epochs - epoch == 10:
                self.loader.dataset.mosaic = False

            p_bar = enumerate(self.loader)
            self.primary.print(f'\n{"epoch":10s}{"memory":10s}{"box":10s}{"cls":10s}{"dfl":10s}')
            p_bar = self.primary.get_progress_bar(p_bar, len(self.loader), f"Epoch {epoch+1}/{self.config.epochs}")

            avg_box_loss = AverageMeter()
            avg_dfl_loss = AverageMeter()
            avg_cls_loss = AverageMeter()

            for i, (samples, targets) in p_bar:
                x = i + len(self.loader) * epoch
                accumulate = self._update_warmup(x, epoch, accumulate)

                loss_box, loss_cls, loss_dfl = self._train_step(samples, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))

                scale_factor = self.config.batch_size * self.config.world_size
                losses = [loss_box, loss_dfl, loss_cls]
                for j, loss in enumerate(losses):
                    losses[j] *= scale_factor

                self.amp_scale.scale(sum(losses)).backward()
                self._update_model(x, accumulate, sum(losses))

                def update_progress():
                    memory = f"{torch.cuda.memory_reserved() / 1E9:.3g}G"
                    s = (
                        f"{epoch + 1:>10}/{self.config.epochs:<10}"
                        f"{memory:>10}"
                        f"{avg_box_loss.avg:>10.3g}"
                        f"{avg_cls_loss.avg:>10.3g}"
                        f"{avg_dfl_loss.avg:>10.3g}"
                    )
                    p_bar.set_description(s)

                self.primary.run(update_progress)

            self.scheduler.step()
            self._handle_validation(epoch, (avg_box_loss, avg_cls_loss, avg_dfl_loss))

        def cleanup():
            self.strip_optimizer("./weights/best.pt")
            self.strip_optimizer("./weights/last.pt")

        self.primary.run(cleanup)
        torch.cuda.empty_cache()
