from typing import Optional, Tuple

import torch
import tqdm
from torch import nn
from torch.utils import data

from evo_science.datasets.coco_dataset import CocoDataset
from evo_science.entities.metrics.ap import AP
from evo_science.entities.metrics.mAP import MAP
from evo_science.entities.utils import wh2xy
from evo_science.entities.utils.nms import NonMaxSuppression
from evo_science.packages.yolo.modules.validator.config import ValidatorConfig


class Validator:
    """YOLO model validator with support for mAP calculation"""

    def __init__(self, model: Optional[nn.Module], config: ValidatorConfig):
        self.config = config
        self.model = model
        self.dataset = None
        self.loader = None
        self.nms = None
        self.iou_vector = None
        self.map_calculator = None

    def setup(self):
        """Initialize validation components"""
        self._setup_dataset()
        self._setup_model()
        self._setup_tools()

    def _setup_dataset(self):
        """Setup validation dataset and dataloader"""
        self.dataset = CocoDataset(
            data_dir=self.config.data_dir, input_size=self.config.input_size, is_augment=False, data_type="val"
        )
        self.loader = data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.dataset.collate_fn,
        )

    def _setup_model(self):
        """Setup model if not provided"""
        if self.model is None:
            self.model = torch.load("./weights/best.pt", map_location="cuda")["model"].float()

        self.model.half()
        self.model.eval()

    def _setup_tools(self):
        """Setup validation tools"""
        self.iou_vector = torch.linspace(self.config.iou_min, self.config.iou_max, self.config.iou_steps).cuda()
        self.nms = NonMaxSuppression(self.config.conf_thres, self.config.iou_thres)
        self.map_calculator = MAP(iou_thresholds=self.iou_vector, model=self.model)

    def _process_batch(
        self, samples: torch.Tensor, targets: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single batch of data"""
        samples = samples.cuda().half() / 255.0
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h)).cuda()

        outputs = self.model(samples)
        outputs = self.nms(outputs)

        metrics = []
        n_iou = self.iou_vector.numel()

        for i, output in enumerate(outputs):
            idx = targets["idx"] == i
            cls = targets["cls"][idx].cuda()
            box = targets["box"][idx].cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue

            if cls.shape[0]:
                target = torch.cat((cls, wh2xy(box) * scale), 1)
                metric = self.map_calculator.calculate(y_true=target, y_pred=output[:, :6])

            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

        return tuple(torch.cat(x, 0).cpu().numpy() for x in zip(*metrics))

    def _compute_metrics(self, metrics) -> Tuple[float, float, float, float]:
        """Compute final metrics"""
        if len(metrics) and metrics[0].any():
            ap = AP(model=self.model, iou_thresholds=self.iou_vector)
            tp, fp, precision, recall, map50, mean_ap = ap.calculate(*metrics)
            return mean_ap, map50, recall, precision
        return 0.0, 0.0, 0.0, 0.0

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, float]:
        """Run validation and return metrics"""
        self.setup()
        metrics = []

        p_bar = tqdm.tqdm(self.loader, desc=f'{"":10s}{"precision":10s}{"recall":10s}{"mAP50":10s}{"mAP":10s}')

        for samples, targets in p_bar:
            batch_metrics = self._process_batch(samples, targets)
            metrics.append(batch_metrics)

        # Concatenate batch metrics
        metrics = [torch.from_numpy(torch.cat([x[i] for x in metrics], 0)) for i in range(len(metrics[0]))]

        mean_ap, map50, recall, precision = self._compute_metrics(metrics)

        # Print results
        print(f'{"":10s}{precision:10.3g}{recall:10.3g}{map50:10.3g}{mean_ap:10.3g}')

        # Prepare model for training
        self.model.float()
        return mean_ap, map50, recall, precision
