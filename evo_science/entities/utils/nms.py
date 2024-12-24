import torch
import torchvision
from time import time
from typing import List
from torch import Tensor

from evo_science.entities.utils import wh2xy


class NonMaxSuppression:
    def __init__(
        self, conf_threshold: float, iou_threshold: float, max_wh: int = 7680, max_det: int = 300, max_nms: int = 30000
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_wh = max_wh
        self.max_det = max_det
        self.max_nms = max_nms

    def __call__(self, outputs: Tensor) -> List[Tensor]:
        bs = outputs.shape[0]
        nc = outputs.shape[1] - 4
        xc = outputs[:, 4 : 4 + nc].amax(1) > self.conf_threshold

        start = time()
        limit = 0.5 + 0.05 * bs

        output = [torch.zeros((0, 6), device=outputs.device)] * bs
        for index, x in enumerate(outputs):
            x = x.transpose(0, -1)[xc[index]]

            if not x.shape[0]:
                continue

            x = self._process_candidates(x, nc)

            if not x.shape[0]:
                continue
            x = x[x[:, 4].argsort(descending=True)[: self.max_nms]]

            x = self._batched_nms(x)

            output[index] = x
            if time() - start > limit:
                break

        return output

    def _process_candidates(self, x: Tensor, nc: int) -> Tensor:
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > self.conf_threshold).nonzero(as_tuple=False).T
            box_i = torch.as_tensor(box[i], dtype=torch.float32, device=x.device)
            conf_scores = torch.as_tensor(x[i, 4 + j, None], dtype=torch.float32, device=x.device)
            class_ids = j.to(dtype=torch.float32, device=x.device).unsqueeze(-1)
            x = torch.cat((box_i, conf_scores, class_ids), dim=1)
        else:
            conf, j = cls.max(1, keepdim=True)
            box = torch.as_tensor(box, dtype=torch.float32, device=x.device)
            x = torch.cat((box, conf, j.to(torch.float32)), dim=1)[conf.view(-1) > self.conf_threshold]
        return x

    def _batched_nms(self, x: Tensor) -> Tensor:
        class_offset = x[:, 5:6] * self.max_wh
        boxes, scores = x[:, :4] + class_offset, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, self.iou_threshold)
        return x[i[: self.max_det]]
