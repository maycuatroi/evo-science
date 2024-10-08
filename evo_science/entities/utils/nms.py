import torch
import torchvision
from time import time


class NonMaxSuppression:
    def __init__(self, conf_threshold, iou_threshold, max_wh=7680, max_det=300, max_nms=30000):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_wh = max_wh
        self.max_det = max_det
        self.max_nms = max_nms

    def __call__(self, outputs):
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
            if (time() - start) > limit:
                break

        return output

    def _process_candidates(self, x, nc):
        box, cls = x.split((4, nc), 1)
        box = self._wh2xy(box)
        if nc > 1:
            i, j = (cls > self.conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.conf_threshold]
        return x

    def _batched_nms(self, x):
        c = x[:, 5:6] * self.max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, self.iou_threshold)
        return x[i[: self.max_det]]

    @staticmethod
    def _wh2xy(x):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
