import os
import random
from typing import Dict, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from evo_science.datasets.abstract_dataset import AbstractDataset
from evo_science.datasets.transformers.image_transformer import ImageTransformer
from evo_science.datasets.transformers.box_transformer import BoxTransformer
from evo_science.datasets.augmenters.image_augmenter import ImageAugmenter

FORMATS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp")


class CocoDataset(AbstractDataset):
    def __init__(self, data_dir: str | Path, input_size: int = 640, is_augment: bool = True, data_type: str = "train"):
        self.data_type = data_type
        self.mosaic = is_augment
        self.augment = is_augment
        self.input_size = input_size

        data_dir = Path(data_dir)

        if data_type == "train":
            data_folder_name = "train2017"
        elif data_type == "val":
            data_folder_name = "val2017"
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        image_dir = data_dir / data_folder_name

        filenames = []
        for fmt in FORMATS:
            filenames.extend(list(image_dir.glob(f"*.{fmt}")))
        filenames = [str(f) for f in filenames]

        if not filenames:
            raise ValueError(f"No valid images found in {image_dir}")

        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())
        self.n = len(self.filenames)
        self.indices = range(self.n)

        self.image_transformer = ImageTransformer()
        self.box_transformer = BoxTransformer()
        self.image_augmenter = ImageAugmenter()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = self.indices[index]
        mosaic = self.mosaic and random.random() < 1

        if mosaic:
            image, label = self.load_mosaic(index)
            if random.random() < 0:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index)
                image, label = self.image_augmenter.mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            image, shape = self.load_image(index)
            h, w = image.shape[:2]
            image, ratio, pad = self.image_transformer.resize(image, self.input_size, self.augment)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = self.box_transformer.wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = self.image_transformer.random_perspective(image, label)

        nl = len(label)
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box = label[:, 1:5]
        box = self.box_transformer.xy2wh(box, w, h)

        if self.augment:
            image, box, cls = self.image_augmenter(image, box, cls)
            nl = len(box)
            self.image_augmenter.augment_hsv(image)
            if random.random() < 0:
                image = np.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            if random.random() < 0.5:
                image = np.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)

        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self) -> int:
        return len(self.filenames)

    def get_names(self) -> Dict[int, str]:
        return {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

    def load_image(self, i: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            interpolation = (
                random.choice(
                    [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
                )
                if self.augment
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=interpolation)
        return image, (h, w)

    def load_mosaic(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        label4 = []
        border = (-self.input_size // 2, -self.input_size // 2)
        image4 = np.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=np.uint8)

        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            image, _ = self.load_image(index)
            shape = image.shape
            if i == 0:
                x1a, y1a = max(xc - shape[1], 0), max(yc - shape[0], 0)
                x2a, y2a = xc, yc
            elif i == 1:
                x1a, y1a = xc, max(yc - shape[0], 0)
                x2a, y2a = min(xc + shape[1], self.input_size * 2), yc
            elif i == 2:
                x1a, y1a = max(xc - shape[1], 0), yc
                x2a, y2a = xc, min(self.input_size * 2, yc + shape[0])
            else:
                x1a, y1a = xc, yc
                x2a, y2a = min(xc + shape[1], self.input_size * 2), min(self.input_size * 2, yc + shape[0])

            x1b, y1b = shape[1] - (x2a - x1a), shape[0] - (y2a - y1a)
            x2b, y2b = shape[1], shape[0]

            pad_w = x1a - x1b
            pad_h = y1a - y1b
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            label = self.labels[index].copy()
            if len(label):
                label[:, 1:] = self.box_transformer.wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)

        label4 = np.concatenate(label4, 0)
        for x in label4[:, 1:]:
            np.clip(x, 0, 2 * self.input_size, out=x)

        image4, label4 = self.image_transformer.random_perspective(image4, label4, border)
        return image4, label4

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        samples, cls, box, indices = zip(*batch)
        cls = torch.cat(cls, 0)
        box = torch.cat(box, 0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, 0)

        targets = {"cls": cls, "box": box, "idx": indices}
        return torch.stack(samples, 0), targets

    @staticmethod
    def load_label(filenames: List[str]) -> Dict[str, np.ndarray]:
        path = f"{os.path.dirname(filenames[0])}.cache"
        if os.path.exists(path):
            return torch.load(path)

        x = {}
        for filename in filenames:
            try:
                with open(filename, "rb") as f:
                    image = Image.open(f)
                    image.verify()
                shape = image.size
                assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
                assert image.format and image.format.lower() in FORMATS, f"invalid image format {image.format}"

                a = f"{os.sep}images{os.sep}"
                b = f"{os.sep}labels{os.sep}"
                label_path = b.join(filename.rsplit(a, 1)).rsplit(".", 1)[0] + ".txt"

                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = np.array(label, dtype=np.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = np.unique(label, axis=0, return_index=True)
                        if len(i) < nl:
                            label = label[i]
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)
                if filename:
                    x[filename] = label
            except (FileNotFoundError, AssertionError):
                continue

        torch.save(x, path)
        return x
