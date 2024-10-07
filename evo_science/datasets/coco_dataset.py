import json
import math
import os
import pickle
import random
from typing import List

from rich.progress import Progress, SpinnerColumn, TextColumn
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from evo_science.datasets.default_augmentation import DefaultAugmentation
from evo_science.datasets.lables.bbox_label import BBoxLabel
from evo_science.datasets.lables.image import Image
from evo_science.entities.vision.box_collection import BoxCollection

FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"


class COCODataset(data.Dataset):
    def __init__(self, data_dir, input_size=640, is_augment=True, data_type="train"):
        """
        Args:
            data_dir: The directory of the COCO dataset.
            input_size: The size of the input image.
            is_augment: Whether to use augmentation.
            data_type: The type of the dataset. ["train", "val", "test"]
        """
        self.data_dir = data_dir
        self.is_mosaic = is_augment
        self.is_augment = is_augment
        self.input_size = input_size
        self.data_type = data_type

        # Read labels

        self.images = self.load_labels()
        self.n = len(self.images)  # number of samples
        self.indices = range(self.n)
        self.augmentations = DefaultAugmentation()

    def __getitem__(self, index):
        index = self.indices[index]

        if self.is_mosaic:
            # Load MOSAIC
            image, label = self.load_mosaic(index)
            # MixUp augmentation
            if random.random() < params["mix_up"]:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, params)

                image, label = self.mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image
            image, shape = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = self.resize(image, self.input_size, self.is_augment)

            if label.size:
                box_collection = BoxCollection(label[:, 1:])
                label[:, 1:] = box_collection.wh2xy(
                    width=int(ratio[0] * w), height=int(ratio[1] * h), pad_width=pad[0], pad_height=pad[1]
                )
            if self.is_augment:
                image, label = self.random_perspective(image, label, params)

        nl = len(label)  # number of labels
        h, w = image.shape[:2]
        cls = label[:, 0:1]
        box_collection = BoxCollection(label[:, 1:5])
        box = box_collection.xy2wh(width=w, height=h)

        if self.is_augment:
            # Albumentations
            image, box, cls = self.augmentations(image, box, cls)
            nl = len(box)  # update after albumentations
            # HSV color-space
            self.augment_hsv(image, params)
            # Flip up-down
            if random.random() < params["flip_ud"]:
                image = np.flipud(image)
                if nl:
                    box[:, 1] = 1 - box[:, 1]
            # Flip left-right
            if random.random() < params["flip_lr"]:
                image = np.fliplr(image)
                if nl:
                    box[:, 0] = 1 - box[:, 0]

        target_cls = torch.zeros((nl, 1))
        target_box = torch.zeros((nl, 4))
        if nl:
            target_cls = torch.from_numpy(cls)
            target_box = torch.from_numpy(box)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)

        return torch.from_numpy(sample), target_cls, target_box, torch.zeros(nl)

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = cv2.imread(self.images[i].image_path)
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(
                image,
                dsize=(int(w * r), int(h * r)),
                interpolation=self.resample() if self.is_augment else cv2.INTER_LINEAR,
            )
        return image, (h, w)

    def load_mosaic(self, index):
        label4 = []
        border = [-self.input_size // 2, -self.input_size // 2]
        image4 = np.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=np.uint8)
        y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (None, None, None, None, None, None, None, None)

        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load image
            image, _ = self.load_image(index)
            shape = image.shape
            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            if i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            if i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            if i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], self.input_size * 2)
                y2a = min(self.input_size * 2, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])

            pad_w = (x1a - x1b) if all(v is not None for v in (x1a, x1b)) else 0
            pad_h = (y1a - y1b) if all(v is not None for v in (y1a, y1b)) else 0
            if all(v is not None for v in (y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b)):
                image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

        # Concat/clip labels
        label4 = np.concatenate(label4, 0)
        for x in label4[:, 1:]:
            np.clip(x, 0, 2 * self.input_size, out=x)

        # Augment
        image4, label4 = self.random_perspective(image4, label4, params, border)

        return image4, label4

    @staticmethod
    def collate_fn(batch):
        samples, cls, box, indices = zip(*batch)

        cls = torch.cat(cls, dim=0)
        box = torch.cat(box, dim=0)

        new_indices = list(indices)
        for i in range(len(indices)):
            new_indices[i] += i
        indices = torch.cat(new_indices, dim=0)

        targets = {"cls": cls, "box": box, "idx": indices}
        return torch.stack(samples, dim=0), targets

    def load_labels(self) -> List[Image]:
        """
        Loads the labels from the COCO dataset.
        Returns:
            List[Image]: The list of images with their labels.
        """

        images: List[Image] = []
        cache_path = f"{self.data_dir}/coco_bboxes_{self.data_type}.pkl"
        if os.path.exists(cache_path):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description="Loading cached labels...", total=None)
                images = pickle.load(open(cache_path, "rb"))
            return images

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            data = json.load(open(f"{self.data_dir}/annotations/instances_{self.data_type}2017.json", "r"))

            annotations = {}
            task1 = progress.add_task(description="Processing annotations...", total=len(data["annotations"]))
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                bbox = annotation["bbox"]
                category_id = annotation["category_id"]
                annotations[image_id] = (bbox, category_id)
                progress.update(task1, advance=1)

            categories = {}
            task2 = progress.add_task(description="Processing categories...", total=len(data["categories"]))
            for category in data["categories"]:
                categories[category["id"]] = category["name"]
                progress.update(task2, advance=1)

            task3 = progress.add_task(description="Processing images...", total=len(data["images"]))
            for image in data["images"]:
                image_id = image["id"]
                if image_id not in annotations:
                    continue
                width = image["width"]
                height = image["height"]
                image_path = f'{self.data_dir}/{self.data_type}2017/{image["file_name"]}'
                bboxes = []
                annotation = annotations[image_id]
                bbox, category_id = annotation
                bboxes.append(
                    BBoxLabel(
                        xmin=bbox[0],
                        ymin=bbox[1],
                        xmax=bbox[0] + bbox[2],
                        ymax=bbox[1] + bbox[3],
                        label=categories[category_id],
                    )
                )

                image = Image(image_path=image_path, labels=bboxes)
                images.append(image)
                progress.update(task3, advance=1)

            progress.add_task(description="Saving labels to cache...", total=None)
            pickle.dump(images, open(cache_path, "wb"))

        return images

    def resample(self):
        choices = (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4)
        return random.choice(seq=choices)

    def augment_hsv(self, image, params):
        # HSV color-space augmentation
        h = params["hsv_h"]
        s = params["hsv_s"]
        v = params["hsv_v"]

        r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype("uint8")
        lut_s = np.clip(x * r[1], 0, 255).astype("uint8")
        lut_v = np.clip(x * r[2], 0, 255).astype("uint8")

        hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed

    def resize(self, image, input_size, augment):
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(input_size / shape[0], input_size / shape[1])
        if not augment:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        pad = int(round(shape[1] * r)), int(round(shape[0] * r))
        w = (input_size - pad[0]) / 2
        h = (input_size - pad[1]) / 2

        if shape[::-1] != pad:  # resize
            image = cv2.resize(image, dsize=pad, interpolation=self.resample() if augment else cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
        return image, (r, r), (w, h)

    def random_perspective(self, image, label, params, border=(0, 0)):
        h = image.shape[0] + border[0] * 2
        w = image.shape[1] + border[1] * 2

        # Center
        center = np.eye(3)
        center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
        center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

        # Perspective
        perspective = np.eye(3)

        # Rotation and Scale
        rotate = np.eye(3)
        a = random.uniform(-params["degrees"], params["degrees"])
        s = random.uniform(1 - params["scale"], 1 + params["scale"])
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        shear = np.eye(3)
        shear[0, 1] = math.tan(random.uniform(-params["shear"], params["shear"]) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-params["shear"], params["shear"]) * math.pi / 180)

        # Translation
        translate = np.eye(3)
        translate[0, 2] = random.uniform(0.5 - params["translate"], 0.5 + params["translate"]) * w
        translate[1, 2] = random.uniform(0.5 - params["translate"], 0.5 + params["translate"]) * h

        # Combined rotation matrix, order of operations (right to left) is IMPORTANT
        matrix = translate @ shear @ rotate @ perspective @ center
        if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():  # image changed
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

        # Transform label coordinates
        n = len(label)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ matrix.T  # transform
            xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            box = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            box[:, [0, 2]] = box[:, [0, 2]].clip(0, w)
            box[:, [1, 3]] = box[:, [1, 3]].clip(0, h)
            # filter candidates
            box_collection = BoxCollection(label[:, 1:5].T * s)
            indices = box_collection.candidates(other_boxes=box.T)

            label = label[indices]
            label[:, 1:5] = box[indices]

        return image, label

    def mix_up(self, image1, label1, image2, label2):
        # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
        alpha = np.random.beta(a=32.0, b=32.0)  # mix-up ratio, alpha=beta=32.0
        image = (image1 * alpha + image2 * (1 - alpha)).astype(np.uint8)
        label = np.concatenate((label1, label2), 0)
        return image, label

    @staticmethod
    def download_coco(dataset_type="full", data_dir="data/COCO"):
        from evo_downloader.downloader import Downloader
        import zipfile

        downloader = Downloader(num_threads=10)
        if dataset_type == "full":
            file_urls = [
                # "http://images.cocodataset.org/zips/train2017.zip",
                # "http://images.cocodataset.org/zips/val2017.zip",
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
            ]
            downloader.download_files(file_urls, folder_name=data_dir)
            # unzip
            for file_url in file_urls:
                file_name = file_url.split("/")[-1]
                zip_path = os.path.join(data_dir, file_name)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                os.remove(zip_path)  # Remove the zip file after extraction


if __name__ == "__main__":
    dataset = COCODataset(data_dir="data/COCO", data_type="train")
    print(dataset[0])
