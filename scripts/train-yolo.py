import click
import copy
import csv
import os
import warnings
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
import tqdm
from torch.utils import data
from torch import nn
from torch.optim.sgd import SGD

from evo_science.datasets.coco_dataset import CocoDataset
from evo_science.entities.metrics.ap import AP
from evo_science.entities.metrics.mAP import MAP
from evo_science.entities.optimizers.ema import ExponentialMovingAverage
from evo_science.entities.utils import setup_multi_processes, setup_seed, wh2xy
from evo_science.entities.utils.average_meter import AverageMeter
from evo_science.entities.utils.nms import NonMaxSuppression
from evo_science.packages.yolo.losses.yolo_loss import YoloLoss
from evo_science.packages.yolo.yolo_v8 import YoloV8, Yolo

coco_dir = f"./data/COCO"
warnings.filterwarnings("ignore")

model = YoloV8.yolo_v8_n()  # build model architecture


def strip_optimizer(path: str):
    x = torch.load(path, map_location=torch.device("cpu"))
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, path)


def train(
    batch_size: int,
    epochs: int,
    world_size: int,
    input_size: int = 640,
    distributed: bool = False,
    lrf: float = 0.2,
    lr0: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    local_rank: int = 0,
    warmup_epochs: int = 5,
    warmup_bias_lr: float = 0.1,
    warmup_momentum: float = 0.8,
):
    global model
    # Model Yolo v8 n
    filenames = []
    for filename in os.listdir("./data/COCO/train2017"):
        filenames.append("./data/COCO/train2017/" + filename)

    sampler = None
    dataset = CocoDataset(filenames, input_size=input_size, is_augment=True, data_type="train")
    # dataset = Dataset(data_dir=coco_dir, input_size=input_size, is_augment=True, data_type="train")
    names = dataset.get_names()
    # model = model(len(names))
    # model = model.load_weight(checkpoint_path="./weights/v8_n.pt")
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (batch_size * world_size)), 1)
    weight_decay = 0.0005 * batch_size * world_size * accumulate / 64

    # Separate parameters into three groups: weights, biases, and batch norm weights
    weights, biases, bn_weights = [], [], []
    for module in model.modules():
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

    # Create optimizer with parameter groups
    optimizer = SGD(
        [
            {"params": biases, "lr": lr0},
            {"params": weights, "lr": lr0, "weight_decay": weight_decay},
            {"params": bn_weights},
        ],
        lr=lr0,
        momentum=momentum,
        nesterov=True,
    )

    # Scheduler
    lr = lambda x: ((1 - x / epochs) * (1.0 - lrf) + lrf)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = ExponentialMovingAverage(model) if local_rank == 0 else None
    sampler = None

    if distributed:
        sampler = data.distributed.DistributedSampler(dataset)  # type: ignore

    loader = data.DataLoader(
        dataset,
        batch_size,
        sampler is None,
        sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    if distributed:
        # DDP mode
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(module=model, device_ids=[local_rank], output_device=local_rank)

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    criterion = model.get_criterion()
    num_warmup = max(round(warmup_epochs * num_batch), 1000)
    with open("weights/step.csv", "w") as f:
        if local_rank == 0:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "box", "dfl", "cls", "Recall", "Precision", "mAP@50", "mAP"]
            )
            writer.writeheader()
        for epoch in range(epochs):
            model.train()
            if distributed:
                sampler.set_epoch(epoch)  # type: ignore
            if epochs - epoch == 10:
                loader.dataset.mosaic = False  # type: ignore

            p_bar = enumerate(loader)

            if local_rank == 0:
                print(f'\n{"epoch":10s}{"memory":10s}{"box":10s}{"cls":10s}{"dfl":10s}')
                p_bar = tqdm.tqdm(p_bar, total=num_batch, desc=f"Epoch {epoch+1}/{epochs}")

            optimizer.zero_grad()
            avg_box_loss = AverageMeter()
            avg_dfl_loss = AverageMeter()
            avg_cls_loss = AverageMeter()
            for i, (samples, targets) in p_bar:
                x = i + num_batch * epoch  # number of iterations
                samples = samples.cuda().float() / 255

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (batch_size * world_size)]
                    accumulate = max(1, np.interp(x, xp, fp).round())  # type: ignore
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [warmup_bias_lr, y["initial_lr"] * lr(epoch)]
                        else:
                            fp = [0.0, y["initial_lr"] * lr(epoch)]
                        y["lr"] = np.interp(x, xp, fp)
                        if "momentum" in y:
                            fp = [warmup_momentum, momentum]
                            y["momentum"] = np.interp(x, xp, fp)

                # Forward
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))

                scale_factor = batch_size * world_size
                losses = [loss_box, loss_dfl, loss_cls]
                for i, _ in enumerate(losses):
                    losses[i] *= scale_factor  # Scale loss by batch size and world size

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if x % accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    model.clip_gradients()  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                # Log
                if local_rank == 0:
                    memory = f"{torch.cuda.memory_reserved() / 1E9:.3g}G"  # (GB)
                    s = (
                        f"{epoch + 1:>10}/{epochs:<10}"
                        f"{memory:>10}"
                        f"{avg_box_loss.avg:>10.3g}"
                        f"{avg_cls_loss.avg:>10.3g}"
                        f"{avg_dfl_loss.avg:>10.3g}"
                    )
                    p_bar.set_description(s)  # type: ignore

            # Scheduler
            scheduler.step()

            if local_rank == 0:
                # mAP
                last = validate(epochs, model)
                writer.writerow(
                    {
                        "epoch": str(epoch + 1).zfill(3),
                        "box": str(f"{avg_box_loss.avg:.3f}"),
                        "cls": str(f"{avg_cls_loss.avg:.3f}"),
                        "dfl": str(f"{avg_dfl_loss.avg:.3f}"),
                        "mAP": str(f"{last[0]:.3f}"),
                        "mAP@50": str(f"{last[1]:.3f}"),
                        "Recall": str(f"{last[2]:.3f}"),
                        "Precision": str(f"{last[2]:.3f}"),
                    }
                )
                f.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {"model": copy.deepcopy(ema.ema).half()}

                # Save last, best and delete
                torch.save(save, "./weights/last.pt")
                if best == last[0]:
                    torch.save(save, "./weights/best.pt")
                del save

    if local_rank == 0:
        strip_optimizer("./weights/best.pt")  # strip optimizers
        strip_optimizer("./weights/last.pt")  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def validate(epochs, model=None):
    dataset = COCODataset(data_dir="", input_size=640, is_augment=False, data_type="val")
    loader = data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn
    )

    if model is None:
        model = torch.load("./weights/best.pt", map_location="cuda")["model"].float()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.0
    m_rec = 0.0
    map50 = 0.0
    mean_ap = 0.0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=f'{"":10s}{"precision":10s}{"recall":10s}{"mAP50":10s}{"mAP":10s}')
    nms = NonMaxSuppression(0.001, 0.7)
    mAP = MAP(iou_thresholds=iou_v, model=model)
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.0  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        outputs = model(samples)
        # NMS
        outputs = nms(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets["idx"] == i
            cls = targets["cls"][idx]
            box = targets["box"][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat((cls, wh2xy(box) * scale), 1)  # type: ignore
                metric = mAP.calculate(y_true=target, y_pred=output[:, :6])
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        ap = AP(model=model, iou_thresholds=iou_v)
        tp, fp, m_pre, m_rec, map50, mean_ap = ap.calculate(*metrics)
    # Print results
    print(f'{"":10s}{m_pre:10.3g}{m_rec:10.3g}{map50:10.3g}{mean_ap:10.3g}')
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


@torch.no_grad()
def demo(input_size: int, model: Yolo):
    camera = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    nms = NonMaxSuppression(conf_threshold=0.25, iou_threshold=0.7)
    while camera.isOpened():
        # Capture frame-by-frame
        success, frame = camera.read()
        if success:
            image = frame.copy()
            shape = image.shape[:2]

            r = input_size / max(shape[0], shape[1])
            if r != 1:
                resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
            height, width = image.shape[:2]

            # Scale ratio (new / old)
            r = min(1.0, input_size / height, input_size / width)

            # Compute padding
            pad = int(round(width * r)), int(round(height * r))
            w = np.mod((input_size - pad[0]), 32) / 2
            h = np.mod((input_size - pad[1]), 32) / 2

            if (width, height) != pad:  # resize
                image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border

            # Convert HWC to CHW, BGR to RGB
            x = image.transpose((2, 0, 1))[::-1]
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
            x = x.unsqueeze(dim=0)
            x = x.cuda()
            x = x.half()
            x = x / 255
            # Inference
            outputs = model(x)
            # NMS
            outputs = nms(outputs=outputs)
            for output in outputs:
                output[:, [0, 2]] -= w  # x padding
                output[:, [1, 3]] -= h  # y padding
                output[:, :4] /= min(height / shape[0], width / shape[1])

                output[:, 0].clamp_(0, shape[1])  # x1
                output[:, 1].clamp_(0, shape[0])  # y1
                output[:, 2].clamp_(0, shape[1])  # x2
                output[:, 3].clamp_(0, shape[0])  # y2

                for box in output:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    camera.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def profile(input_size: int, batch_size: int, num_classes: int):
    model.eval()
    model(torch.zeros((batch_size, 3, input_size, input_size)))
    params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {int(params)}")


@click.command()
@click.option("--input-size", default=640, type=int, help="Input size")
@click.option("--batch-size", default=32, type=int, help="Batch size")
@click.option("--epochs", default=300, type=int, help="Number of epochs")
@click.option("--is_train", is_flag=True, help="Run training", default=True)
@click.option("--is_test", is_flag=True, help="Run testing", default=False)
@click.option("--is_demo", is_flag=True, help="Run demo", default=False)
@click.option("--local-rank", default=0, type=int, help="Local rank")
@click.option("--world-size", default=1, type=int, help="World size")
@click.option("--distributed", is_flag=True, help="Distributed training")
def main(input_size, batch_size, epochs, is_train, is_test, is_demo, local_rank, world_size, distributed):
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

    profile(input_size=input_size, batch_size=batch_size, num_classes=2)

    if is_train:
        train(
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
        model = model()
        model.load_weight(checkpoint_path="./weights/best.pt")
        model.half()
        model.eval()
        validate(epochs, model)
    if is_demo:
        demo(input_size=input_size, model=model)


if __name__ == "__main__":
    main()
