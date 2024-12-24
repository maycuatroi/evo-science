import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import multiprocessing
import matplotlib.pyplot as plt
from rich.progress import Progress
from rich.console import Console
from rich.table import Table
from torch.optim import Adam


class MiniYOLOv8(nn.Module):
    def __init__(self, num_classes, S=7, B=2):
        super(MiniYOLOv8, self).__init__()
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per grid cell
        self.C = num_classes  # Number of classes

        # Backbone with downsampling to reduce spatial dimensions to S x S
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # Output: 224x224
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Output: 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Output: 28x28
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # Output: 14x14
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # Output: 7x7
            nn.ReLU(),
        )

        # Detection head
        # Output channels: B * (5 + C)
        self.head = nn.Conv2d(512, self.B * (5 + self.C), 1)  # Output: [batch_size, B*(5+C), 7, 7]

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        # Reshape x to [batch_size, S, S, B*(5+C)]
        x = x.permute(0, 2, 3, 1)  # From [batch_size, channels, height, width] to [batch_size, height, width, channels]
        x = x.view(-1, self.S, self.S, self.B, 5 + self.C)
        return x


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=90, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.ce_loss = nn.CrossEntropyLoss()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        # predictions: [batch_size, S, S, B, 5 + C]
        # target: [batch_size, S, S, B, 5 + C]

        # Extract target components
        obj_mask = target[..., 4] > 0  # Object mask
        noobj_mask = target[..., 4] == 0  # No object mask

        # Localization loss
        loc_loss = self.lambda_coord * self.mse_loss(predictions[obj_mask][..., 0:4], target[obj_mask][..., 0:4])

        # Confidence loss
        conf_loss_obj = self.mse_loss(predictions[obj_mask][..., 4], target[obj_mask][..., 4])

        conf_loss_noobj = self.lambda_noobj * self.mse_loss(predictions[noobj_mask][..., 4], target[noobj_mask][..., 4])

        # Classification loss
        class_loss = self.ce_loss(
            predictions[obj_mask][..., 5:].view(-1, self.C), torch.argmax(target[obj_mask][..., 5:], dim=-1).view(-1)
        )

        # Total loss
        total_loss = loc_loss + conf_loss_obj + conf_loss_noobj + class_loss
        return total_loss


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    annotations = [item[1] for item in batch]  # List of annotations for each image
    target_tensors = []

    S = 7  # Grid size
    B = 2  # Number of bounding boxes per grid cell
    C = 90  # Number of classes (including background)

    for i, annotation in enumerate(annotations):
        target_tensor = torch.zeros((S, S, B, 5 + C))
        for obj in annotation:
            bbox = obj["bbox"]  # [x_min, y_min, width, height]
            label = obj["category_id"]  # COCO labels start from 1

            # Ensure label is within the valid range
            if label < 1 or label > C:
                continue  # Skip this object if the label is out of range

            x_min, y_min, width, height = bbox
            x_center = x_min + width / 2
            y_center = y_min + height / 2

            img_width, img_height = images.size(3), images.size(2)
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            grid_x = int(x_center * S)
            grid_y = int(y_center * S)

            grid_x = max(0, min(grid_x, S - 1))
            grid_y = max(0, min(grid_y, S - 1))

            # Assign object to grid cell
            for b in range(B):
                if target_tensor[grid_y, grid_x, b, 4] == 0:  # if there is no object in the cell
                    target_tensor[grid_y, grid_x, b, 0] = x_center * S - grid_x  # x offset within the cell
                    target_tensor[grid_y, grid_x, b, 1] = y_center * S - grid_y  # y offset within the cell
                    target_tensor[grid_y, grid_x, b, 2] = width
                    target_tensor[grid_y, grid_x, b, 3] = height
                    target_tensor[grid_y, grid_x, b, 4] = 1  # Confidence score
                    target_tensor[grid_y, grid_x, b, 5 + label - 1] = (
                        1  # One-hot encoding for class label (subtract 1 as labels start from 1)
                    )
                    break  # Only assign one bounding box per object
        target_tensors.append(target_tensor)
    target_tensors = torch.stack(target_tensors)
    return images, target_tensors


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    console = Console()

    # Set up COCO dataset loader
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Loading COCO dataset
    console.print("Loading COCO dataset...", style="bold green")
    coco_train = torchvision.datasets.CocoDetection(
        root="./data/COCO/train2017", annFile="./data/COCO/annotations/instances_train2017.json", transform=transform
    )

    coco_loader = DataLoader(coco_train, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    console.print(f"COCO dataset loaded. Number of samples: {len(coco_train)}", style="bold green")

    # Define the model, loss function, and optimizer
    num_classes = 90  # COCO dataset has 90 classes (including background)
    S = 7
    B = 2
    model = MiniYOLOv8(num_classes, S=S, B=B)
    criterion = YoloLoss(S=S, B=B, C=num_classes)
    optimizer = Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    console.print(f"Model moved to device: {device}", style="bold green")

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        console.print(f"Starting epoch {epoch+1}/{num_epochs}", style="bold blue")
        model.train()
        running_loss = 0.0

        with Progress() as progress:
            task = progress.add_task(f"Epoch {epoch+1}/{num_epochs} Training", total=len(coco_loader))

            for i, (images, targets) in enumerate(coco_loader):
                images = images.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 0:
                    progress.update(task, advance=10)
                    console.print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(coco_loader)}], Loss: {loss.item():.4f}",
                        style="bold yellow",
                    )
                progress.update(task, advance=1)

        # Display progress summary as a table
        table = Table(title=f"Epoch {epoch+1} Summary")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold magenta")
        table.add_row("Running Loss", f"{running_loss / len(coco_loader):.4f}")
        console.print(table)

    console.print("Finished Training", style="bold green")
