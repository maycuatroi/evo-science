import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import multiprocessing
import sys

# Define a mini version of YOLOv8
class MiniYOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(MiniYOLOv8, self).__init__()
        # Simple backbone with Convolution and ReLU layers
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        # Detection head with convolutions for predicting boxes and class scores
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes + 4, 1),  # num_classes + 4 for bbox (x, y, w, h) and class scores
        )

    def forward(self, x):
        print("Forward pass input shape:", x.shape)
        x = self.backbone(x)
        print("After backbone shape:", x.shape)
        x = self.head(x)
        print("After head shape:", x.shape)
        return x

if __name__ == "__main__":
    # Ensure the collate_fn is picklable by defining it outside of the __main__ scope
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]  # List of annotations for each image
        return images, targets

    multiprocessing.set_start_method("spawn", force=True)
    
    # Set up COCO dataset loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Loading COCO dataset
    print("Loading COCO dataset...")
    coco_train = torchvision.datasets.CocoDetection(
        root='./data/COCO/train2017',
        annFile='./data/COCO/annotations/instances_train2017.json',
        transform=transform
    )

    coco_loader = DataLoader(coco_train, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate_fn)
    print("COCO dataset loaded. Number of samples:", len(coco_train))

    # Define the model, loss function, and optimizer
    num_classes = 80  # COCO dataset has 80 classes
    model = MiniYOLOv8(num_classes)
    criterion = nn.MSELoss()  # Simple MSE Loss for demonstration purposes (bounding box regression)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Model moved to device:", device)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for i, (images, targets) in enumerate(coco_loader):
            print(f"Processing batch {i+1}/{len(coco_loader)}")
            images = images.to(device)
            print("Images moved to device. Shape:", images.shape)
            
            # Creating dummy targets with fixed size for simplicity
            dummy_targets = torch.rand((images.size(0), num_classes + 4, 224, 224)).to(device)
            print("Dummy targets created. Shape:", dummy_targets.shape)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, dummy_targets)
            print("Loss computed:", loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Optimizer step completed")
            
            running_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(coco_loader)}], Loss: {loss.item():.4f}')

    print('Finished Training')