# YOLO Implementation Documentation

This package provides a comprehensive implementation of the YOLO (You Only Look Once) object detection model, specifically focusing on YOLOv8 architecture. The implementation is designed for both training and inference, with support for distributed training and real-time detection.

## Architecture Overview

### Model Components

#### 1. Backbone (DarkNet)
The backbone is implemented in `layers/darknet.py` and consists of:
- CSP (Cross-Stage-Partial) layers for feature extraction
- Depth-wise separable convolutions for efficiency
- Residual connections for better gradient flow

#### 2. Neck (DarkFPN)
The Feature Pyramid Network (FPN) implemented in `layers/dark_fpn.py`:
- Multi-scale feature fusion
- Bottom-up and top-down pathways
- Skip connections for preserving spatial information

#### 3. Detection Head
The detection head (`layers/head.py`) includes:
- Multi-scale prediction
- Distribution Focal Loss (DFL) for better bounding box regression
- Class-specific predictions with confidence scores

### Key Components

#### Layers
- **Conv Layer**: Basic convolution block with batch normalization and activation
- **CSP Layer**: Cross-Stage-Partial layers for efficient feature extraction
- **DFL Layer**: Distribution Focal Loss layer for bounding box regression
- **Residual Block**: Skip connections for better gradient flow

#### Loss Functions
The loss computation (`losses/yolo_loss.py`) consists of three primary components that work together to optimize object detection performance:

##### 1. Classification Loss (BCE)
Binary Cross-Entropy loss for object classification:

```math
\mathcal{L}_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} [y_{ic} \log(\hat{y}_{ic}) + (1-y_{ic})\log(1-\hat{y}_{ic})]
```

where:
- $N$ is the number of predictions
- $C$ is the number of classes
- $y_{ic}$ is the ground truth for class $c$ in prediction $i$
- $\hat{y}_{ic}$ is the predicted probability for class $c$ in prediction $i$

##### 2. Box Regression Loss (IoU)
IoU-based loss for bounding box regression:

```math
\mathcal{L}_{box} = 1 - \frac{|B_{pred} \cap B_{gt}|}{|B_{pred} \cup B_{gt}|}
```

where:
- $B_{pred}$ is the predicted bounding box
- $B_{gt}$ is the ground truth bounding box
- $|\cdot|$ denotes the area operation

##### 3. Distribution Focal Loss (DFL)
Quality-aware loss for bounding box distribution:

```math
\mathcal{L}_{dfl} = -\sum_{i=1}^{n} \sum_{j=1}^{K} y_{ij} \log(\hat{p}_{ij}) \cdot |j - \hat{j}|^\alpha
```

where:
- $K$ is the number of discretized levels
- $y_{ij}$ is the ground truth distribution
- $\hat{p}_{ij}$ is the predicted distribution
- $\alpha$ is the focusing parameter (default: 2)
- $\hat{j}$ is the predicted level with highest probability

##### Total Loss
The final loss is a weighted combination of the three components:

```math
\mathcal{L}_{total} = \lambda_{box}\mathcal{L}_{box} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{dfl}\mathcal{L}_{dfl}
```

where:
- $\lambda_{box}$ is the box loss gain (default: 7.5)
- $\lambda_{cls}$ is the classification loss gain (default: 0.5)
- $\lambda_{dfl}$ is the DFL gain (default: 1.5)

##### Target Assignment
The dynamic target assignment process uses a quality-based matching strategy:

```math
Q_{ij} = \sqrt{\max(0, \text{IoU}_{ij})} \cdot \exp(-\frac{d_{ij}^2}{2\sigma^2})
```

where:
- $Q_{ij}$ is the matching quality between prediction $i$ and ground truth $j$
- $\text{IoU}_{ij}$ is the intersection over union
- $d_{ij}$ is the center distance
- $\sigma$ is the standard deviation of the Gaussian kernel

The top-k matches are selected based on $Q_{ij}$ scores, where k is dynamically adjusted based on object size and image complexity.

## Training Methodology

### Optimizer Configuration
The training uses AdamW optimizer with the following default settings:
- Learning rate: 0.01
- Weight decay: 0.0005
- Momentum beta1: 0.937
- Momentum beta2: 0.999

### Learning Rate Scheduling
- Cosine learning rate decay
- Warm-up period of 5 epochs
- Initial warm-up bias learning rate: 0.1
- Warm-up momentum: 0.8

### Training Features
1. **Distributed Training**
   - Multi-GPU support using PyTorch DDP
   - Automatic batch size scaling
   - Synchronized batch normalization

2. **Model EMA (Exponential Moving Average)**
   - Maintains a moving average of model weights
   - Improves model stability and generalization

3. **Data Augmentation**
   - Mosaic augmentation
   - Random affine transformations
   - Adaptive image scaling
   - Random horizontal flip

## Metrics and Evaluation

### Detection Metrics
- **Mean Average Precision (mAP)**: Primary metric for model evaluation
- **IoU Thresholds**: Multiple IoU thresholds (0.5, 0.75) for comprehensive evaluation
- **Precision and Recall**: Per-class metrics

### Performance Profiling
The profiler module (`modules/profiler.py`) provides:
- FPS measurement
- Parameter count
- FLOPS calculation
- Memory usage analysis

## Usage Examples

### Training Configuration
```python
config = TrainerConfig(
    data_dir="path/to/coco",
    batch_size=32,
    epochs=300,
    input_size=640,
    world_size=1,  # Number of GPUs
    distributed=False,
    lrf=0.2,  # Final learning rate fraction
    lr0=0.01,  # Initial learning rate
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    warmup_bias_lr=0.1,
    warmup_momentum=0.8,
)
```

### Distributed Training
```python
# Initialize distributed training
handle_distributed()

# Create model and trainer
model = YoloV8.yolo_v8_n(num_classes=80)
trainer = Trainer(model, config)
trainer.train()
```

### Real-time Detection
```python
from evo_science.packages.yolo.modules.demo import demo

model = YoloV8.yolo_v8_n(num_classes=80)
# Load weights if needed
model.load_state_dict(torch.load('weights/best.pt'))
demo(input_size=640, model=model)
```


## References

1. Original YOLO: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
2. YOLOv4: [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
3. YOLOv7: [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
4. Distribution Focal Loss: [Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection](https://arxiv.org/abs/2011.12885)
5. CSP Networks: [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)
