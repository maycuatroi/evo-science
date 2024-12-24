# Evolution Science

[![codecov](https://codecov.io/gh/maycuatroi/evo-science/branch/main/graph/badge.svg?token=evo-science_token_here)](https://codecov.io/gh/maycuatroi/evo-science)
[![CI](https://github.com/maycuatroi/evo-science/actions/workflows/main.yml/badge.svg)](https://github.com/maycuatroi/evo-science/actions/workflows/main.yml)

Awesome evo_science created by maycuatroi

## Install it from PyPI

```bash
pip install evo-science
```

## Example
```python
    model = LinearRegressionModel()

    x = FeatureSet(features=[PClass, Sex, Age, SibSp, Parch, Fare])
    y = FeatureSet(features=[Survived])

    (x + y).build(
        csv_path="https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    )

    model.fit(x=x, y=y)
    model.evaluate(x=x, y=y, metrics=[Slope, ErrorStd])
    model.calculate_coefficients(x=x)
```

## YOLO Object Detection

The library includes a comprehensive implementation of YOLO (You Only Look Once) object detection models, including YOLOv8. The implementation features:

- Full YOLOv8 architecture with backbone, neck (FPN), and detection head
- Distributed training support
- Real-time object detection with webcam
- Model profiling and EMA (Exponential Moving Average) support
- Custom loss functions including DFL (Distribution Focal Loss)

### YOLO Example

```python
from evo_science.packages.yolo.yolo_v8 import YoloV8

# Initialize YOLOv8-nano model
model = YoloV8.yolo_v8_n(num_classes=80)  # 80 classes for COCO dataset

# For training
from evo_science.packages.yolo.modules.trainer import Trainer, TrainerConfig

config = TrainerConfig(
    data_dir="path/to/coco",
    batch_size=32,
    epochs=300,
    input_size=640
)

trainer = Trainer(model, config)
trainer.train()

# For real-time detection using webcam
from evo_science.packages.yolo.modules.demo import demo

demo(input_size=640, model=model)
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
