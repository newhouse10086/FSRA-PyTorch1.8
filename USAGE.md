# FSRA Usage Guide

This guide provides detailed instructions for using the refactored FSRA project.

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n fsra_pytorch18 python=3.8
conda activate fsra_pytorch18

# Install PyTorch 1.8 with CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

1. Download the University-1652 dataset
2. Extract to your desired location
3. Update the data paths in `config/default_config.yaml`:

```yaml
data:
  data_dir: "/path/to/University-1652/train"
  test_dir: "/path/to/University-1652/test"
```

### 3. Test Setup

```bash
python scripts/test_setup.py
```

### 4. Training

```bash
python scripts/train.py --config config/default_config.yaml
```

## Configuration

### Configuration File Structure

The configuration is organized into four main sections:

#### Model Configuration
```yaml
model:
  name: "FSRA"
  backbone: "vit_small_patch16_224"
  num_classes: 701
  block_size: 3
  share_weights: true
  return_features: true
  dropout_rate: 0.1
```

#### Data Configuration
```yaml
data:
  data_dir: "data/train"
  test_dir: "data/test"
  batch_size: 14
  num_workers: 1
  image_height: 256
  image_width: 256
  views: 2
  sample_num: 7
  pad: 0
  color_jitter: false
  random_erasing_prob: 0.0
```

#### Training Configuration
```yaml
training:
  num_epochs: 120
  learning_rate: 0.01
  weight_decay: 0.0005
  momentum: 0.9
  warm_epochs: 0
  lr_scheduler_steps: [70, 110]
  lr_scheduler_gamma: 0.1
  triplet_loss_weight: 0.3
  kl_loss_weight: 0.0
  use_kl_loss: false
  use_fp16: false
  use_autocast: false
  use_data_augmentation: false
  moving_avg: 1.0
```

#### System Configuration
```yaml
system:
  gpu_ids: "0"
  use_gpu: true
  seed: 42
  log_interval: 10
  save_interval: 10
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  pretrained_dir: "pretrained"
```

### Command Line Overrides

You can override configuration values from the command line:

```bash
python scripts/train.py \
  --config config/default_config.yaml \
  --batch_size 16 \
  --learning_rate 0.005 \
  --num_epochs 100 \
  --gpu_ids "0,1"
```

## Training

### Basic Training

```bash
python scripts/train.py --config config/default_config.yaml
```

### Resume Training

```bash
python scripts/train.py \
  --config config/default_config.yaml \
  --resume checkpoints/checkpoint_epoch_50.pth
```

### Using Pretrained Weights

```bash
python scripts/train.py \
  --config config/default_config.yaml \
  --pretrained pretrained/vit_small_p16_224-15ec54c9.pth
```

### Multi-GPU Training

```bash
python scripts/train.py \
  --config config/default_config.yaml \
  --gpu_ids "0,1,2,3"
```

## Testing and Evaluation

### Basic Testing

```bash
python scripts/test.py \
  --config config/default_config.yaml \
  --checkpoint checkpoints/best_model.pth
```

### Evaluation Metrics

```bash
python scripts/evaluate.py \
  --config config/default_config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --output_dir results/
```

## Model Architecture

### FSRA Model Components

1. **Vision Transformer Backbone**: ViT-Small with overlapping patches
2. **Multi-Scale Features**: Global and local feature extraction
3. **Cross Attention**: Feature alignment between different views
4. **Classification Heads**: Multiple classifiers for different granularities

### Key Features

- **Two-View Learning**: Satellite and drone image processing
- **Feature Segmentation**: Multi-scale feature extraction
- **Region Alignment**: Cross-attention mechanism for view alignment
- **Shared Weights**: Optional weight sharing between views

## Data Loading

### Dataset Structure

```
data/
├── train/
│   ├── satellite/
│   │   ├── 0001/
│   │   ├── 0002/
│   │   └── ...
│   └── drone/
│       ├── 0001/
│       ├── 0002/
│       └── ...
└── test/
    ├── query_satellite/
    ├── gallery_drone/
    └── ...
```

### Custom Datasets

To use your own dataset:

1. Organize data in the same structure as University-1652
2. Update the configuration file
3. Modify the dataset class if needed

## Monitoring and Logging

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

### Log Files

- Training logs: `logs/train_YYYYMMDD_HHMMSS.log`
- Metrics: `logs/metrics_YYYYMMDD_HHMMSS.csv`

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or image resolution
2. **Slow Training**: Increase number of workers or use SSD storage
3. **Poor Performance**: Check learning rate and data augmentation settings

### Performance Optimization

1. **Batch Size**: Use largest batch size that fits in memory
2. **Mixed Precision**: Enable autocast for faster training
3. **Data Loading**: Use multiple workers and pin memory
4. **Checkpointing**: Save checkpoints regularly

## Advanced Usage

### Custom Loss Functions

```python
from src.losses import CombinedLoss

loss_fn = CombinedLoss(
    num_classes=701,
    triplet_weight=0.3,
    kl_weight=0.1,
    use_kl_loss=True
)
```

### Custom Models

```python
from src.models import make_fsra_model

model = make_fsra_model(
    num_classes=701,
    block_size=4,
    views=2,
    share_weights=True
)
```

### Custom Data Transforms

```python
from src.datasets.transforms import get_train_transforms

sat_transform, drone_transform = get_train_transforms(config)
```

## API Reference

See the individual module documentation for detailed API information:

- `src.models`: Model definitions
- `src.datasets`: Data loading utilities
- `src.losses`: Loss functions
- `src.utils`: Utility functions
- `config`: Configuration management
