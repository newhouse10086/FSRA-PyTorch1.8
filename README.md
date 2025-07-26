# FSRA - Refactored for PyTorch 1.8

A refactored version of the FSRA (Feature Segmentation and Region Alignment) project optimized for PyTorch 1.8 with CUDA 10.2 and cuDNN 7 on Ubuntu 18.04.

## Original Paper
[A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization](https://arxiv.org/abs/2201.09206), IEEE Transactions on Circuits and Systems for Video Technology.

## Environment Requirements

- PyTorch 1.8.x
- CUDA 10.2
- cuDNN 7
- Ubuntu 18.04

## Project Structure

```
new_project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   └── default_config.yaml  # Default configuration
├── src/                     # Source code
│   ├── __init__.py
│   ├── models/              # Model definitions
│   │   ├── __init__.py
│   │   ├── fsra/           # FSRA model components
│   │   ├── cross_attention/ # Cross attention model
│   │   └── backbones/      # Backbone networks
│   ├── datasets/           # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── transforms.py
│   │   └── university_dataset.py
│   ├── losses/             # Loss functions
│   │   ├── __init__.py
│   │   ├── triplet_loss.py
│   │   └── combined_loss.py
│   ├── optimizers/         # Optimizer configurations
│   │   ├── __init__.py
│   │   └── optimizer.py
│   ├── utils/              # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   └── metrics.py
│   └── trainer/            # Training logic
│       ├── __init__.py
│       ├── trainer.py
│       └── evaluator.py
├── scripts/                # Training and testing scripts
│   ├── train.py
│   ├── test.py
│   └── evaluate.py
├── data/                   # Data directory (symlink to original)
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── pretrained/             # Pretrained models
```

## Installation

1. Create a conda environment:
```bash
conda create -n fsra_pytorch18 python=3.8
conda activate fsra_pytorch18
```

2. Install PyTorch 1.8 with CUDA 10.2:
```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train.py --config config/default_config.yaml
```

### Testing
```bash
python scripts/test.py --config config/default_config.yaml --checkpoint checkpoints/best_model.pth
```

### Evaluation
```bash
python scripts/evaluate.py --config config/default_config.yaml --checkpoint checkpoints/best_model.pth
```

## Key Improvements

1. **PyTorch 1.8 Compatibility**: All code adapted for PyTorch 1.8 API
2. **Modular Structure**: Clean separation of concerns with organized modules
3. **Configuration Management**: YAML-based configuration system
4. **Better Error Handling**: Robust error handling and logging
5. **Code Quality**: Improved code readability and maintainability
6. **Documentation**: Comprehensive documentation and comments

## Changes from Original

- **PyTorch 1.8 Compatibility**: All code adapted for PyTorch 1.8 API, including proper handling of autocast and mixed precision training
- **Improved Project Structure**: Clean separation of concerns with organized modules (models, datasets, losses, utils)
- **Enhanced Configuration Management**: YAML-based configuration system replacing command-line arguments
- **Better Error Handling**: Robust error handling and comprehensive logging
- **Cleaner Data Loading Pipeline**: Refactored dataset classes with better transforms and sampling
- **Optimized for Target Environment**: Specifically optimized for CUDA 10.2, cuDNN 7, Ubuntu 18.04
- **Modular Design**: Easy to extend and modify individual components
- **Better Documentation**: Comprehensive docstrings and type hints throughout

## Testing the Setup

Before training, you can test if everything is set up correctly:

```bash
cd new_project
python scripts/test_setup.py
```

This will verify:
- Configuration loading
- Model creation and forward pass
- PyTorch compatibility
- Transform functionality

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**: Ensure you have CUDA 10.2 installed
2. **PyTorch Version**: Make sure you're using PyTorch 1.8.x
3. **Dataset Path**: Update the data paths in the configuration file
4. **Memory Issues**: Reduce batch size if you encounter OOM errors

### Performance Tips

1. **Batch Size**: Start with smaller batch sizes and increase gradually
2. **Number of Workers**: Adjust `num_workers` based on your CPU cores
3. **Mixed Precision**: Enable autocast for faster training (if supported)
4. **Data Loading**: Use SSD storage for faster data loading

## Contributing

When contributing to this refactored version:

1. Follow the existing code structure
2. Add type hints to new functions
3. Include comprehensive docstrings
4. Test compatibility with PyTorch 1.8
5. Update configuration files as needed

## License

Same as the original FSRA project.

## Acknowledgments

- Original FSRA paper authors
- PyTorch team for the deep learning framework
- University-1652 dataset creators
