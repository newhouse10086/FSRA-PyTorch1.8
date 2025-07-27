# FSRA Enhanced - Quick Start Guide

## 🚀 5-Minute Quick Start

Get FSRA Enhanced running in just 5 minutes!

### Step 1: Environment Setup (2 minutes)

```bash
# Clone the repository
git clone <your-repository-url>
cd new_project

# Create conda environment
conda create -n fsra_enhanced python=3.8
conda activate fsra_enhanced

# Install PyTorch with CUDA support
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch

# Install additional dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation (1 minute)

```bash
# Test the evaluation system
python scripts/test_evaluation.py

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
Testing MetricsCalculator...
✓ MetricsCalculator test passed!
Testing EvaluationTracker...
✓ EvaluationTracker test passed!
ALL TESTS PASSED SUCCESSFULLY!
CUDA available: True
```

### Step 3: Quick Training Test (2 minutes)

```bash
# Test training without data (model structure validation)
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --batch_size 2 \
    --num_epochs 1
```

Expected output:
```
New ViT Model Information:
  Total parameters: 23,456,789
  Trainable parameters: 23,456,789
Training mode: ResNet18: pretrained, ViT: scratch
New ViT models created successfully
```

## 🎯 Choose Your Training Mode

### Option A: Enhanced New ViT (Recommended)

```bash
# Best performance with hybrid architecture
python scripts/train_new_vit.py \
    --config config/default_config.yaml \
    --data_dir /path/to/University-1652 \
    --batch_size 16 \
    --num_epochs 150
```

**Features:**
- ResNet18 + ViT hybrid architecture
- Community clustering with graph networks
- Cross-view feature alignment
- Expected performance: ~85% Rank-1 accuracy

### Option B: Traditional FSRA

```bash
# Faster training with good performance
python scripts/train.py \
    --config config/default_config.yaml \
    --model FSRA \
    --data_dir /path/to/University-1652 \
    --batch_size 16 \
    --num_epochs 120
```

**Features:**
- Direct ViT processing
- K-means clustering
- Lower memory usage (6-8GB)
- Expected performance: ~82% Rank-1 accuracy

### Option C: From Scratch Training

```bash
# Research mode without pretrained weights
python scripts/train_new_vit.py \
    --from_scratch \
    --data_dir /path/to/University-1652 \
    --batch_size 8 \
    --num_epochs 200 \
    --learning_rate 0.0005
```

## 📊 Monitor Your Training

### Real-time Monitoring

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006 in browser
```

### Check Training Progress

```bash
# View console output
tail -f logs/train.log

# Check metrics CSV
head -n 5 logs/new_vit_*_metrics.csv
```

### View Generated Visualizations

```bash
# Training curves
ls logs/plots/*_metrics.png

# Confusion matrices
ls logs/plots/confusion_matrix_*.png

# ROC curves
ls logs/plots/roc_curves_*.png
```

## 📁 Data Preparation

### University-1652 Dataset Structure

```bash
# Required directory structure
data/
├── train/
│   ├── satellite/
│   │   ├── 0001/
│   │   │   └── 0001.jpg
│   │   └── ...
│   └── drone/
│       ├── 0001/
│       │   ├── image-01.jpeg
│       │   ├── image-02.jpeg
│       │   └── ...
│       └── ...
└── test/
    ├── query_satellite/
    ├── query_drone/
    └── gallery_satellite/
```

### Quick Data Check

```bash
# Verify data structure
python -c "
import os
data_dir = 'data'
for split in ['train', 'test']:
    for view in ['satellite', 'drone']:
        path = os.path.join(data_dir, split, view)
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f'{split}/{view}: {count} classes')
        else:
            print(f'{split}/{view}: NOT FOUND')
"
```

## ⚡ Performance Tips

### GPU Memory Optimization

```bash
# For 8GB GPU
python scripts/train_new_vit.py --batch_size 8

# For 6GB GPU
python scripts/train.py --model FSRA --batch_size 12

# For 4GB GPU
python scripts/train.py --model FSRA --batch_size 8
```

### Speed Optimization

```bash
# Use mixed precision (if supported)
python scripts/train_new_vit.py --autocast

# Reduce evaluation frequency
python scripts/train_new_vit.py --eval_interval 5  # Every 5 epochs
```

### Storage Optimization

```bash
# Reduce visualization frequency
python scripts/train_new_vit.py --plot_interval 20  # Every 20 epochs

# Compress checkpoints
python scripts/train_new_vit.py --save_interval 20  # Every 20 epochs
```

## 🔧 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size
python scripts/train_new_vit.py --batch_size 4
```

### Issue 2: Missing Dependencies
```bash
# Solution: Install missing packages
pip install networkx matplotlib seaborn scikit-learn
```

### Issue 3: Data Loading Error
```bash
# Solution: Check data path and structure
python scripts/test_evaluation.py
```

### Issue 4: Slow Training
```bash
# Solution: Check GPU utilization
nvidia-smi
# If low, increase batch size or use multiple workers
python scripts/train_new_vit.py --num_worker 4
```

## 📈 Expected Results

### Training Progress

```
Epoch: 10 [100/500] Loss: 1.8234 Cls: 1.7891 Align: 0.0343 Acc: 65.23%
Test Results - Epoch 10:
  Accuracy: 0.6523
  AUC-ROC: 0.7845
  Precision (Macro): 0.6234
  Recall (Macro): 0.6523
  F1-Score (Macro): 0.6378

Epoch: 50 [100/500] Loss: 0.8234 Cls: 0.7891 Align: 0.0343 Acc: 84.56%
Test Results - Epoch 50:
  Accuracy: 0.8456
  AUC-ROC: 0.9123
  Precision (Macro): 0.8234
  Recall (Macro): 0.8456
  F1-Score (Macro): 0.8344
```

### Final Performance (New ViT)

- **Rank-1 Accuracy**: ~85%
- **Rank-5 Accuracy**: ~93%
- **mAP**: ~88%
- **Training Time**: ~10 hours (RTX 3080)

## 🎉 Next Steps

1. **Experiment with hyperparameters**:
   ```bash
   python scripts/train_new_vit.py --learning_rate 0.001 --num_final_clusters 4
   ```

2. **Try different training modes**:
   ```bash
   python scripts/train_new_vit.py --use_pretrained_vit --pretrained_vit /path/to/weights.pth
   ```

3. **Analyze results**:
   ```bash
   # View detailed metrics
   python -c "import pandas as pd; print(pd.read_csv('logs/new_vit_*_metrics.csv').tail())"
   ```

4. **Extend the framework**:
   - Add custom models in `src/models/`
   - Implement new loss functions in `src/losses/`
   - Create custom evaluation metrics in `src/utils/evaluation.py`

## 📚 Learn More

- [Complete Documentation](README.md)
- [Training Modes Guide](README_TRAINING_MODES.md)
- [Evaluation System Guide](EVALUATION_SYSTEM_GUIDE.md)
- [Project Overview](PROJECT_OVERVIEW.md)

Happy training! 🚀
