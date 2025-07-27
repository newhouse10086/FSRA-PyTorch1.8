#!/bin/bash
# Linux deployment script for FSRA Enhanced project
# Compatible with Ubuntu 18.04, CUDA 10.2, cuDNN 7

set -e  # Exit on any error

echo "🚀 FSRA Enhanced - Linux Deployment Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems only."
    exit 1
fi

# Check Ubuntu version
if command -v lsb_release &> /dev/null; then
    UBUNTU_VERSION=$(lsb_release -rs)
    print_status "Detected Ubuntu version: $UBUNTU_VERSION"
    if [[ "$UBUNTU_VERSION" != "18.04" ]]; then
        print_warning "This project is optimized for Ubuntu 18.04. Current version: $UBUNTU_VERSION"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Step 1: Check system requirements
print_step "1. Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python version: $PYTHON_VERSION"
else
    print_error "Python 3 is not installed!"
    exit 1
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_status "CUDA version: $CUDA_VERSION"
    if [[ "$CUDA_VERSION" != "10.2" ]]; then
        print_warning "Expected CUDA 10.2, found $CUDA_VERSION"
    fi
else
    print_warning "CUDA not found in PATH. Make sure CUDA 10.2 is installed."
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_status "GPU: $GPU_INFO"
else
    print_warning "nvidia-smi not found. GPU may not be available."
fi

# Step 2: Setup conda environment
print_step "2. Setting up conda environment..."

if command -v conda &> /dev/null; then
    print_status "Conda found. Creating environment..."
    
    # Create conda environment
    conda env create -f environment.yml -n fsra_enhanced || {
        print_warning "Environment creation failed. Trying to update existing environment..."
        conda env update -f environment.yml -n fsra_enhanced
    }
    
    print_status "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate fsra_enhanced
    
else
    print_error "Conda not found! Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Step 3: Install PyTorch with CUDA support
print_step "3. Installing PyTorch 1.8 with CUDA 10.2..."

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch -y

# Step 4: Install additional dependencies
print_step "4. Installing additional dependencies..."

pip install -r requirements.txt

# Step 5: Verify installation
print_step "5. Verifying installation..."

python3 -c "
import torch
import torchvision
import numpy as np
import matplotlib
import seaborn
import sklearn
import networkx
import yaml
import tqdm

print('✓ All core packages imported successfully')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU count: {torch.cuda.device_count()}')
    print(f'✓ Current GPU: {torch.cuda.get_device_name(0)}')
"

# Step 6: Test project imports
print_step "6. Testing project imports..."

python3 -c "
import sys
import os
sys.path.append('.')

try:
    from config import load_config, Config, merge_config_with_args
    from src.models.model_factory import create_model, create_cross_attention_model
    from src.utils.evaluation import evaluate_model, EvaluationTracker
    print('✓ All project imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Step 7: Create necessary directories
print_step "7. Creating project directories..."

mkdir -p data/{train,test}
mkdir -p logs/{plots,tensorboard}
mkdir -p checkpoints
mkdir -p pretrained

print_status "Created directories: data, logs, checkpoints, pretrained"

# Step 8: Set permissions
print_step "8. Setting file permissions..."

chmod +x scripts/*.py
chmod +x *.sh

print_status "Set executable permissions for scripts"

# Step 9: Test basic functionality
print_step "9. Testing basic functionality..."

if [[ -f "test_fixed_imports.py" ]]; then
    python3 test_fixed_imports.py
else
    print_warning "test_fixed_imports.py not found, skipping basic test"
fi

# Final status
echo
echo "=========================================="
print_status "🎉 FSRA Enhanced deployment completed successfully!"
echo
echo "📋 Next steps:"
echo "1. Prepare your University-1652 dataset in the data/ directory"
echo "2. Activate the environment: conda activate fsra_enhanced"
echo "3. Start training:"
echo "   python scripts/train.py --config config/default_config.yaml --model FSRA --data_dir data"
echo "4. Monitor training:"
echo "   tensorboard --logdir logs/tensorboard"
echo
echo "📚 Documentation:"
echo "- Quick Start: cat QUICK_START.md"
echo "- Training Modes: cat README_TRAINING_MODES.md"
echo "- Evaluation Guide: cat EVALUATION_SYSTEM_GUIDE.md"
echo
echo "🔧 Troubleshooting:"
echo "- Check logs in logs/ directory"
echo "- Run test_fixed_imports.py for diagnostics"
echo "- Ensure CUDA 10.2 and cuDNN 7 are properly installed"
echo
print_status "Happy training! 🚀"
