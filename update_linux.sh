#!/bin/bash
# Linux update script for FSRA Enhanced project
# Use this script to update the project from Git repository

set -e  # Exit on any error

echo "🔄 FSRA Enhanced - Linux Update Script"
echo "======================================"

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

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not in a Git repository! Please run this script from the project root."
    exit 1
fi

# Step 1: Backup current state
print_step "1. Backing up current state..."

# Create backup directory with timestamp
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup important directories
if [ -d "logs" ]; then
    cp -r logs "$BACKUP_DIR/"
    print_status "Backed up logs to $BACKUP_DIR/logs"
fi

if [ -d "checkpoints" ]; then
    cp -r checkpoints "$BACKUP_DIR/"
    print_status "Backed up checkpoints to $BACKUP_DIR/checkpoints"
fi

if [ -d "data" ]; then
    # Only backup data structure, not the actual data files
    find data -type d -exec mkdir -p "$BACKUP_DIR/{}" \;
    print_status "Backed up data structure to $BACKUP_DIR/data"
fi

# Step 2: Check for local changes
print_step "2. Checking for local changes..."

if ! git diff --quiet; then
    print_warning "You have uncommitted local changes:"
    git status --porcelain
    echo
    read -p "Do you want to stash these changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git stash push -m "Auto-stash before update $(date)"
        print_status "Changes stashed successfully"
    else
        print_warning "Proceeding with local changes (may cause conflicts)"
    fi
fi

# Step 3: Fetch latest changes
print_step "3. Fetching latest changes from remote..."

git fetch origin

# Check if we're behind the remote
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "")
BASE=$(git merge-base @ @{u} 2>/dev/null || echo "")

if [ -z "$REMOTE" ]; then
    print_warning "No upstream branch set. Setting upstream to origin/main"
    git branch --set-upstream-to=origin/main
    REMOTE=$(git rev-parse @{u})
    BASE=$(git merge-base @ @{u})
fi

if [ "$LOCAL" = "$REMOTE" ]; then
    print_status "Already up to date!"
elif [ "$LOCAL" = "$BASE" ]; then
    print_status "Updates available. Pulling changes..."
    
    # Step 4: Pull changes
    print_step "4. Pulling latest changes..."
    git pull origin main
    
elif [ "$REMOTE" = "$BASE" ]; then
    print_warning "Your branch is ahead of remote. Consider pushing your changes."
else
    print_warning "Branches have diverged. Attempting to merge..."
    git pull origin main
fi

# Step 5: Check for dependency updates
print_step "5. Checking for dependency updates..."

# Check if requirements.txt or environment.yml changed
if git diff HEAD~1 HEAD --name-only | grep -E "(requirements\.txt|environment\.yml)" > /dev/null; then
    print_status "Dependencies updated. Updating conda environment..."
    
    # Activate conda if not already active
    if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "fsra_enhanced" ]; then
        print_status "Activating conda environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate fsra_enhanced
    fi
    
    # Update environment
    if [ -f "environment.yml" ]; then
        conda env update -f environment.yml
    fi
    
    # Update pip packages
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt --upgrade
    fi
    
    print_status "Dependencies updated successfully"
else
    print_status "No dependency changes detected"
fi

# Step 6: Update file permissions
print_step "6. Updating file permissions..."

# Make scripts executable
find scripts/ -name "*.py" -exec chmod +x {} \; 2>/dev/null || true
find . -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

print_status "File permissions updated"

# Step 7: Verify installation
print_step "7. Verifying installation..."

# Activate environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "fsra_enhanced" ]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate fsra_enhanced
fi

# Test imports
python3 -c "
try:
    from config import load_config, Config, merge_config_with_args
    from src.models.model_factory import create_model, create_cross_attention_model
    from src.utils.evaluation import evaluate_model, EvaluationTracker
    print('✓ All imports successful')
except Exception as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || {
    print_error "Import verification failed!"
    exit 1
}

# Test CUDA if available
python3 -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU count: {torch.cuda.device_count()}')
"

# Step 8: Show update summary
print_step "8. Update summary..."

echo
echo "=========================================="
print_status "🎉 Update completed successfully!"
echo

# Show git log of recent changes
echo "📋 Recent changes:"
git log --oneline -5

echo
echo "📁 Backup created in: $BACKUP_DIR"
echo
echo "🚀 Ready to continue training!"
echo
echo "📚 Quick commands:"
echo "- Activate environment: conda activate fsra_enhanced"
echo "- Start training: python scripts/train_new_vit.py --data_dir data"
echo "- Monitor training: tensorboard --logdir logs/tensorboard"
echo "- Check status: python test_fixed_imports.py"
echo

# Check if there are any stashed changes
if git stash list | grep -q "Auto-stash before update"; then
    echo "💾 You have stashed changes. To restore them:"
    echo "   git stash pop"
    echo
fi

print_status "Update completed! Happy training! 🚀"
