#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_config_imports():
    """Test configuration imports."""
    try:
        from config import load_config, Config, merge_config_with_args
        print("✓ Config imports successful")
        
        # Test loading default config
        config = load_config('config/default_config.yaml')
        print(f"✓ Config loaded: {config.model.name}")
        
        # Test merge function
        args = {'batch_size': 16, 'learning_rate': 0.001}
        config = merge_config_with_args(config, args)
        print(f"✓ Config merge successful: batch_size={config.data.batch_size}")
        
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_model_imports():
    """Test model imports."""
    try:
        from src.models import make_fsra_model, make_new_vit_model, CrossAttentionModel
        from src.models.model_factory import create_model, create_cross_attention_model
        print("✓ Model imports successful")
        return True
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_dataset_imports():
    """Test dataset imports."""
    try:
        from src.datasets import make_dataloader
        print("✓ Dataset imports successful")
        return True
    except Exception as e:
        print(f"❌ Dataset import failed: {e}")
        return False

def test_utils_imports():
    """Test utility imports."""
    try:
        from src.utils.logger import setup_logger
        from src.utils.checkpoint import save_checkpoint, load_checkpoint
        from src.utils.metrics import AverageMeter, accuracy
        from src.utils.evaluation import evaluate_model, EvaluationTracker
        print("✓ Utils imports successful")
        return True
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False

def main():
    """Run all import tests."""
    print("Testing imports for FSRA Enhanced project...")
    print("=" * 50)
    
    tests = [
        test_config_imports,
        test_model_imports,
        test_dataset_imports,
        test_utils_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Import tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All imports successful! Ready to train.")
        return 0
    else:
        print("❌ Some imports failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
