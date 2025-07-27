#!/usr/bin/env python3
"""
Test PyTorch 1.8 compatibility fixes.
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_weight_initialization():
    """Test weight initialization functions."""
    print("🧪 Testing weight initialization functions...")
    
    try:
        from src.models.fsra.components import weights_init_kaiming, weights_init_classifier
        
        # Create test modules
        linear_with_bias = nn.Linear(10, 5)
        linear_without_bias = nn.Linear(10, 5, bias=False)
        conv_with_bias = nn.Conv2d(3, 16, 3)
        conv_without_bias = nn.Conv2d(3, 16, 3, bias=False)
        batchnorm = nn.BatchNorm2d(16)
        
        # Test kaiming initialization
        print("  Testing weights_init_kaiming...")
        weights_init_kaiming(linear_with_bias)
        weights_init_kaiming(linear_without_bias)
        weights_init_kaiming(conv_with_bias)
        weights_init_kaiming(conv_without_bias)
        weights_init_kaiming(batchnorm)
        print("  ✓ weights_init_kaiming passed")
        
        # Test classifier initialization
        print("  Testing weights_init_classifier...")
        weights_init_classifier(linear_with_bias)
        weights_init_classifier(linear_without_bias)
        print("  ✓ weights_init_classifier passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Weight initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation with fixed components."""
    print("\n🧪 Testing model creation...")
    
    try:
        from config import load_config
        from src.models.model_factory import create_model
        
        # Load config
        config = load_config('config/default_config.yaml')
        config.model.name = "FSRA"
        
        # Create model (CPU only to avoid GPU issues)
        model, cross_attention = create_model(config, device='cpu', use_pretrained=False)
        
        print(f"  ✓ FSRA model created: {type(model).__name__}")
        print(f"  ✓ Cross attention created: {type(cross_attention).__name__}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 256, 256)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input, None)  # FSRA expects two inputs
            print(f"  ✓ Forward pass successful: {type(output)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_vit_model():
    """Test New ViT model creation."""
    print("\n🧪 Testing New ViT model...")
    
    try:
        from config import load_config
        from src.models.new_vit import make_new_vit_model
        
        # Load config
        config = load_config('config/default_config.yaml')
        config.model.name = "NewViT"
        config.model.num_classes = 701
        
        # Create New ViT model
        model = make_new_vit_model(
            config,
            use_pretrained_resnet=False,  # Avoid downloading pretrained weights
            use_pretrained_vit=False,
            num_final_clusters=3
        )
        
        print(f"  ✓ New ViT model created: {type(model).__name__}")
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(1, 3, 256, 256)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input, None)
            print(f"  ✓ Forward pass successful: {type(output)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ New ViT model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pytorch_version():
    """Test PyTorch version and basic functionality."""
    print("🧪 Testing PyTorch version and basic functionality...")
    
    try:
        print(f"  PyTorch version: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print("  ! CUDA not available (CPU mode)")
        
        # Test basic tensor operations
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        z = torch.matmul(x, y.transpose(-1, -2))
        print(f"  ✓ Basic tensor operations work: {z.shape}")
        
        # Test autograd
        x.requires_grad_(True)
        loss = (x ** 2).sum()
        loss.backward()
        print(f"  ✓ Autograd works")
        
        # Test bias checking (the main fix)
        linear = nn.Linear(10, 5)
        if linear.bias is not None:
            print(f"  ✓ Bias check works: bias shape {linear.bias.shape}")
        
        linear_no_bias = nn.Linear(10, 5, bias=False)
        if linear_no_bias.bias is None:
            print(f"  ✓ No bias check works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_compatibility():
    """Test training script compatibility."""
    print("\n🧪 Testing training script compatibility...")
    
    try:
        # Test configuration loading
        from config import load_config, merge_config_with_args
        config = load_config('config/default_config.yaml')
        
        # Simulate training script arguments
        args = {
            'model': 'FSRA',
            'batch_size': 2,
            'num_epochs': 1,
            'data_dir': 'data'
        }
        
        # Test the fixed configuration handling
        if 'model' in args:
            config.model.name = args['model']
        
        args_dict = args.copy()
        if 'model' in args_dict:
            del args_dict['model']
        
        config = merge_config_with_args(config, args_dict)
        
        print(f"  ✓ Configuration handling works")
        print(f"    Model: {config.model.name}")
        print(f"    Batch size: {config.data.batch_size}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training script compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all PyTorch compatibility tests."""
    print("🔧 Testing PyTorch 1.8 Compatibility Fixes")
    print("=" * 60)
    
    tests = [
        test_pytorch_version,
        test_weight_initialization,
        test_training_script_compatibility,
        test_model_creation,
        test_new_vit_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"PyTorch compatibility tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All PyTorch compatibility fixes successful!")
        print("\nThe training scripts should now work with PyTorch 1.8.2")
        print("\nYou can run:")
        print("python scripts/train.py --config config/default_config.yaml --model FSRA --data_dir data --batch_size 2 --num_epochs 1")
        return 0
    else:
        print("❌ Some PyTorch compatibility tests failed.")
        print("Please check the errors above and ensure PyTorch 1.8 is properly installed.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
