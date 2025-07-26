#!/usr/bin/env python3
"""
Test script to verify the refactored FSRA project setup.
This script tests basic functionality without requiring the full dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from config import load_config, Config
from src.models import FSRAModel, CrossAttentionModel, make_fsra_model


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    try:
        # Test default config
        config = Config()
        print(f"✓ Default config created successfully")
        print(f"  - Model name: {config.model.name}")
        print(f"  - Batch size: {config.data.batch_size}")
        print(f"  - Learning rate: {config.training.learning_rate}")
        
        # Test config to dict and from dict
        config_dict = config.to_dict()
        config_restored = Config.from_dict(config_dict)
        print(f"✓ Config serialization/deserialization works")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        # Test FSRA model
        fsra_model = FSRAModel(num_classes=701, block_size=3, return_f=True)
        print(f"✓ FSRA model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in fsra_model.parameters()):,}")
        
        # Test CrossAttention model
        cross_attn = CrossAttentionModel(d_model=512, block_size=4)
        print(f"✓ CrossAttention model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in cross_attn.parameters()):,}")
        
        # Test make_fsra_model function
        model = make_fsra_model(num_classes=701, views=2, share_weights=True)
        print(f"✓ Two-view model created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_model_forward():
    """Test model forward pass."""
    print("\nTesting model forward pass...")
    
    try:
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 256, 256)
        
        # Test FSRA model forward
        fsra_model = FSRAModel(num_classes=701, block_size=3, return_f=True)
        fsra_model.eval()
        
        with torch.no_grad():
            predictions, features = fsra_model(dummy_input)
        
        print(f"✓ FSRA forward pass successful")
        print(f"  - Predictions: {len(predictions)} outputs")
        print(f"  - Features: {len(features)} feature tensors")
        
        # Test CrossAttention forward
        cross_attn = CrossAttentionModel(d_model=512, block_size=4)
        cross_attn.eval()
        
        # Create dummy features
        queries = torch.randn(batch_size, 1, 512 * 4)
        support_set = torch.randn(batch_size, 7, 512 * 4)
        
        with torch.no_grad():
            q_proto, c_proto = cross_attn(queries, support_set, mode=0)
        
        print(f"✓ CrossAttention forward pass successful")
        print(f"  - Query prototype shape: {q_proto.shape}")
        print(f"  - Class prototype shape: {c_proto.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return False


def test_pytorch_compatibility():
    """Test PyTorch 1.8 compatibility."""
    print("\nTesting PyTorch compatibility...")
    
    try:
        # Check PyTorch version
        print(f"PyTorch version: {torch.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
        else:
            print("! CUDA not available (CPU mode)")
        
        # Test basic operations
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        z = torch.matmul(x, y.transpose(-1, -2))
        print(f"✓ Basic tensor operations work")
        
        # Test autograd
        x.requires_grad_(True)
        loss = (x ** 2).sum()
        loss.backward()
        print(f"✓ Autograd works")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch compatibility test failed: {e}")
        return False


def test_transforms():
    """Test data transforms."""
    print("\nTesting data transforms...")
    
    try:
        from src.datasets.transforms import get_train_transforms, get_test_transforms
        
        # Create dummy config
        config = Config()
        
        # Test train transforms
        sat_transform, drone_transform = get_train_transforms(config)
        print(f"✓ Train transforms created successfully")
        
        # Test test transforms
        test_transform = get_test_transforms(config)
        print(f"✓ Test transforms created successfully")
        
        # Test transform application (if PIL is available)
        try:
            from PIL import Image
            import numpy as np
            
            # Create dummy image
            dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            
            # Apply transforms
            sat_tensor = sat_transform(dummy_img)
            test_tensor = test_transform(dummy_img)
            
            print(f"✓ Transform application successful")
            print(f"  - Satellite tensor shape: {sat_tensor.shape}")
            print(f"  - Test tensor shape: {test_tensor.shape}")
            
        except ImportError:
            print("! PIL not available, skipping transform application test")
        
        return True
    except Exception as e:
        print(f"✗ Transform test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FSRA Project Setup Test")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_pytorch_compatibility,
        test_model_creation,
        test_model_forward,
        test_transforms,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored FSRA project is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
