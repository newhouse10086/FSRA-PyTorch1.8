#!/usr/bin/env python3
"""
Basic test script for FSRA refactored project.
"""

import torch
import sys
import os

def test_basic_functionality():
    """Test basic functionality without complex imports."""
    print("Testing basic PyTorch functionality...")
    
    # Test PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available (CPU mode)")
    
    # Test basic tensor operations
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    z = torch.matmul(x, y.transpose(-1, -2))
    print(f"Basic tensor operations work: {z.shape}")
    
    # Test autograd
    x.requires_grad_(True)
    loss = (x ** 2).sum()
    loss.backward()
    print("Autograd works")
    
    print("✓ Basic PyTorch functionality test passed!")
    return True

def test_config():
    """Test configuration loading."""
    try:
        from config import Config
        config = Config()
        print(f"✓ Configuration loaded: {config.model.name}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_models():
    """Test model creation."""
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        from src.models.backbones.vit_pytorch import vit_small_patch16_224_fsra
        
        # Test ViT model creation
        model = vit_small_patch16_224_fsra(num_classes=701)
        print(f"✓ ViT model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Model forward pass successful: {len(output)} outputs")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("FSRA Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_config,
        test_models,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
    else:
        print("⚠️  Some tests failed.")
    
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
