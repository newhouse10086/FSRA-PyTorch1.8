#!/usr/bin/env python3
"""
Test script to verify configuration fixes.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_config_loading():
    """Test configuration loading and merging."""
    print("🧪 Testing configuration loading and merging...")
    
    try:
        # Test imports
        from config import load_config, Config, merge_config_with_args
        print("✓ Config imports successful")
        
        # Test config loading
        config = load_config('config/default_config.yaml')
        print(f"✓ Config loaded: {type(config)}")
        print(f"✓ Model config type: {type(config.model)}")
        print(f"✓ Model name: {config.model.name}")
        
        # Test config merging
        test_args = {
            'batch_size': 8,
            'learning_rate': 0.001,
            'num_epochs': 100
        }
        
        config = merge_config_with_args(config, test_args)
        print(f"✓ Config merge successful")
        print(f"✓ Batch size: {config.data.batch_size}")
        print(f"✓ Learning rate: {config.training.learning_rate}")
        print(f"✓ Num epochs: {config.training.num_epochs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation with fixed config."""
    print("\n🧪 Testing model creation...")
    
    try:
        from config import load_config
        from src.models.model_factory import create_model
        
        # Load config
        config = load_config('config/default_config.yaml')
        config.model.name = "FSRA"  # Set model type
        
        # Create models
        model, cross_attention = create_model(config, device='cpu', use_pretrained=False)
        print(f"✓ Models created successfully")
        print(f"  Main model: {type(model).__name__}")
        print(f"  Cross attention: {type(cross_attention).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_args():
    """Test training script argument parsing."""
    print("\n🧪 Testing training script argument simulation...")
    
    try:
        from config import load_config, merge_config_with_args
        
        # Simulate command line arguments
        simulated_args = {
            'model': 'FSRA',
            'data_dir': 'data',
            'batch_size': 4,
            'num_epochs': 2,
            'learning_rate': 0.005
        }
        
        # Load config
        config = load_config('config/default_config.yaml')
        
        # Apply model name first (like in fixed train.py)
        if 'model' in simulated_args:
            config.model.name = simulated_args['model']
        
        # Remove model from args to avoid conflicts
        args_dict = simulated_args.copy()
        if 'model' in args_dict:
            del args_dict['model']
        
        # Merge remaining args
        config = merge_config_with_args(config, args_dict)
        
        print(f"✓ Argument simulation successful")
        print(f"  Model name: {config.model.name}")
        print(f"  Data dir: {config.data.data_dir}")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Num epochs: {config.training.num_epochs}")
        print(f"  Learning rate: {config.training.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training script argument test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_vit_config():
    """Test New ViT specific configuration."""
    print("\n🧪 Testing New ViT configuration...")
    
    try:
        from config import load_config, merge_config_with_args
        
        # Load config
        config = load_config('config/default_config.yaml')
        
        # Set New ViT specific settings
        config.model.name = "NewViT"
        config.model.use_pretrained_resnet = True
        config.model.use_pretrained_vit = False
        config.model.num_final_clusters = 3
        
        print(f"✓ New ViT config successful")
        print(f"  Model name: {config.model.name}")
        print(f"  Use pretrained ResNet: {config.model.use_pretrained_resnet}")
        print(f"  Use pretrained ViT: {config.model.use_pretrained_vit}")
        print(f"  Num final clusters: {config.model.num_final_clusters}")
        
        return True
        
    except Exception as e:
        print(f"❌ New ViT config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all configuration tests."""
    print("🔧 Testing Configuration Fixes")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_model_creation,
        test_training_script_args,
        test_new_vit_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Configuration tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All configuration fixes successful!")
        print("\nYou can now run:")
        print("python scripts/train.py --config config/default_config.yaml --model FSRA --data_dir data --batch_size 2 --num_epochs 1")
        print("python scripts/train_new_vit.py --config config/default_config.yaml --data_dir data --batch_size 2 --num_epochs 1")
        return 0
    else:
        print("❌ Some configuration tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
