#!/usr/bin/env python3
"""
Basic test for training script functionality.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_basic_training_setup():
    """Test basic training setup without actual training."""
    try:
        # Test imports
        from config import load_config, Config, merge_config_with_args
        from src.models.model_factory import create_model, create_cross_attention_model
        
        print("✓ All imports successful")
        
        # Test config loading
        config = load_config('config/default_config.yaml')
        print(f"✓ Config loaded: {config.model.name}")
        
        # Test argument merging
        args = {
            'model': 'FSRA',
            'data_dir': 'data',
            'batch_size': 2,
            'num_epochs': 1
        }
        config = merge_config_with_args(config, args)
        print(f"✓ Config merged: model={config.model.name}, batch_size={config.data.batch_size}")
        
        # Test model creation
        model, cross_attention = create_model(config, device='cpu', use_pretrained=False)
        print(f"✓ Model created: {type(model).__name__}")
        
        # Test cross attention model
        cross_attention_model = create_cross_attention_model(config)
        print(f"✓ Cross attention model created: {type(cross_attention_model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run basic training test."""
    print("Testing basic training setup...")
    print("=" * 50)
    
    if test_basic_training_setup():
        print("=" * 50)
        print("🎉 Basic training setup test passed!")
        print("You can now run the full training command:")
        print("python scripts/train.py --config config/default_config.yaml --model FSRA --data_dir data")
        return 0
    else:
        print("=" * 50)
        print("❌ Basic training setup test failed!")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
