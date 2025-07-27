#!/usr/bin/env python3
"""
Test the fixed imports and basic functionality.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def main():
    """Test the fixed functionality."""
    try:
        print("Testing fixed imports...")
        
        # Test config imports
        from config import load_config, Config, merge_config_with_args
        print("✓ Config imports work")
        
        # Test config loading
        config = load_config('config/default_config.yaml')
        print(f"✓ Config loaded: {config.model.name}")
        
        # Test argument merging
        args = {'model': 'FSRA', 'batch_size': 2}
        config = merge_config_with_args(config, args)
        print(f"✓ Config merged: model={config.model.name}")
        
        # Test model factory imports
        from src.models.model_factory import create_model, create_cross_attention_model
        print("✓ Model factory imports work")
        
        # Test model creation (CPU only to avoid GPU issues)
        model, cross_attention = create_model(config, device='cpu', use_pretrained=False)
        print(f"✓ Models created: {type(model).__name__}, {type(cross_attention).__name__}")
        
        print("\n🎉 All tests passed! The training script should now work.")
        print("\nYou can now run:")
        print("python scripts/train.py --config config/default_config.yaml --model FSRA --data_dir data --batch_size 2 --num_epochs 1")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
