#!/usr/bin/env python3
"""
Simplified training test script to verify fixes.
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FSRA Training Test')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='FSRA', choices=['FSRA', 'NewViT'],
                       help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    return parser.parse_args()

def test_fsra_training(config):
    """Test FSRA model training setup."""
    print("🧪 Testing FSRA model setup...")
    
    try:
        from src.models.model_factory import create_model
        
        # Create FSRA model
        model, cross_attention = create_model(config, device='cpu', use_pretrained=False)
        
        print(f"✓ FSRA models created successfully")
        print(f"  Main model: {type(model).__name__}")
        print(f"  Cross attention: {type(cross_attention).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ FSRA model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_vit_training(config):
    """Test New ViT model training setup."""
    print("🧪 Testing New ViT model setup...")
    
    try:
        from src.models.new_vit import make_new_vit_model
        from src.models.model_factory import create_cross_attention_model
        
        # Create New ViT model
        model = make_new_vit_model(
            config,
            use_pretrained_resnet=True,
            use_pretrained_vit=False,
            num_final_clusters=3
        )
        
        cross_attention = create_cross_attention_model(config)
        
        print(f"✓ New ViT models created successfully")
        print(f"  Main model: {type(model).__name__}")
        print(f"  Cross attention: {type(cross_attention).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ New ViT model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🔧 Testing Fixed Training Scripts")
    print("=" * 50)
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Test configuration loading and merging
        print("🧪 Testing configuration handling...")
        
        from config import load_config, merge_config_with_args
        
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
        else:
            print(f"❌ Config file not found: {args.config}")
            return 1
        
        print(f"✓ Config loaded: {config.model.name}")
        
        # Override model name BEFORE merging (like in fixed train.py)
        if args.model:
            config.model.name = args.model
            print(f"✓ Model set to: {config.model.name}")
        
        # Merge with command line arguments (excluding model to avoid conflicts)
        args_dict = vars(args).copy()
        for key in ['model', 'config']:
            if key in args_dict:
                del args_dict[key]
        
        config = merge_config_with_args(config, args_dict)
        print(f"✓ Config merged successfully")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")
        print(f"  Num epochs: {config.training.num_epochs}")
        
        # Test model creation based on model type
        if config.model.name.upper() == "FSRA":
            success = test_fsra_training(config)
        elif config.model.name.upper() == "NEWVIT":
            success = test_new_vit_training(config)
        else:
            print(f"❌ Unknown model type: {config.model.name}")
            return 1
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 All tests passed! Training scripts should work now.")
            print("\nYou can run:")
            if config.model.name.upper() == "FSRA":
                print(f"python scripts/train.py --config {args.config} --model FSRA --data_dir {args.data_dir} --batch_size {args.batch_size} --num_epochs {args.num_epochs}")
            else:
                print(f"python scripts/train_new_vit.py --config {args.config} --data_dir {args.data_dir} --batch_size {args.batch_size} --num_epochs {args.num_epochs}")
            return 0
        else:
            print("\n❌ Model creation test failed.")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
