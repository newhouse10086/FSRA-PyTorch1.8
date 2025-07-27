"""Model factory for creating different types of models with flexible training options."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os

from .fsra import make_fsra_model, FSRAModel, TwoViewNet
from .new_vit import make_new_vit_model, NewViTModel, NewTwoViewNet
from .cross_attention import CrossAttentionModel


def create_model(config: Dict[str, Any], device: str = 'cuda', use_pretrained: bool = True) -> tuple:
    """
    Create model based on configuration.

    Args:
        config: Configuration dictionary
        device: Device to place model on
        use_pretrained: Whether to use pretrained weights

    Returns:
        Tuple of (model, cross_attention_model)
    """
    model_name = config.model.name.upper()

    if model_name == "FSRA":
        model = create_fsra_model(config, use_pretrained)
    elif model_name == "NEWVIT":
        model = create_new_vit_model(config, use_pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Move model to device
    if device != 'cpu' and torch.cuda.is_available():
        model = model.cuda()

    # Create cross attention model
    cross_attention = create_cross_attention_model(config)
    if device != 'cpu' and torch.cuda.is_available():
        cross_attention = cross_attention.cuda()

    return model, cross_attention


def create_fsra_model(config: Dict[str, Any], use_pretrained: bool = True) -> nn.Module:
    """
    Create FSRA model.
    
    Args:
        config: Configuration dictionary
        use_pretrained: Whether to use pretrained weights
        
    Returns:
        FSRA model instance
    """
    model = make_fsra_model(
        num_classes=config.model.num_classes,
        block_size=config.model.block_size,
        return_f=config.model.return_features,
        views=config.data.views,
        share_weights=config.model.share_weights
    )
    
    # Load pretrained weights if requested
    if use_pretrained and hasattr(config.model, 'use_pretrained') and config.model.use_pretrained:
        pretrained_path = getattr(config.model, 'pretrained_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            if hasattr(model, 'model_1'):
                # Two-view model
                model.model_1.load_pretrained(pretrained_path)
                if not config.model.share_weights and hasattr(model, 'model_2'):
                    model.model_2.load_pretrained(pretrained_path)
            else:
                # Single-view model
                model.load_pretrained(pretrained_path)
        else:
            print("Pretrained path not found or not specified. Training from scratch.")
    else:
        print("Training FSRA from scratch without pretrained weights.")
    
    return model


def create_new_vit_model(config: Dict[str, Any], use_pretrained: bool = True) -> nn.Module:
    """
    Create New ViT model.
    
    Args:
        config: Configuration dictionary
        use_pretrained: Whether to use pretrained weights
        
    Returns:
        New ViT model instance
    """
    # Determine pretrained settings
    use_pretrained_resnet = True  # Default to True for ResNet18
    use_pretrained_vit = False    # Default to False for ViT (train from scratch)
    
    if use_pretrained:
        use_pretrained_resnet = getattr(config.model, 'use_pretrained_resnet', True)
        use_pretrained_vit = getattr(config.model, 'use_pretrained_vit', False)
    else:
        # From scratch training
        use_pretrained_resnet = False
        use_pretrained_vit = False
    
    model = make_new_vit_model(
        num_classes=config.model.num_classes,
        use_pretrained_resnet=use_pretrained_resnet,
        use_pretrained_vit=use_pretrained_vit,
        num_final_clusters=getattr(config.model, 'num_final_clusters', 3),
        return_f=config.model.return_features,
        views=config.data.views,
        share_weights=config.model.share_weights
    )
    
    # Load pretrained ViT weights if specified
    if use_pretrained_vit and hasattr(config.model, 'pretrained_path'):
        pretrained_path = config.model.pretrained_path
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained ViT weights from: {pretrained_path}")
            if hasattr(model, 'model_1'):
                # Two-view model
                model.model_1.load_pretrained_vit(pretrained_path)
                if not config.model.share_weights and hasattr(model, 'model_2'):
                    model.model_2.load_pretrained_vit(pretrained_path)
            else:
                # Single-view model
                model.load_pretrained_vit(pretrained_path)
        else:
            print("Pretrained ViT path not found. Using random initialization for ViT.")
    
    print(f"New ViT model created:")
    print(f"  - ResNet18 pretrained: {use_pretrained_resnet}")
    print(f"  - ViT pretrained: {use_pretrained_vit}")
    print(f"  - Final clusters: {getattr(config.model, 'num_final_clusters', 3)}")
    
    return model


def create_cross_attention_model(config: Dict[str, Any]) -> CrossAttentionModel:
    """
    Create cross attention model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Cross attention model instance
    """
    return CrossAttentionModel(
        d_model=512,
        block_size=config.model.block_size + 1
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   strict: bool = True) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Epoch number from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded successfully. Epoch: {epoch}")
    return epoch


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, filepath: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: Model to count parameters for
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_info(model: nn.Module, model_name: str = "Model") -> None:
    """
    Print model information.
    
    Args:
        model: Model to print info for
        model_name: Name of the model
    """
    param_counts = count_parameters(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"  Model size: {param_counts['total'] * 4 / 1024 / 1024:.2f} MB")
