"""Checkpoint utilities for FSRA project."""

import os
import torch
from typing import Dict, Any, Optional


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str, 
                   additional_info: Optional[Dict[str, Any]] = None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        additional_info: Additional information to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Add additional information
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> int:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Epoch number from checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    if device is None:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, loss {loss:.4f})")
    
    return epoch


def save_model_only(model: torch.nn.Module, filepath: str):
    """
    Save only model state dict.
    
    Args:
        model: Model to save
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")


def load_model_only(filepath: str, model: torch.nn.Module, 
                   device: Optional[torch.device] = None):
    """
    Load only model state dict.
    
    Args:
        filepath: Path to model file
        model: Model to load state into
        device: Device to load model on
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    if device is None:
        state_dict = torch.load(filepath)
    else:
        state_dict = torch.load(filepath, map_location=device)
    
    model.load_state_dict(state_dict)
    print(f"Model loaded: {filepath}")


def load_pretrained_weights(model: torch.nn.Module, filepath: str, 
                          strict: bool = False, device: Optional[torch.device] = None):
    """
    Load pretrained weights with flexible matching.
    
    Args:
        model: Model to load weights into
        filepath: Path to pretrained weights
        strict: Whether to strictly match state dict keys
        device: Device to load weights on
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pretrained weights not found: {filepath}")
    
    # Load weights
    if device is None:
        pretrained_dict = torch.load(filepath)
    else:
        pretrained_dict = torch.load(filepath, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_state_dict']
    elif 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    
    # Get model state dict
    model_dict = model.state_dict()
    
    if strict:
        # Strict loading
        model.load_state_dict(pretrained_dict)
    else:
        # Flexible loading - only load matching keys
        matched_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched_dict[k] = v
            else:
                print(f"Skipping key {k}: shape mismatch or not found")
        
        # Update model dict
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)
        
        print(f"Loaded {len(matched_dict)}/{len(pretrained_dict)} pretrained weights")
    
    print(f"Pretrained weights loaded: {filepath}")


class CheckpointManager:
    """Manager for handling model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, loss: float, metric: float = None, 
             is_best: bool = False):
        """Save checkpoint and manage checkpoint history."""
        # Create checkpoint filename
        filename = f"checkpoint_epoch_{epoch}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        additional_info = {}
        if metric is not None:
            additional_info['metric'] = metric
        
        save_checkpoint(model, optimizer, epoch, loss, filepath, additional_info)
        
        # Add to checkpoint list
        self.checkpoints.append({
            'epoch': epoch,
            'filepath': filepath,
            'loss': loss,
            'metric': metric
        })
        
        # Save best model if specified
        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, loss, best_filepath, additional_info)
        
        # Remove old checkpoints if exceeding limit
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint['filepath']):
                os.remove(old_checkpoint['filepath'])
                print(f"Removed old checkpoint: {old_checkpoint['filepath']}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]['filepath']
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            return best_path
        return None
