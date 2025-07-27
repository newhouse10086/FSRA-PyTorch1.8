"""Configuration management for FSRA project."""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "FSRA"
    backbone: str = "vit_small_patch16_224"
    num_classes: int = 701
    block_size: int = 3
    share_weights: bool = True
    return_features: bool = True
    dropout_rate: float = 0.1

    # Pretrained model settings
    use_pretrained: bool = True
    pretrained_path: str = "pretrained/vit_small_p16_224-15ec54c9.pth"

    # New ViT specific settings
    use_pretrained_resnet: bool = True
    use_pretrained_vit: bool = False
    num_final_clusters: int = 3


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "data/train"
    test_dir: str = "data/test"
    batch_size: int = 14
    num_workers: int = 1
    image_height: int = 256
    image_width: int = 256
    views: int = 2
    sample_num: int = 7
    pad: int = 0
    color_jitter: bool = False
    random_erasing_prob: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 120
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9
    warm_epochs: int = 0
    lr_scheduler_steps: list = field(default_factory=lambda: [70, 110])
    lr_scheduler_gamma: float = 0.1
    
    # Loss weights
    triplet_loss_weight: float = 0.3
    kl_loss_weight: float = 0.0
    use_kl_loss: bool = False
    
    # Mixed precision training
    use_fp16: bool = False
    use_autocast: bool = False
    
    # Data augmentation
    use_data_augmentation: bool = False
    moving_avg: float = 1.0


@dataclass
class SystemConfig:
    """System configuration."""
    gpu_ids: str = "0"
    use_gpu: bool = True
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    pretrained_dir: str = "pretrained"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Mode: 0 for training, 1 for testing
    mode: int = 0
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure directories exist
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)
        os.makedirs(self.system.pretrained_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__,
            'mode': self.mode
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        if 'system' in config_dict:
            for key, value in config_dict['system'].items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        if 'mode' in config_dict:
            config.mode = config_dict['mode']
        
        return config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


def merge_config_with_args(config: Config, args: Optional[Dict[str, Any]] = None) -> Config:
    """Merge configuration with command line arguments."""
    if args is None:
        return config

    # Ensure config is a proper Config object
    if not isinstance(config, Config):
        raise TypeError(f"Expected Config object, got {type(config)}")

    # Update config with command line arguments
    for key, value in args.items():
        # Skip None values and special arguments
        if value is None or key in ['config', 'resume', 'pretrained', 'from_scratch']:
            continue

        try:
            # Check each config section in order
            if hasattr(config, 'model') and hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config, 'data') and hasattr(config.data, key):
                setattr(config.data, key, value)
            elif hasattr(config, 'training') and hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config, 'system') and hasattr(config.system, key):
                setattr(config.system, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                # Skip unknown arguments with a warning
                print(f"Warning: Unknown configuration argument '{key}' with value '{value}'")
        except Exception as e:
            print(f"Warning: Failed to set config.{key} = {value}: {e}")

    return config
