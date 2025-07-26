"""Data loading utilities for FSRA project."""

import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader

from .university_dataset import (
    UniversityDataset, 
    UniversitySampler, 
    UniversityTestDataset,
    train_collate_fn,
    test_collate_fn
)
from .transforms import get_train_transforms, get_test_transforms


def make_dataloader(config, mode: str = 'train') -> Tuple[DataLoader, list, Dict[str, int]]:
    """
    Create data loader for training or testing.
    
    Args:
        config: Configuration object
        mode: 'train' or 'test'
        
    Returns:
        Tuple of (dataloader, class_names, dataset_sizes)
    """
    if mode == 'train':
        return _make_train_dataloader(config)
    elif mode == 'test':
        return _make_test_dataloader(config)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _make_train_dataloader(config) -> Tuple[DataLoader, list, Dict[str, int]]:
    """Create training data loader."""
    # Get transforms
    satellite_transform, drone_transform = get_train_transforms(config)
    
    # Create dataset
    dataset = UniversityDataset(
        root=config.data.data_dir,
        satellite_transform=satellite_transform,
        drone_transform=drone_transform,
        views=['satellite', 'drone'],
        mode='train'
    )
    
    # Create sampler
    sampler = UniversitySampler(
        dataset=dataset,
        batch_size=config.data.batch_size,
        sample_num=config.data.sample_num
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True if config.system.use_gpu else False,
        drop_last=True
    )
    
    # Dataset info
    class_names = dataset.class_names
    dataset_sizes = {
        'satellite': len(dataset) * config.data.sample_num,
        'drone': len(dataset) * config.data.sample_num,
        'total': len(dataset) * config.data.sample_num
    }
    
    return dataloader, class_names, dataset_sizes


def _make_test_dataloader(config) -> Tuple[DataLoader, list, Dict[str, int]]:
    """Create test data loader."""
    # Get test transforms
    test_transform = get_test_transforms(config)
    
    # Create test dataset
    dataset = UniversityTestDataset(
        root=config.data.test_dir,
        transform=test_transform,
        query_dir='query_satellite',
        gallery_dir='gallery_drone'
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True if config.system.use_gpu else False
    )
    
    # Dataset info
    query_size, gallery_size = dataset.get_query_gallery_split()
    class_names = []  # Will be populated from dataset if needed
    dataset_sizes = {
        'query': query_size,
        'gallery': gallery_size,
        'total': len(dataset)
    }
    
    return dataloader, class_names, dataset_sizes


def make_multi_view_dataloader(config) -> Tuple[DataLoader, list, Dict[str, int]]:
    """
    Create multi-view data loader for training with multiple views.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (dataloader, class_names, dataset_sizes)
    """
    if config.data.views == 2:
        return _make_train_dataloader(config)
    else:
        raise NotImplementedError(f"Multi-view with {config.data.views} views not implemented")


class DataLoaderManager:
    """Manager class for handling multiple data loaders."""
    
    def __init__(self, config):
        self.config = config
        self.train_loader = None
        self.test_loader = None
        self.class_names = None
        self.dataset_sizes = None
    
    def get_train_loader(self) -> Tuple[DataLoader, list, Dict[str, int]]:
        """Get or create training data loader."""
        if self.train_loader is None:
            self.train_loader, self.class_names, self.dataset_sizes = make_dataloader(
                self.config, mode='train')
        return self.train_loader, self.class_names, self.dataset_sizes
    
    def get_test_loader(self) -> Tuple[DataLoader, list, Dict[str, int]]:
        """Get or create test data loader."""
        if self.test_loader is None:
            self.test_loader, _, test_sizes = make_dataloader(
                self.config, mode='test')
        else:
            _, _, test_sizes = make_dataloader(self.config, mode='test')
        return self.test_loader, self.class_names or [], test_sizes
    
    def reset(self):
        """Reset all loaders."""
        self.train_loader = None
        self.test_loader = None
        self.class_names = None
        self.dataset_sizes = None
