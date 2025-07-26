"""University-1652 dataset implementation for FSRA."""

import os
import random
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


class UniversityDataset(Dataset):
    """University-1652 dataset for geo-localization."""
    
    def __init__(self, root: str, satellite_transform: Optional[Callable] = None,
                 drone_transform: Optional[Callable] = None, 
                 views: List[str] = ['satellite', 'drone'],
                 mode: str = 'train'):
        """
        Initialize University dataset.
        
        Args:
            root: Root directory of the dataset
            satellite_transform: Transform for satellite images
            drone_transform: Transform for drone images
            views: List of view types to load
            mode: Dataset mode ('train' or 'test')
        """
        super(UniversityDataset, self).__init__()
        
        self.root = root
        self.views = views
        self.mode = mode
        self.satellite_transform = satellite_transform
        self.drone_transform = drone_transform
        
        # Build dataset structure
        self.data_dict = self._build_dataset_dict()
        
        # Get class names and create mapping
        self.class_names = sorted(os.listdir(os.path.join(root, views[0])))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        self.num_classes = len(self.class_names)
        
    def _build_dataset_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """Build dictionary mapping view -> class -> image paths."""
        data_dict = {}
        
        for view in self.views:
            view_path = os.path.join(self.root, view)
            if not os.path.exists(view_path):
                raise FileNotFoundError(f"View directory not found: {view_path}")
            
            view_dict = {}
            for class_name in os.listdir(view_path):
                class_path = os.path.join(view_path, class_name)
                if os.path.isdir(class_path):
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    image_paths = [os.path.join(class_path, img) for img in image_files]
                    view_dict[class_name] = image_paths
            
            data_dict[view] = view_dict
        
        return data_dict
    
    def sample_from_class(self, view: str, class_name: str) -> Image.Image:
        """Sample a random image from a specific view and class."""
        if view not in self.data_dict:
            raise ValueError(f"View {view} not found in dataset")
        if class_name not in self.data_dict[view]:
            raise ValueError(f"Class {class_name} not found in view {view}")
        
        image_paths = self.data_dict[view][class_name]
        if not image_paths:
            raise ValueError(f"No images found for view {view}, class {class_name}")
        
        image_path = random.choice(image_paths)
        image = Image.open(image_path).convert('RGB')
        return image
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            index: Class index
            
        Returns:
            Tuple of (satellite_image, drone_image, class_index)
        """
        class_name = self.idx_to_class[index]
        
        # Sample satellite image
        satellite_img = self.sample_from_class('satellite', class_name)
        if self.satellite_transform:
            satellite_img = self.satellite_transform(satellite_img)
        
        # Sample drone image
        drone_img = self.sample_from_class('drone', class_name)
        if self.drone_transform:
            drone_img = self.drone_transform(drone_img)
        
        return satellite_img, drone_img, index
    
    def __len__(self) -> int:
        """Return number of classes."""
        return self.num_classes
    
    def get_class_name(self, index: int) -> str:
        """Get class name by index."""
        return self.idx_to_class[index]
    
    def get_class_index(self, class_name: str) -> int:
        """Get class index by name."""
        return self.class_to_idx[class_name]


class UniversitySampler(Sampler):
    """Custom sampler for University dataset with repeated sampling."""
    
    def __init__(self, dataset: UniversityDataset, batch_size: int = 8, 
                 sample_num: int = 4):
        """
        Initialize sampler.
        
        Args:
            dataset: University dataset
            batch_size: Batch size
            sample_num: Number of times to repeat each class
        """
        super(UniversitySampler, self).__init__(dataset)
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.data_len = len(dataset)
    
    def __iter__(self):
        """Generate indices for sampling."""
        # Create list of class indices
        indices = list(range(self.data_len))
        
        # Shuffle the indices
        random.shuffle(indices)
        
        # Repeat each index sample_num times
        repeated_indices = []
        for idx in indices:
            repeated_indices.extend([idx] * self.sample_num)
        
        return iter(repeated_indices)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.data_len * self.sample_num


class UniversityTestDataset(Dataset):
    """University-1652 test dataset."""
    
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 query_dir: str = 'query_satellite', gallery_dir: str = 'gallery_drone'):
        """
        Initialize test dataset.
        
        Args:
            root: Root directory of test data
            transform: Transform to apply to images
            query_dir: Query directory name
            gallery_dir: Gallery directory name
        """
        super(UniversityTestDataset, self).__init__()
        
        self.root = root
        self.transform = transform
        self.query_dir = query_dir
        self.gallery_dir = gallery_dir
        
        # Build query and gallery lists
        self.query_list = self._build_image_list(query_dir)
        self.gallery_list = self._build_image_list(gallery_dir)
        
    def _build_image_list(self, subdir: str) -> List[Tuple[str, int]]:
        """Build list of (image_path, class_id) tuples."""
        image_list = []
        subdir_path = os.path.join(self.root, subdir)
        
        if not os.path.exists(subdir_path):
            raise FileNotFoundError(f"Directory not found: {subdir_path}")
        
        for class_name in os.listdir(subdir_path):
            class_path = os.path.join(subdir_path, class_name)
            if os.path.isdir(class_path):
                class_id = int(class_name)  # Assuming class names are numeric
                
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_file)
                        image_list.append((img_path, class_id))
        
        return sorted(image_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get item by index.
        
        Args:
            index: Index
            
        Returns:
            Tuple of (image, class_id, image_path)
        """
        # Determine if this is a query or gallery sample
        if index < len(self.query_list):
            img_path, class_id = self.query_list[index]
            is_query = True
        else:
            gallery_index = index - len(self.query_list)
            img_path, class_id = self.gallery_list[gallery_index]
            is_query = False
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, class_id, img_path
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.query_list) + len(self.gallery_list)
    
    def get_query_gallery_split(self) -> Tuple[int, int]:
        """Get the split point between query and gallery samples."""
        return len(self.query_list), len(self.gallery_list)


def train_collate_fn(batch: List[Tuple]) -> Tuple[List, List]:
    """
    Custom collate function for training.
    
    Args:
        batch: List of (satellite_img, drone_img, class_idx) tuples
        
    Returns:
        Tuple of ([satellite_batch, labels], [drone_batch, labels])
    """
    satellite_imgs, drone_imgs, labels = zip(*batch)
    
    # Stack images and convert labels to tensor
    satellite_batch = torch.stack(satellite_imgs, dim=0)
    drone_batch = torch.stack(drone_imgs, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return ([satellite_batch, labels_tensor], [drone_batch, labels_tensor])


def test_collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Custom collate function for testing.
    
    Args:
        batch: List of (image, class_id, image_path) tuples
        
    Returns:
        Tuple of (image_batch, class_ids, image_paths)
    """
    images, class_ids, image_paths = zip(*batch)
    
    image_batch = torch.stack(images, dim=0)
    class_ids_tensor = torch.tensor(class_ids, dtype=torch.long)
    
    return image_batch, class_ids_tensor, list(image_paths)
