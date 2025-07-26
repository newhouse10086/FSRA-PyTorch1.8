"""Data transforms for FSRA project."""

import torch
import torchvision.transforms as transforms
from typing import List, Tuple
import random
import math
from PIL import Image
import numpy as np


class RandomErasing(object):
    """Random erasing augmentation."""
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class AutoAugment(object):
    """AutoAugment for images."""
    
    def __init__(self):
        self.policies = [
            [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
            [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
            [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
            [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
            [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
            [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
            [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
            [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
            [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
            [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        ]

    def __call__(self, img):
        policy = random.choice(self.policies)
        for name, pr, level in policy:
            if random.random() > pr:
                continue
            img = self._apply_op(img, name, level)
        return img

    def _apply_op(self, img, name, level):
        # Simplified implementation - in practice you'd implement all operations
        if name == 'Equalize':
            return transforms.functional.equalize(img)
        elif name == 'Color':
            return transforms.functional.adjust_saturation(img, 1 + level * 0.1)
        elif name == 'Rotate':
            return transforms.functional.rotate(img, level * 3)
        # Add more operations as needed
        return img


def get_train_transforms(config) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training transforms for satellite and drone images.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (satellite_transforms, drone_transforms)
    """
    # Base transforms
    base_transforms = [
        transforms.Resize((config.data.image_height, config.data.image_width), 
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad(config.data.pad, padding_mode='edge'),
    ]
    
    # Satellite-specific transforms
    satellite_transforms = base_transforms + [
        transforms.RandomAffine(90),  # Random rotation up to 90 degrees
        transforms.RandomCrop((config.data.image_height, config.data.image_width)),
        transforms.RandomHorizontalFlip(),
    ]
    
    # Drone-specific transforms  
    drone_transforms = base_transforms + [
        transforms.RandomCrop((config.data.image_height, config.data.image_width)),
        transforms.RandomHorizontalFlip(),
    ]
    
    # Add color jitter if enabled
    if config.data.color_jitter:
        color_jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
        satellite_transforms.insert(-2, color_jitter)
        drone_transforms.insert(-2, color_jitter)
    
    # Convert to tensor and normalize
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    satellite_transforms.extend(final_transforms)
    drone_transforms.extend(final_transforms)
    
    # Add random erasing if enabled
    if config.data.random_erasing_prob > 0:
        satellite_transforms.append(
            RandomErasing(probability=config.data.random_erasing_prob))
        drone_transforms.append(
            RandomErasing(probability=config.data.random_erasing_prob))
    
    return (transforms.Compose(satellite_transforms), 
            transforms.Compose(drone_transforms))


def get_test_transforms(config) -> transforms.Compose:
    """
    Get test transforms.
    
    Args:
        config: Configuration object
        
    Returns:
        Test transforms
    """
    test_transforms = [
        transforms.Resize((config.data.image_height, config.data.image_width), 
                         interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(test_transforms)


def get_augmentation_transforms(config) -> List[transforms.Compose]:
    """
    Get data augmentation transforms.
    
    Args:
        config: Configuration object
        
    Returns:
        List of augmentation transforms
    """
    augmentations = []
    
    if config.training.use_data_augmentation:
        # Add AutoAugment
        augmentations.append(AutoAugment())
        
        # Add additional augmentations
        augmentations.extend([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.1),
        ])
    
    return augmentations


class TensorToNumpy:
    """Convert tensor to numpy array."""
    
    def __call__(self, tensor):
        return tensor.numpy()


class NumpyToTensor:
    """Convert numpy array to tensor."""
    
    def __call__(self, array):
        return torch.from_numpy(array)


class Denormalize:
    """Denormalize tensor using ImageNet stats."""
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, tensor):
        return tensor * self.std + self.mean
