"""New ViT model with ResNet18 feature extraction and community clustering."""

from .new_vit_model import (
    NewViTModel,
    NewTwoViewNet,
    make_new_vit_model,
    ResNet18FeatureExtractor,
    CommunityClusteringModule
)

__all__ = [
    'NewViTModel',
    'NewTwoViewNet', 
    'make_new_vit_model',
    'ResNet18FeatureExtractor',
    'CommunityClusteringModule'
]
