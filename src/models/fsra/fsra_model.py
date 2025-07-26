"""FSRA model implementation."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from ..backbones.vit_pytorch import vit_small_patch16_224_fsra
from .components import ClassBlock, GeM, weights_init_kaiming, weights_init_classifier


class FSRAModel(nn.Module):
    """FSRA model for geo-localization."""
    
    def __init__(self, num_classes: int, block_size: int = 4, return_f: bool = False,
                 img_size: Tuple[int, int] = (256, 256), stride_size: int = 16,
                 drop_rate: float = 0., attn_drop_rate: float = 0., 
                 drop_path_rate: float = 0.1):
        super(FSRAModel, self).__init__()
        
        self.num_classes = num_classes
        self.block_size = block_size
        self.return_f = return_f
        
        # Backbone transformer
        self.backbone = vit_small_patch16_224_fsra(
            img_size=img_size,
            stride_size=stride_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            local_feature=False,
            num_classes=num_classes
        )
        
        # Feature dimension from ViT-Small
        self.feature_dim = 768
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # GeM pooling for better feature aggregation
        self.gem = GeM(dim=self.feature_dim)
        
        # Classification blocks for different granularities
        self.global_classifier = ClassBlock(
            input_dim=self.feature_dim,
            class_num=num_classes,
            droprate=0.5,
            relu=False,
            bnorm=True,
            num_bottleneck=512,
            linear=True,
            return_f=return_f
        )
        
        # Local classifiers for multi-scale features
        self.local_classifiers = nn.ModuleList([
            ClassBlock(
                input_dim=self.feature_dim,
                class_num=num_classes,
                droprate=0.5,
                relu=False,
                bnorm=True,
                num_bottleneck=512,
                linear=True,
                return_f=return_f
            ) for _ in range(block_size)
        ])
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.feature_dim * (block_size + 1), self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final classifier
        self.final_classifier = ClassBlock(
            input_dim=self.feature_dim,
            class_num=num_classes,
            droprate=0.5,
            relu=False,
            bnorm=True,
            num_bottleneck=512,
            linear=True,
            return_f=return_f
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        self.fusion_layer.apply(weights_init_kaiming)
    
    def forward(self, x: torch.Tensor) -> Tuple[List, List]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (predictions, features)
        """
        # Extract features from backbone
        features, all_features = self.backbone(x)
        
        # Global feature from CLS token
        global_feat = features[:, 0]  # CLS token
        
        # Local features from patch tokens
        patch_features = features[:, 1:]  # Remove CLS token
        B, N, D = patch_features.shape
        
        # Divide patches into blocks for local features
        block_size = min(self.block_size, N)
        patches_per_block = N // block_size
        
        local_feats = []
        for i in range(block_size):
            start_idx = i * patches_per_block
            end_idx = start_idx + patches_per_block if i < block_size - 1 else N
            block_feat = patch_features[:, start_idx:end_idx].mean(dim=1)  # Average pooling
            local_feats.append(block_feat)
        
        # Global classification
        global_pred = self.global_classifier(global_feat)
        
        # Local classifications
        local_preds = []
        for i, local_feat in enumerate(local_feats):
            if i < len(self.local_classifiers):
                local_pred = self.local_classifiers[i](local_feat)
                local_preds.append(local_pred)
        
        # Feature fusion
        all_feats = [global_feat] + local_feats
        fused_feat = torch.cat(all_feats, dim=1)
        fused_feat = self.fusion_layer(fused_feat)
        
        # Final prediction
        final_pred = self.final_classifier(fused_feat)
        
        # Prepare outputs
        if self.return_f:
            predictions = [global_pred[0]] + [pred[0] for pred in local_preds] + [final_pred[0]]
            features = [global_pred[1]] + [pred[1] for pred in local_preds] + [final_pred[1]]
        else:
            predictions = [global_pred] + local_preds + [final_pred]
            features = [global_feat] + local_feats + [fused_feat]
        
        return predictions, features
    
    def load_pretrained(self, model_path: str):
        """Load pretrained weights."""
        self.backbone.load_param(model_path)


class TwoViewNet(nn.Module):
    """Two-view network for satellite and drone images."""
    
    def __init__(self, num_classes: int, block_size: int = 4, return_f: bool = False,
                 share_weights: bool = True):
        super(TwoViewNet, self).__init__()
        
        self.share_weights = share_weights
        
        # First model for satellite images
        self.model_1 = FSRAModel(
            num_classes=num_classes,
            block_size=block_size,
            return_f=return_f
        )
        
        # Second model for drone images (shared or separate)
        if share_weights:
            self.model_2 = self.model_1
        else:
            self.model_2 = FSRAModel(
                num_classes=num_classes,
                block_size=block_size,
                return_f=return_f
            )
    
    def forward(self, x1: Optional[torch.Tensor], x2: Optional[torch.Tensor]):
        """
        Forward pass for two views.
        
        Args:
            x1: Satellite image tensor
            x2: Drone image tensor
            
        Returns:
            Tuple of outputs from both models
        """
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)
        
        if x2 is None:
            y2 = None
        else:
            y2 = self.model_2(x2)
        
        return y1, y2


def make_fsra_model(num_classes: int, block_size: int = 4, return_f: bool = False,
                    views: int = 2, share_weights: bool = True) -> nn.Module:
    """
    Create FSRA model.
    
    Args:
        num_classes: Number of classes
        block_size: Number of local blocks
        return_f: Whether to return features
        views: Number of views (1 or 2)
        share_weights: Whether to share weights between views
        
    Returns:
        FSRA model
    """
    if views == 1:
        return FSRAModel(
            num_classes=num_classes,
            block_size=block_size,
            return_f=return_f
        )
    elif views == 2:
        return TwoViewNet(
            num_classes=num_classes,
            block_size=block_size,
            return_f=return_f,
            share_weights=share_weights
        )
    else:
        raise ValueError(f"Unsupported number of views: {views}")
