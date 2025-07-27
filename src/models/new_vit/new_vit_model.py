"""New ViT model with ResNet18 feature extraction and community clustering."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
from networkx.algorithms import community

from ..backbones.vit_pytorch import VisionTransformer, PatchEmbed
from ..fsra.components import ClassBlock


class ResNet18FeatureExtractor(nn.Module):
    """ResNet18 feature extractor for preprocessing images."""
    
    def __init__(self, pretrained: bool = True):
        super(ResNet18FeatureExtractor, self).__init__()
        
        # Load ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add a conv layer to adjust channel dimension for ViT
        self.channel_adapter = nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0)
        
        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using ResNet18.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, 768, 10, 10)
        """
        # Extract features
        features = self.features(x)  # (B, 512, H', W')
        
        # Adapt to target size
        features = self.adaptive_pool(features)  # (B, 512, 10, 10)
        
        # Adjust channels for ViT
        features = self.channel_adapter(features)  # (B, 768, 10, 10)
        
        return features


class CommunityClusteringModule(nn.Module):
    """Community clustering module for patch features."""
    
    def __init__(self, feature_dim: int = 768, num_final_clusters: int = 3):
        super(CommunityClusteringModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_final_clusters = num_final_clusters
        
        # Attention mechanism for computing edge weights
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature projection for clustering
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def compute_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights between patches.
        
        Args:
            features: Patch features of shape (B, N, D)
            
        Returns:
            Attention weights of shape (B, N, N)
        """
        B, N, D = features.shape
        
        # Self-attention to compute relationships
        attn_output, attn_weights = self.attention(features, features, features)
        
        # Average attention weights across heads
        attn_weights = attn_weights.mean(dim=1)  # (B, N, N)
        
        return attn_weights
    
    def community_clustering(self, attention_weights: torch.Tensor) -> List[List[int]]:
        """
        Perform community clustering on attention graph.
        
        Args:
            attention_weights: Attention weights of shape (N, N)
            
        Returns:
            List of communities (each community is a list of node indices)
        """
        # Convert to numpy for networkx
        adj_matrix = attention_weights.detach().cpu().numpy()
        
        # Create graph
        G = nx.from_numpy_array(adj_matrix)
        
        # Perform community detection using Louvain algorithm
        communities = community.greedy_modularity_communities(G)
        
        # Convert to list of lists
        community_list = [list(comm) for comm in communities]
        
        return community_list
    
    def kmeans_clustering(self, features: torch.Tensor, communities: List[List[int]]) -> torch.Tensor:
        """
        Apply K-means clustering to community features.
        
        Args:
            features: Patch features of shape (N, D)
            communities: List of communities
            
        Returns:
            Final cluster assignments of shape (N,)
        """
        # Aggregate features for each community
        community_features = []
        community_indices = []
        
        for comm in communities:
            if len(comm) > 0:
                comm_feat = features[comm].mean(dim=0)  # Average pooling
                community_features.append(comm_feat.detach().cpu().numpy())
                community_indices.extend(comm)
        
        if len(community_features) == 0:
            # Fallback: use all features
            community_features = features.detach().cpu().numpy()
            community_indices = list(range(features.shape[0]))
        
        # Apply K-means clustering
        if len(community_features) > self.num_final_clusters:
            kmeans = KMeans(n_clusters=self.num_final_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(community_features)
        else:
            # If fewer communities than desired clusters, assign each to its own cluster
            cluster_labels = list(range(len(community_features)))
        
        # Map back to original indices
        final_assignments = torch.zeros(features.shape[0], dtype=torch.long)
        for i, comm in enumerate(communities):
            if i < len(cluster_labels):
                for idx in comm:
                    final_assignments[idx] = cluster_labels[i]
        
        return final_assignments
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for community clustering.
        
        Args:
            features: Patch features of shape (B, N, D)
            
        Returns:
            Tuple of (clustered_features, cluster_assignments)
        """
        B, N, D = features.shape
        
        # Project features for clustering
        proj_features = self.feature_proj(features)  # (B, N, 128)
        
        clustered_features = []
        cluster_assignments = []
        
        for b in range(B):
            # Compute attention weights for this sample
            attn_weights = self.compute_attention_weights(features[b:b+1])  # (1, N, N)
            attn_weights = attn_weights.squeeze(0)  # (N, N)
            
            # Community clustering
            communities = self.community_clustering(attn_weights)
            
            # K-means clustering
            assignments = self.kmeans_clustering(proj_features[b], communities)
            
            # Aggregate features by cluster
            cluster_feats = []
            for c in range(self.num_final_clusters):
                mask = (assignments == c)
                if mask.sum() > 0:
                    cluster_feat = features[b][mask].mean(dim=0)
                else:
                    cluster_feat = torch.zeros(D, device=features.device)
                cluster_feats.append(cluster_feat)
            
            clustered_features.append(torch.stack(cluster_feats))  # (num_clusters, D)
            cluster_assignments.append(assignments)
        
        clustered_features = torch.stack(clustered_features)  # (B, num_clusters, D)
        
        return clustered_features, cluster_assignments


class NewViTModel(nn.Module):
    """New ViT model with ResNet18 feature extraction and community clustering."""
    
    def __init__(self, num_classes: int, use_pretrained_resnet: bool = True,
                 use_pretrained_vit: bool = False, num_final_clusters: int = 3,
                 return_f: bool = False):
        super(NewViTModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_final_clusters = num_final_clusters
        self.return_f = return_f
        
        # ResNet18 feature extractor
        self.resnet_extractor = ResNet18FeatureExtractor(pretrained=use_pretrained_resnet)
        
        # Patch embedding for 10x10 patches
        self.patch_embed = PatchEmbed(
            img_size=(10, 10),
            patch_size=1,
            in_chans=768,
            embed_dim=768
        )
        
        # Vision Transformer
        self.vit = VisionTransformer(
            img_size=(10, 10),
            patch_size=1,
            in_chans=768,
            num_classes=0,  # No classification head
            embed_dim=768,
            depth=8,
            num_heads=8,
            mlp_ratio=3.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm
        )
        
        # Community clustering module
        self.clustering = CommunityClusteringModule(
            feature_dim=768,
            num_final_clusters=num_final_clusters
        )
        
        # Global classifier (using CLS token)
        self.global_classifier = ClassBlock(
            input_dim=768,
            class_num=num_classes,
            droprate=0.5,
            relu=False,
            bnorm=True,
            num_bottleneck=512,
            linear=True,
            return_f=return_f
        )
        
        # Local classifiers for each cluster
        self.local_classifiers = nn.ModuleList([
            ClassBlock(
                input_dim=768,
                class_num=num_classes,
                droprate=0.5,
                relu=False,
                bnorm=True,
                num_bottleneck=512,
                linear=True,
                return_f=return_f
            ) for _ in range(num_final_clusters)
        ])
        
        # Feature alignment layer for cross-view matching
        self.alignment_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[List, List]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (predictions, features)
        """
        B = x.shape[0]
        
        # Extract features using ResNet18
        resnet_features = self.resnet_extractor(x)  # (B, 768, 10, 10)
        
        # Patch embedding
        patch_features = self.patch_embed(resnet_features)  # (B, 100, 768)
        
        # Add CLS token
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        vit_input = torch.cat([cls_token, patch_features], dim=1)  # (B, 101, 768)
        
        # Add positional embedding
        if hasattr(self.vit, 'pos_embed'):
            pos_embed = self.vit.pos_embed[:, :vit_input.shape[1], :]
            vit_input = vit_input + pos_embed
        
        # ViT forward pass
        vit_features = self.vit.forward_features(vit_input)  # (B, 101, 768)
        
        # Separate CLS token and patch tokens
        cls_features = vit_features[:, 0]  # (B, 768)
        patch_tokens = vit_features[:, 1:]  # (B, 100, 768)
        
        # Community clustering
        clustered_features, cluster_assignments = self.clustering(patch_tokens)  # (B, num_clusters, 768)
        
        # Global classification
        global_pred = self.global_classifier(cls_features)
        
        # Local classifications
        local_preds = []
        local_features = []
        for i in range(self.num_final_clusters):
            cluster_feat = clustered_features[:, i]  # (B, 768)
            local_pred = self.local_classifiers[i](cluster_feat)
            local_preds.append(local_pred)
            local_features.append(cluster_feat)
        
        # Feature alignment for cross-view matching
        aligned_global = self.alignment_layer(cls_features)
        aligned_locals = [self.alignment_layer(feat) for feat in local_features]
        
        # Prepare outputs
        if self.return_f:
            predictions = [global_pred[0]] + [pred[0] for pred in local_preds]
            features = [global_pred[1]] + [pred[1] for pred in local_preds]
        else:
            predictions = [global_pred] + local_preds
            features = [cls_features] + local_features
        
        return predictions, features
    
    def load_pretrained_vit(self, model_path: str):
        """Load pretrained ViT weights."""
        if hasattr(self.vit, 'load_param'):
            self.vit.load_param(model_path)
        else:
            # Load state dict
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Filter ViT-related weights
            vit_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.') or k.startswith('transformer.'):
                    new_key = k.replace('backbone.', '').replace('transformer.', '')
                    vit_state_dict[new_key] = v
            
            self.vit.load_state_dict(vit_state_dict, strict=False)


class NewTwoViewNet(nn.Module):
    """Two-view network for new ViT model."""

    def __init__(self, num_classes: int, use_pretrained_resnet: bool = True,
                 use_pretrained_vit: bool = False, num_final_clusters: int = 3,
                 return_f: bool = False, share_weights: bool = True):
        super(NewTwoViewNet, self).__init__()

        self.share_weights = share_weights

        # First model for satellite images
        self.model_1 = NewViTModel(
            num_classes=num_classes,
            use_pretrained_resnet=use_pretrained_resnet,
            use_pretrained_vit=use_pretrained_vit,
            num_final_clusters=num_final_clusters,
            return_f=return_f
        )

        # Second model for drone images (shared or separate)
        if share_weights:
            self.model_2 = self.model_1
        else:
            self.model_2 = NewViTModel(
                num_classes=num_classes,
                use_pretrained_resnet=use_pretrained_resnet,
                use_pretrained_vit=use_pretrained_vit,
                num_final_clusters=num_final_clusters,
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


def make_new_vit_model(num_classes: int, use_pretrained_resnet: bool = True,
                       use_pretrained_vit: bool = False, num_final_clusters: int = 3,
                       return_f: bool = False, views: int = 2,
                       share_weights: bool = True) -> nn.Module:
    """
    Create new ViT model.

    Args:
        num_classes: Number of classes
        use_pretrained_resnet: Whether to use pretrained ResNet18
        use_pretrained_vit: Whether to use pretrained ViT
        num_final_clusters: Number of final clusters
        return_f: Whether to return features
        views: Number of views (1 or 2)
        share_weights: Whether to share weights between views

    Returns:
        New ViT model
    """
    if views == 1:
        return NewViTModel(
            num_classes=num_classes,
            use_pretrained_resnet=use_pretrained_resnet,
            use_pretrained_vit=use_pretrained_vit,
            num_final_clusters=num_final_clusters,
            return_f=return_f
        )
    elif views == 2:
        return NewTwoViewNet(
            num_classes=num_classes,
            use_pretrained_resnet=use_pretrained_resnet,
            use_pretrained_vit=use_pretrained_vit,
            num_final_clusters=num_final_clusters,
            return_f=return_f,
            share_weights=share_weights
        )
    else:
        raise ValueError(f"Unsupported number of views: {views}")
