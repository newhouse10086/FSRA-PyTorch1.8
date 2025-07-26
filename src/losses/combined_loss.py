"""Combined loss functions for FSRA project."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .triplet_loss import TripletLoss


class CombinedLoss(nn.Module):
    """Combined loss function for FSRA training."""
    
    def __init__(self, num_classes: int, triplet_weight: float = 0.3,
                 kl_weight: float = 0.0, ca_weight: float = 1.0,
                 use_kl_loss: bool = False):
        super(CombinedLoss, self).__init__()
        
        self.triplet_weight = triplet_weight
        self.kl_weight = kl_weight
        self.ca_weight = ca_weight
        self.use_kl_loss = use_kl_loss
        
        # Classification loss
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Triplet loss
        if triplet_weight > 0:
            self.triplet_loss = TripletLoss(margin=triplet_weight)
        else:
            self.triplet_loss = None
        
        # KL divergence loss for mutual learning
        if use_kl_loss:
            self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        else:
            self.kl_loss = None
        
        # Cross attention loss
        self.ca_loss = nn.MSELoss()
    
    def forward(self, outputs_s: List[torch.Tensor], outputs_d: List[torch.Tensor],
                features_s: List[torch.Tensor], features_d: List[torch.Tensor],
                labels_s: torch.Tensor, labels_d: torch.Tensor,
                ca_result_s: Optional[torch.Tensor] = None,
                ca_result_d: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            outputs_s: Satellite predictions
            outputs_d: Drone predictions  
            features_s: Satellite features
            features_d: Drone features
            labels_s: Satellite labels
            labels_d: Drone labels
            ca_result_s: Cross attention result for satellite
            ca_result_d: Cross attention result for drone
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Classification loss
        cls_loss_s = self.classification_loss(outputs_s[0], labels_s)
        cls_loss_d = self.classification_loss(outputs_d[0], labels_d)
        cls_loss = cls_loss_s + cls_loss_d
        
        losses['classification'] = cls_loss
        total_loss += cls_loss
        
        # Triplet loss
        if self.triplet_loss is not None and self.triplet_weight > 0:
            triplet_loss_s = self.triplet_loss(features_s[0], labels_s)
            triplet_loss_d = self.triplet_loss(features_d[0], labels_d)
            triplet_loss = triplet_loss_s + triplet_loss_d
            
            losses['triplet'] = triplet_loss
            total_loss += self.triplet_weight * triplet_loss
        else:
            losses['triplet'] = torch.tensor(0.0, device=labels_s.device)
        
        # KL divergence loss for mutual learning
        if self.kl_loss is not None and self.use_kl_loss and self.kl_weight > 0:
            # Compute KL divergence between satellite and drone predictions
            log_prob_s = F.log_softmax(outputs_s[0], dim=1)
            prob_d = F.softmax(outputs_d[0], dim=1)
            
            log_prob_d = F.log_softmax(outputs_d[0], dim=1)
            prob_s = F.softmax(outputs_s[0], dim=1)
            
            kl_loss = (self.kl_loss(log_prob_s, prob_d) + 
                      self.kl_loss(log_prob_d, prob_s)) / 2
            
            losses['kl'] = kl_loss
            total_loss += self.kl_weight * kl_loss
        else:
            losses['kl'] = torch.tensor(0.0, device=labels_s.device)
        
        # Cross attention loss
        if ca_result_s is not None and ca_result_d is not None and self.ca_weight > 0:
            ca_loss = self.ca_loss(ca_result_s, ca_result_d)
            losses['cross_attention'] = ca_loss
            total_loss += self.ca_weight * ca_loss
        else:
            losses['cross_attention'] = torch.tensor(0.0, device=labels_s.device)
        
        losses['total'] = total_loss
        
        return losses


class MultiScaleLoss(nn.Module):
    """Multi-scale loss for different granularities."""
    
    def __init__(self, num_classes: int, num_scales: int = 3,
                 scale_weights: Optional[List[float]] = None):
        super(MultiScaleLoss, self).__init__()
        
        self.num_scales = num_scales
        
        if scale_weights is None:
            self.scale_weights = [1.0] * num_scales
        else:
            self.scale_weights = scale_weights
        
        # Classification losses for different scales
        self.classification_losses = nn.ModuleList([
            nn.CrossEntropyLoss() for _ in range(num_scales)
        ])
    
    def forward(self, predictions: List[torch.Tensor], 
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale loss.
        
        Args:
            predictions: List of predictions at different scales
            labels: Ground truth labels
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        for i, (pred, weight) in enumerate(zip(predictions, self.scale_weights)):
            if isinstance(pred, list):
                # Handle case where prediction is [logits, features]
                pred = pred[0]
            
            scale_loss = self.classification_losses[i](pred, labels)
            losses[f'scale_{i}'] = scale_loss
            total_loss += weight * scale_loss
        
        losses['total'] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CenterLoss(nn.Module):
    """Center loss for feature learning."""
    
    def __init__(self, num_classes: int, feat_dim: int, use_gpu: bool = True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute center loss.
        
        Args:
            x: Features of shape (batch_size, feat_dim)
            labels: Labels of shape (batch_size,)
            
        Returns:
            Center loss value
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


def calculate_loss(outputs_s, outputs_d, features_s, features_d, 
                  labels_s, labels_d, criterion, triplet_loss, 
                  config, ca_result_s=None, ca_result_d=None):
    """
    Calculate combined loss (backward compatibility function).
    
    Args:
        outputs_s: Satellite outputs
        outputs_d: Drone outputs
        features_s: Satellite features
        features_d: Drone features
        labels_s: Satellite labels
        labels_d: Drone labels
        criterion: Classification criterion
        triplet_loss: Triplet loss function
        config: Configuration object
        ca_result_s: Cross attention result for satellite
        ca_result_d: Cross attention result for drone
        
    Returns:
        Dictionary of losses
    """
    combined_loss = CombinedLoss(
        num_classes=config.model.num_classes,
        triplet_weight=config.training.triplet_loss_weight,
        kl_weight=config.training.kl_loss_weight,
        use_kl_loss=config.training.use_kl_loss
    )
    
    return combined_loss(outputs_s, outputs_d, features_s, features_d,
                        labels_s, labels_d, ca_result_s, ca_result_d)
