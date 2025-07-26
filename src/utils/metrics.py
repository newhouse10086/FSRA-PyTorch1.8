"""Metrics and evaluation utilities for FSRA project."""

import torch
import numpy as np
from typing import Tuple, List


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update statistics with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, 
             topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model predictions of shape (batch_size, num_classes)
        target: Ground truth labels of shape (batch_size,)
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_distance_matrix(query_features: torch.Tensor, 
                          gallery_features: torch.Tensor,
                          metric: str = 'euclidean') -> torch.Tensor:
    """
    Compute distance matrix between query and gallery features.
    
    Args:
        query_features: Query features of shape (num_queries, feature_dim)
        gallery_features: Gallery features of shape (num_gallery, feature_dim)
        metric: Distance metric ('euclidean' or 'cosine')
        
    Returns:
        Distance matrix of shape (num_queries, num_gallery)
    """
    if metric == 'euclidean':
        # Euclidean distance
        m, n = query_features.size(0), gallery_features.size(0)
        xx = torch.pow(query_features, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(gallery_features, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = dist - 2 * torch.matmul(query_features, gallery_features.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
    elif metric == 'cosine':
        # Cosine distance
        query_norm = torch.nn.functional.normalize(query_features, p=2, dim=1)
        gallery_norm = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
        dist = 1 - torch.matmul(query_norm, gallery_norm.t())
        
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return dist


def evaluate_ranking(dist_matrix: torch.Tensor, query_labels: torch.Tensor,
                    gallery_labels: torch.Tensor, query_cams: torch.Tensor = None,
                    gallery_cams: torch.Tensor = None) -> Tuple[float, List[float]]:
    """
    Evaluate ranking performance.
    
    Args:
        dist_matrix: Distance matrix of shape (num_queries, num_gallery)
        query_labels: Query labels of shape (num_queries,)
        gallery_labels: Gallery labels of shape (num_gallery,)
        query_cams: Query camera IDs (optional)
        gallery_cams: Gallery camera IDs (optional)
        
    Returns:
        Tuple of (mAP, CMC)
    """
    num_queries, num_gallery = dist_matrix.shape
    
    # Sort by distance
    indices = torch.argsort(dist_matrix, dim=1)
    
    # Compute mAP and CMC
    APs = []
    CMC = torch.zeros(num_gallery)
    
    for i in range(num_queries):
        # Get query info
        query_label = query_labels[i]
        query_cam = query_cams[i] if query_cams is not None else None
        
        # Get sorted gallery info
        sorted_gallery_labels = gallery_labels[indices[i]]
        sorted_gallery_cams = gallery_cams[indices[i]] if gallery_cams is not None else None
        
        # Find valid gallery samples (same identity, different camera)
        if query_cam is not None and sorted_gallery_cams is not None:
            valid_mask = (sorted_gallery_labels == query_label) & (sorted_gallery_cams != query_cam)
        else:
            valid_mask = (sorted_gallery_labels == query_label)
        
        if valid_mask.sum() == 0:
            continue
        
        # Compute AP
        valid_indices = torch.where(valid_mask)[0]
        ap = 0
        for j, idx in enumerate(valid_indices):
            precision = (j + 1) / (idx + 1)
            ap += precision
        ap /= len(valid_indices)
        APs.append(ap)
        
        # Update CMC
        first_match = valid_indices[0]
        CMC[first_match:] += 1
    
    # Compute final metrics
    mAP = np.mean(APs) if APs else 0.0
    CMC = CMC / num_queries
    
    return mAP, CMC.tolist()


def compute_map(query_features: torch.Tensor, gallery_features: torch.Tensor,
                query_labels: torch.Tensor, gallery_labels: torch.Tensor,
                query_cams: torch.Tensor = None, gallery_cams: torch.Tensor = None,
                metric: str = 'euclidean') -> Tuple[float, List[float]]:
    """
    Compute mAP and CMC for retrieval evaluation.
    
    Args:
        query_features: Query features
        gallery_features: Gallery features
        query_labels: Query labels
        gallery_labels: Gallery labels
        query_cams: Query camera IDs (optional)
        gallery_cams: Gallery camera IDs (optional)
        metric: Distance metric
        
    Returns:
        Tuple of (mAP, CMC)
    """
    # Compute distance matrix
    dist_matrix = compute_distance_matrix(query_features, gallery_features, metric)
    
    # Evaluate ranking
    mAP, CMC = evaluate_ranking(dist_matrix, query_labels, gallery_labels, 
                               query_cams, gallery_cams)
    
    return mAP, CMC


class RetrievalMetrics:
    """Class for computing retrieval metrics."""
    
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.reset()
    
    def reset(self):
        """Reset all stored features and labels."""
        self.query_features = []
        self.gallery_features = []
        self.query_labels = []
        self.gallery_labels = []
        self.query_cams = []
        self.gallery_cams = []
    
    def update(self, features: torch.Tensor, labels: torch.Tensor, 
               is_query: bool, cams: torch.Tensor = None):
        """Update with new features and labels."""
        if is_query:
            self.query_features.append(features.cpu())
            self.query_labels.append(labels.cpu())
            if cams is not None:
                self.query_cams.append(cams.cpu())
        else:
            self.gallery_features.append(features.cpu())
            self.gallery_labels.append(labels.cpu())
            if cams is not None:
                self.gallery_cams.append(cams.cpu())
    
    def compute(self) -> Tuple[float, List[float]]:
        """Compute final metrics."""
        # Concatenate all features and labels
        query_features = torch.cat(self.query_features, dim=0)
        gallery_features = torch.cat(self.gallery_features, dim=0)
        query_labels = torch.cat(self.query_labels, dim=0)
        gallery_labels = torch.cat(self.gallery_labels, dim=0)
        
        query_cams = None
        gallery_cams = None
        if self.query_cams and self.gallery_cams:
            query_cams = torch.cat(self.query_cams, dim=0)
            gallery_cams = torch.cat(self.gallery_cams, dim=0)
        
        # Compute metrics
        return compute_map(query_features, gallery_features, query_labels, 
                          gallery_labels, query_cams, gallery_cams, self.metric)


def print_metrics(mAP: float, CMC: List[float], ranks: List[int] = [1, 5, 10, 20]):
    """Print retrieval metrics in a formatted way."""
    print(f"mAP: {mAP:.4f}")
    print("CMC curve:")
    for rank in ranks:
        if rank <= len(CMC):
            print(f"  Rank-{rank}: {CMC[rank-1]:.4f}")
        else:
            print(f"  Rank-{rank}: N/A")
