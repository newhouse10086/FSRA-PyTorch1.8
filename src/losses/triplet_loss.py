"""Triplet loss implementation for FSRA project."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Normalize tensor to unit length along specified dimension."""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-6)
    return x


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute euclidean distance between two tensors.
    
    Args:
        x: Tensor of shape (m, d)
        y: Tensor of shape (n, d)
        
    Returns:
        Distance matrix of shape (m, n)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance between two tensors.
    
    Args:
        x: Tensor of shape (m, d)
        y: Tensor of shape (n, d)
        
    Returns:
        Distance matrix of shape (m, n)
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection / (x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat: torch.Tensor, labels: torch.Tensor, 
                       return_inds: bool = False):
    """
    For each anchor, find the hardest positive and negative sample.
    
    Args:
        dist_mat: Distance matrix of shape (N, N)
        labels: Labels of shape (N,)
        return_inds: Whether to return indices
        
    Returns:
        Tuple of (dist_ap, dist_an) or (dist_ap, dist_an, p_inds, n_inds)
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # Shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    
    # Shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # Shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # Shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # Shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining."""
    
    def __init__(self, margin: float = 0.3, hard_factor: float = 0.0, 
                 normalize_feature: bool = True, distance_metric: str = 'euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_factor = hard_factor
        self.normalize_feature = normalize_feature
        self.distance_metric = distance_metric
        
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, global_feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of triplet loss.
        
        Args:
            global_feat: Feature tensor of shape (batch_size, feat_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Triplet loss value
        """
        if self.normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        
        if self.distance_metric == 'euclidean':
            dist_mat = euclidean_dist(global_feat, global_feat)
        elif self.distance_metric == 'cosine':
            dist_mat = cosine_dist(global_feat, global_feat)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        
        return loss


class HardTripletLoss(nn.Module):
    """Hard triplet loss with custom hard mining strategy."""
    
    def __init__(self, margin: float = 0.3, hard_factor: float = 0.0):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.hard_factor = hard_factor

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with custom hard mining.
        
        Args:
            inputs: Feature matrix of shape (batch_size, feat_dim)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Triplet loss value
        """
        n = inputs.size(0)
        
        # Normalize features
        inputs = normalize(inputs, axis=-1)
        
        # Compute distance matrix
        dist = euclidean_dist(inputs, inputs)
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            if i < n // 2:
                # First half samples
                pos_mask = mask[i][n//2:n]
                neg_mask = (mask[i] == 0)[n//2:n]
                
                if pos_mask.any():
                    dist_ap.append(dist[i][n//2:n][pos_mask].max().unsqueeze(0))
                else:
                    dist_ap.append(torch.tensor(0.0, device=dist.device).unsqueeze(0))
                
                if neg_mask.any():
                    dist_an.append(dist[i][n//2:n][neg_mask].min().unsqueeze(0))
                else:
                    dist_an.append(torch.tensor(1.0, device=dist.device).unsqueeze(0))
            else:
                # Second half samples
                pos_mask = mask[i][0:n//2]
                neg_mask = (mask[i] == 0)[0:n//2]
                
                if pos_mask.any():
                    dist_ap.append(dist[i][0:n//2][pos_mask].max().unsqueeze(0))
                else:
                    dist_ap.append(torch.tensor(0.0, device=dist.device).unsqueeze(0))
                
                if neg_mask.any():
                    dist_an.append(dist[i][0:n//2][neg_mask].min().unsqueeze(0))
                else:
                    dist_an.append(torch.tensor(1.0, device=dist.device).unsqueeze(0))
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Apply hard factor
        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


class BatchHardTripletLoss(nn.Module):
    """Batch hard triplet loss."""
    
    def __init__(self, margin: float = 0.3):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Batch hard triplet loss forward pass.
        
        Args:
            embeddings: Embedding tensor of shape (batch_size, embedding_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Batch hard triplet loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distance matrix
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Get the hardest positive and negative for each anchor
        mask_anchor_positive = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_anchor_negative = labels.unsqueeze(1) != labels.unsqueeze(0)
        
        # Exclude self-comparison
        mask_anchor_positive = mask_anchor_positive.float()
        mask_anchor_positive.fill_diagonal_(0)
        
        # Hardest positive: largest distance among positive pairs
        hardest_positive_dist = (pairwise_dist * mask_anchor_positive).max(dim=1)[0]
        
        # Hardest negative: smallest distance among negative pairs
        max_anchor_negative_dist = (pairwise_dist + 1e5 * (1 - mask_anchor_negative.float())).min(dim=1)[0]
        
        # Triplet loss
        triplet_loss = F.relu(hardest_positive_dist - max_anchor_negative_dist + self.margin)
        
        return triplet_loss.mean()
