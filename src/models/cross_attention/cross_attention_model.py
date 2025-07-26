"""Cross attention model for FSRA."""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class CrossAttentionModel(nn.Module):
    """Cross attention model for feature alignment between different views."""
    
    def __init__(self, d_model: int = 512, block_size: int = 4, 
                 hidden_dim: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        super(CrossAttentionModel, self).__init__()
        
        self.d_model = d_model
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input dimension is d_model * block_size
        input_dim = d_model * block_size
        
        # Linear projections for queries, keys, and values
        self.q_linear = nn.Linear(input_dim, hidden_dim)
        self.k_linear = nn.Linear(input_dim, hidden_dim)
        self.v_linear = nn.Linear(input_dim, hidden_dim)
        
        # Layer normalization
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in [self.q_linear, self.k_linear, self.v_linear, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, queries: torch.Tensor, support_set: torch.Tensor, 
                mode: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross attention.
        
        Args:
            queries: Query features of shape (B, 1, D*block_size)
            support_set: Support set features of shape (B, N, D*block_size)
            mode: Operation mode (0 for training, 1 for testing)
            
        Returns:
            Tuple of (query_prototype, class_prototype)
        """
        # Apply linear transformations
        queries_q = self.q_linear(queries)  # (B, 1, hidden_dim)
        support_k = self.k_linear(support_set)  # (B, N, hidden_dim)
        support_v = self.v_linear(support_set)  # (B, N, hidden_dim)
        queries_v = self.v_linear(queries)  # (B, 1, hidden_dim)
        
        # Apply layer normalization
        queries_q = self.norm_q(queries_q)
        support_k = self.norm_k(support_k)
        support_v = self.norm_v(support_v)
        queries_v = self.norm_v(queries_v)
        
        # Compute attention weights
        # queries_q: (B, 1, hidden_dim), support_k: (B, N, hidden_dim)
        attention_scores = torch.bmm(queries_q, support_k.transpose(1, 2))  # (B, 1, N)
        attention_scores = attention_scores / math.sqrt(self.hidden_dim)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)  # (B, 1, N)
        attention_weights = self.dropout(attention_weights)
        
        # Compute class prototype using attention weights
        class_prototype = torch.bmm(attention_weights, support_v)  # (B, 1, hidden_dim)
        
        # Query prototype
        query_prototype = queries_v  # (B, 1, hidden_dim)
        
        # Apply output projection
        class_prototype = self.output_proj(class_prototype)
        query_prototype = self.output_proj(query_prototype)
        
        # Remove the singleton dimension
        class_prototype = class_prototype.squeeze(1)  # (B, hidden_dim)
        query_prototype = query_prototype.squeeze(1)  # (B, hidden_dim)
        
        if mode == 0:  # Training mode
            return query_prototype, class_prototype
        elif mode == 1:  # Testing mode
            return query_prototype, support_v.squeeze(1) if support_v.size(1) == 1 else support_v.mean(dim=1)
        else:
            raise ValueError(f"Invalid mode: {mode}")


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention model."""
    
    def __init__(self, d_model: int = 512, block_size: int = 4, 
                 num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.d_model = d_model
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = (d_model * block_size) // num_heads
        
        assert (d_model * block_size) % num_heads == 0, "d_model * block_size must be divisible by num_heads"
        
        # Input dimension
        input_dim = d_model * block_size
        
        # Linear projections
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.q_linear, self.k_linear, self.v_linear, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, queries: torch.Tensor, support_set: torch.Tensor, 
                mode: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head cross attention forward pass.
        
        Args:
            queries: Query features
            support_set: Support set features
            mode: Operation mode
            
        Returns:
            Tuple of processed features
        """
        batch_size = queries.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_linear(queries).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(support_set).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(support_set).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model * self.block_size)
        
        output = self.output_proj(attended_values)
        
        # Residual connection and layer normalization
        if output.size() == queries.size():
            output = self.layer_norm(output + queries)
        else:
            output = self.layer_norm(output)
        
        # Return based on mode
        if mode == 0:
            return output.squeeze(1), attended_values.squeeze(1)
        else:
            return output.squeeze(1), V.mean(dim=2).squeeze(1)


# Alias for backward compatibility
CrossTransformer = CrossAttentionModel
