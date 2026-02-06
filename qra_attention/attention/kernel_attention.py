"""
Kernel Self-Attention Implementation

This module implements a kernel-based self-attention mechanism that replaces
the standard dot-product similarity (QK^T) with RFF kernel similarity φ(Q)φ(K)^T.

The implementation is a drop-in replacement for standard attention, keeping:
- Q, K, V projections identical
- Softmax and output computation identical
- Only the similarity computation changes

Reference: Protocol v1.0, Section 3 (Implementation Plan)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any

from qra_attention.kernels.rff import RFFKernel


class KernelSelfAttention(nn.Module):
    """
    Multi-head self-attention with RFF kernel similarity.
    
    This replaces the standard attention mechanism:
        Standard: Attention(Q,K,V) = softmax(QK^T / √d_k) V
        Kernel:   Attention(Q,K,V) = softmax(φ(Q)φ(K)^T) V
    
    Args:
        hidden_size (int): Dimension of hidden states
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
        alpha (float): Interpolation factor (default: 0.9)
            0.9 means 90% dot-product, 10% kernel.
        kernel_config (dict): Configuration for RFF kernel
            - num_features (int): Number of random features (m)
            - sigma (float): Kernel bandwidth
            - normalize (bool): Whether to normalize feature maps
        
    Attributes:
        head_dim (int): Dimension per attention head (hidden_size / num_heads)
        q_proj (nn.Linear): Query projection
        k_proj (nn.Linear): Key projection
        v_proj (nn.Linear): Value projection
        out_proj (nn.Linear): Output projection
        kernels (nn.ModuleList): RFF kernels (one per head)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        alpha: float = 0.9,
        kernel_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        
        # Q, K, V projections (standard)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # RFF kernel configuration
        if kernel_config is None:
            kernel_config = {
                'num_features': 128,
                'sigma': 1.0,
                'normalize': True
            }
        
        # Create a single RFF kernel for vectorized computation across all heads
        self.kernel = RFFKernel(
            input_dim=self.head_dim,
            num_features=kernel_config.get('num_features', 128),
            num_heads=self.num_heads,
            sigma=kernel_config.get('sigma', 1.0),
            normalize=kernel_config.get('normalize', True)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> tuple:
        """
        Forward pass of kernel self-attention.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask (torch.Tensor, optional): Mask of shape (batch, 1, 1, seq_len)
                where 0 = attend, 1 = mask
            output_attentions (bool): Whether to return attention weights
            
        Returns:
            tuple: (output, attention_weights) if output_attentions else (output,)
                - output: (batch, seq_len, hidden_size)
                - attention_weights: (batch, num_heads, seq_len, seq_len) if requested
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        # Shape: (batch, seq_len, hidden_size)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        # Shape: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # FIX 2: Unit-norm Q and K to restore angular meaning for kernel
        # Shape: (batch, num_heads, seq_len, head_dim)
        q_normed = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k_normed = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Compute standard dot-product similarity (Raw QK^T)
        # Shape: (batch, num_heads, seq_len, seq_len)
        dot_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Compute kernel similarity vectorized over all heads
        # Shape: (batch, num_heads, seq_len, seq_len)
        kernel_scores = self.kernel(q_normed, k_normed)

        # FIX 1: Hybrid Blending
        # Preservation of pretrained inductive bias + geometric refinement
        attention_scores = self.alpha * dot_scores + (1 - self.alpha) * kernel_scores
        
        if not self.training:
            print(f"Dot scores: {dot_scores.mean():.4f} ± {dot_scores.std():.4f}")
            print(f"Kernel scores: {kernel_scores.mean():.4f} ± {kernel_scores.std():.4f}")
            print(f"Blended scores: {attention_scores.mean():.4f} ± {attention_scores.std():.4f}")
        
        # FIX 4: Temperature Scaling
        # Mirror the sqrt(d_k) scaling of standard attention
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask shape: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            # Convert 0/1 mask to additive mask: 0 -> 0, 1 -> -1e4 (for FP16 safety)
            attention_scores = attention_scores + (attention_mask * -1e4)
        
        # Apply softmax to get attention weights
        # Shape: (batch, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, v)
        
        # Reshape back to (batch, seq_len, hidden_size)
        # Use -1 for batch dimension to be robust against DDP/Gather behaviors
        context = context.transpose(1, 2).contiguous()
        context = context.view(-1, seq_len, self.hidden_size)
        
        # Guard: Ensure we return the expected local batch size
        if context.size(0) > batch_size:
            context = context[:batch_size]
        
        # Apply output projection
        output = self.out_proj(context)
        
        if output_attentions:
            return (output, attention_weights)
        else:
            return (output,)
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"alpha={self.alpha}"
        )
