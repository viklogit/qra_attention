"""
Random Fourier Features (RFF) Kernel Implementation

This module implements an RFF approximation of the RBF kernel for use in
attention mechanisms. The RFF kernel provides a quantum-ready drop-in replacement
for standard dot-product similarity.

Mathematical Foundation:
    k(x, y) ≈ φ(x)ᵀ φ(y)
    
    where φ(x) = √(2/m) cos(Wx + b)
    
    W ∈ ℝ^(m × d_k) with entries ~ N(0, σ⁻²)
    b ~ Uniform(0, 2π)
    m = number of random features
    σ = kernel bandwidth

Reference: Protocol v1.0, Section 2 (Kernel Attention Definition)
"""

import torch
import torch.nn as nn
import math


class RFFKernel(nn.Module):
    """
    Random Fourier Features kernel for attention mechanisms.
    
    This implementation uses fixed random features (MVE-safe approach) where
    W and b are sampled once and registered as buffers (not trained). This
    ensures that "kernel geometry does the work" rather than learned projections.
    
    Args:
        input_dim (int): Dimension of input features (d_k)
        num_features (int): Number of random features (m)
        sigma (float): Kernel bandwidth parameter
        device (str or torch.device): Device to place tensors on
        normalize (bool): If True, normalize feature maps to unit norm (default: True)
        
    Attributes:
        W (torch.Tensor): Random projection matrix, shape (num_features, input_dim)
        b (torch.Tensor): Random phase shifts, shape (num_features,)
        scale (float): Normalization factor √(2/m)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_features: int,
        sigma: float = 1.0,
        device: str = "cpu",
        normalize: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.sigma = sigma
        self.normalize = normalize
        self.scale = math.sqrt(2.0 / num_features)
        
        # Sample W from N(0, σ⁻²)
        # Using fixed random features (not trainable)
        W = torch.randn(num_features, input_dim, device=device) / sigma
        self.register_buffer('W', W)
        
        # Sample b from Uniform(0, 2π)
        b = torch.rand(num_features, device=device) * 2 * math.pi
        self.register_buffer('b', b)
    
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the RFF feature map φ(x).
        
        φ(x) = √(2/m) cos(Wx + b)
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., input_dim)
            
        Returns:
            torch.Tensor: Feature map of shape (..., num_features)
        """
        # Compute Wx + b
        # x: (..., input_dim), W: (num_features, input_dim)
        # Result: (..., num_features)
        projection = torch.matmul(x, self.W.t()) + self.b
        
        # Apply cosine and scale
        features = self.scale * torch.cos(projection)
        
        # Optional: normalize to unit norm to prevent softmax saturation
        if self.normalize:
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return features
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel similarity matrix S_ij = φ(q_i)ᵀ φ(k_j).
        
        This replaces the standard dot-product similarity in attention:
            Standard: S_ij = q_iᵀ k_j / √d_k
            RFF Kernel: S_ij = φ(q_i)ᵀ φ(k_j)
        
        Args:
            q (torch.Tensor): Query tensor of shape (batch, num_heads, seq_len_q, input_dim)
            k (torch.Tensor): Key tensor of shape (batch, num_heads, seq_len_k, input_dim)
            
        Returns:
            torch.Tensor: Similarity matrix of shape (batch, num_heads, seq_len_q, seq_len_k)
        """
        # Compute feature maps
        phi_q = self.phi(q)  # (batch, num_heads, seq_len_q, num_features)
        phi_k = self.phi(k)  # (batch, num_heads, seq_len_k, num_features)
        
        # Compute similarity: φ(q) @ φ(k)ᵀ
        # Result: (batch, num_heads, seq_len_q, seq_len_k)
        similarity = torch.matmul(phi_q, phi_k.transpose(-2, -1))
        
        return similarity
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"input_dim={self.input_dim}, "
            f"num_features={self.num_features}, "
            f"sigma={self.sigma}, "
            f"normalize={self.normalize}"
        )


def compute_rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute the true RBF kernel for comparison.
    
    k(x, y) = exp(-||x - y||² / (2σ²))
    
    Args:
        x (torch.Tensor): First input, shape (..., n, d)
        y (torch.Tensor): Second input, shape (..., m, d)
        sigma (float): Bandwidth parameter
        
    Returns:
        torch.Tensor: RBF kernel matrix, shape (..., n, m)
    """
    # Compute pairwise squared distances
    # ||x - y||² = ||x||² + ||y||² - 2x·y
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)  # (..., n, 1)
    y_norm = (y ** 2).sum(dim=-1, keepdim=True)  # (..., m, 1)
    
    dist_sq = x_norm + y_norm.transpose(-2, -1) - 2 * torch.matmul(x, y.transpose(-2, -1))
    
    # Apply RBF kernel
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))
    
    return kernel
