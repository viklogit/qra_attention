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
        num_heads: int = 12,
        sigma: float = 1.0,
        device: str = "cpu",
        normalize: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.num_heads = num_heads
        self.sigma = sigma
        self.normalize = normalize
        self.scale = math.sqrt(2.0 / num_features)
        
        # Sample W from N(0, σ⁻²)
        # Shape: (num_heads, num_features, input_dim)
        W = torch.randn(num_heads, num_features, input_dim, device=device) / sigma
        self.register_buffer('W', W)
        
        # Sample b from Uniform(0, 2π)
        # Shape: (num_heads, num_features)
        b = torch.rand(num_heads, num_features, device=device) * 2 * math.pi
        self.register_buffer('b', b)
    
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the RFF feature map φ(x) in a vectorized manner over multiple heads.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, heads, seq, input_dim)
            
        Returns:
            torch.Tensor: Feature map of shape (batch, heads, seq, num_features)
        """
        # x: (B, H, L, D)
        # W: (H, M, D) -> t() -> (H, D, M)
        # b: (H, M) -> view -> (1, H, 1, M)
        
        # Batch matrix multiplication: (B, H, L, D) @ (H, D, M) -> (B, H, L, M)
        # We need to use transpose correctly for the head-specific weights
        W_t = self.W.transpose(1, 2)
        projection = torch.matmul(x, W_t) + self.b.view(1, self.num_heads, 1, self.num_features)
        
        # Apply cosine and scale
        features = self.scale * torch.cos(projection)
        
        if self.normalize:
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return features

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel similarity matrix S_ij = φ(q_i)ᵀ φ(k_j) vectorized over heads.
        """
        phi_q = self.phi(q)  # (batch, heads, seq_len_q, num_features)
        phi_k = self.phi(k)  # (batch, heads, seq_len_k, num_features)
        
        # Compute similarity: φ(q) @ φ(k)ᵀ
        # q: (B, H, L1, M), k: (B, H, L2, M) -> (B, H, L1, L2)
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
