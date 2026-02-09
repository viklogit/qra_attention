"""
Kernel Self-Attention Implementation

This module implements a kernel-based self-attention mechanism that replaces
(or blends with) the standard dot-product similarity (QK^T) with an RFF-based
kernel similarity φ(Q)φ(K)^T.

Mask convention used here:
- attention_mask is expected as a *binary keep-mask* of shape (batch, 1, 1, seq_len)
    1.0 = keep/attend
    0.0 = mask out
The module converts it internally to an additive mask.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from qra_attention.kernels.rff import RFFKernel


class KernelSelfAttention(nn.Module):
    """Multi-head self-attention with RFF kernel similarity.

    Standard attention:
        softmax(QK^T / sqrt(d_k)) V

    Kernel attention (RFF approximation):
        softmax( φ(Q) φ(K)^T ) V

    Blended similarity:
        scores = alpha * (QK^T / sqrt(d_k)) + (1-alpha) * (φ(Q)φ(K)^T)

    Args:
        hidden_size: model dimension (e.g., 768)
        num_heads: number of attention heads
        dropout: dropout on attention probabilities
        alpha: interpolation factor (1.0 => pure dot-product, 0.0 => pure kernel)
        kernel_config: dict with keys: num_features, sigma, normalize
        debug: if True, prints diagnostic stats
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        alpha: float = 0.9,
        kernel_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = nn.Dropout(dropout)
        self.alpha = float(alpha)
        self.debug = bool(debug)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        if kernel_config is None:
            kernel_config = {"num_features": 128, "sigma": 1.0, "normalize": True}

        self.kernels = nn.ModuleList(
            [
                RFFKernel(
                    input_dim=self.head_dim,
                    num_features=kernel_config.get("num_features", 128),
                    sigma=kernel_config.get("sigma", 1.0),
                    normalize=kernel_config.get("normalize", True),
                )
                for _ in range(num_heads)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,  # kept for API parity
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # (b, h, s, d)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Unit-norm Q/K for RBF-like kernels (stabilizes RFF)
        q_normed = q / (q.norm(dim=-1, keepdim=True) + 1e-5)
        k_normed = k / (k.norm(dim=-1, keepdim=True) + 1e-5)

        dot_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (b,h,s,s)

        # Kernel similarity per head (stable + explicit)
        kernel_scores_list = []
        for head_idx in range(self.num_heads):
            q_head = q_normed[:, head_idx, :, :].unsqueeze(1)  # (b,1,s,d)
            k_head = k_normed[:, head_idx, :, :].unsqueeze(1)  # (b,1,s,d)
            scores_head = self.kernels[head_idx](q_head, k_head)  # (b,1,s,s)
            kernel_scores_list.append(scores_head)
        kernel_scores = torch.cat(kernel_scores_list, dim=1)  # (b,h,s,s)

        attention_scores = self.alpha * dot_scores + (1.0 - self.alpha) * kernel_scores

        if self.debug and (not self.training):
            ds_mean = dot_scores.mean().item() if not torch.isnan(dot_scores).any() else float("nan")
            ks_mean = kernel_scores.mean().item() if not torch.isnan(kernel_scores).any() else float("nan")
            print(f"[KernelSelfAttention] dot_scores mean={ds_mean:.4f} | kernel_scores mean={ks_mean:.4f}")

        # Apply binary keep-mask -> additive mask
        if attention_mask is not None:
            if attention_mask.dim() != 4:
                raise ValueError(f"Expected attention_mask shape (b,1,1,s), got {tuple(attention_mask.shape)}")
            # keep=1 => add 0; mask=0 => add -1e4
            additive = (attention_mask - 1.0) * 10000.0
            attention_scores = attention_scores + additive

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Optional head mask
        if head_mask is not None:
            attention_weights = attention_weights * head_mask

        context = torch.matmul(attention_weights, v)  # (b,h,s,d)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.out_proj(context)
        return output, attention_weights

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"alpha={self.alpha}, debug={self.debug}"
        )
