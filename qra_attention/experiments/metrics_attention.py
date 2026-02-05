"""
Metrics and Visualization Utilities for Attention Analysis.

This module provides tools to quantify and visualize attention distributions,
comparing standard dot-product attention with kernel-based variants.

Metrics:
- Entropy: Information theoretic measure of focus.
- Distance Bias: Tendency of heads to look at local vs. global context.
- Gini Coefficient: Measure of concentration/sparsity.

Visualization:
- Attention heatmaps.
- Distribution plots.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Tuple

def compute_entropy(attention_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of attention distribution.
    
    Args:
        attention_probs: (batch, num_heads, seq_len, seq_len)
        
    Returns:
        torch.Tensor: (batch, num_heads, seq_len) entropy values
    """
    # Avoid log(0) with epsilon
    eps = 1e-12
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + eps), dim=-1)
    return entropy

def compute_distance_bias(attention_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the average token distance weighted by attention.
    
    Args:
        attention_probs: (batch, num_heads, seq_len, seq_len)
        
    Returns:
        torch.Tensor: (batch, num_heads, seq_len) mean distances
    """
    seq_len = attention_probs.size(-1)
    # Create distance matrix (relative positions)
    # distance[i, j] = |i - j|
    indices = torch.arange(seq_len, device=attention_probs.device)
    dist_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)).float()
    
    # distance_bias[b, h, i] = sum_j (attention[b, h, i, j] * |i - j|)
    # (b, h, seq, seq) * (1, 1, seq, seq) -> sum over j
    distance_bias = torch.sum(attention_probs * dist_matrix, dim=-1)
    return distance_bias

def compute_gini(attention_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute the Gini coefficient for each attention distribution.
    Gini = 1 (max sparsity/concentration), Gini = 0 (uniform distribution).
    
    Args:
        attention_probs: (batch, num_heads, seq_len, seq_len)
        
    Returns:
        torch.Tensor: (batch, num_heads, seq_len) Gini coefficients
    """
    batch, heads, seq, _ = attention_probs.shape
    # Flatten across the last dimension and sort
    sorted_probs, _ = torch.sort(attention_probs, dim=-1)
    
    # Calculate Gini
    # formula: G = (sum_{i=1}^n (2i - n - 1) * x_i) / (n * sum x_i)
    # Since it's a probability distribution, sum x_i = 1
    n = sorted_probs.size(-1)
    index = torch.arange(1, n + 1, device=attention_probs.device, dtype=torch.float)
    index = index.view(1, 1, 1, n)
    
    gini = torch.sum((2 * index - n - 1) * sorted_probs, dim=-1) / n
    return gini

def plot_attention_heatmap(
    attention_probs: torch.Tensor, 
    tokens: Optional[list] = None,
    head_idx: int = 0,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None
):
    """
    Plot a heatmap of the attention matrix for a specific head.
    
    Args:
        attention_probs: (seq_len, seq_len)
        tokens: Optional list of token names for labels
        head_idx: Index of the head to plot
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    if len(attention_probs.shape) == 4: # (batch, head, seq, seq)
        attn = attention_probs[0, head_idx].detach().cpu().numpy()
    elif len(attention_probs.shape) == 3: # (head, seq, seq)
        attn = attention_probs[head_idx].detach().cpu().numpy()
    else:
        attn = attention_probs.detach().cpu().numpy()
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn, 
        xticklabels=tokens if tokens else False,
        yticklabels=tokens if tokens else False,
        cmap="viridis"
    )
    plt.title(f"{title} (Head {head_idx})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metric_distribution(
    metric_values: torch.Tensor,
    metric_name: str = "Metric",
    bins: int = 50,
    save_path: Optional[str] = None
):
    """
    Plot the distribution of metric values across all heads and positions.
    """
    vals = metric_values.flatten().detach().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.hist(vals, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
