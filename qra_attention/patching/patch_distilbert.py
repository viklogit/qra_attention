"""
DistilBERT Patching Utilities

This module provides functions to patch DistilBERT models by replacing standard
attention layers with kernel attention, and utilities for freezing layers and
verifying patches.

Reference: Protocol v1.0, Section 1 (Model Intervention)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from transformers import DistilBertModel, DistilBertForSequenceClassification

from qra_attention.attention.kernel_attention import KernelSelfAttention


class KernelAttentionWrapper(nn.Module):
    """
    Wrapper to make KernelSelfAttention compatible with DistilBERT's attention interface.
    
    DistilBERT expects:
        forward(hidden_states, attention_mask, head_mask, output_attentions)
    
    Our KernelSelfAttention expects:
        forward(hidden_states, attention_mask, output_attentions)
    
    This wrapper handles the interface mismatch and head_mask (usually None).
    """
    
    def __init__(self, kernel_attention: KernelSelfAttention):
        super().__init__()
        self.kernel_attention = kernel_attention
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ):
        """
        Forward pass matching DistilBERT attention interface.
        
        DistilBERT's MultiHeadSelfAttention uses:
            forward(query, key, value, mask, head_mask, output_attentions)
        
        For self-attention: query = key = value = hidden_states
        
        Args:
            query: (batch, seq_len, hidden_size)
            key: (batch, seq_len, hidden_size) - usually same as query
            value: (batch, seq_len, hidden_size) - usually same as query
            mask: (batch, seq_len) with 0=attend, 1=mask
            head_mask: Not used (for compatibility)
            output_attentions: Whether to return attention weights
            
        Returns:
            tuple: (attention_output,) or (attention_output, attention_weights)
        """
        # For self-attention, key and value default to query
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Convert attention mask from (batch, seq_len) to (batch, 1, 1, seq_len)
        attention_mask = None
        if mask is not None:
            # DistilBERT mask: 0 = attend, 1 = mask
            # Reshape to (batch, 1, 1, seq_len) for broadcasting
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Call kernel attention (it expects hidden_states, not separate q/k/v)
        # Since this is self-attention, query = key = value
        return self.kernel_attention(query, attention_mask, output_attentions)


def patch_distilbert_attention(
    model: nn.Module,
    layers_to_patch: List[int],
    kernel_config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Replace attention in specified DistilBERT layers with kernel attention.
    
    Args:
        model: DistilBERT model (DistilBertModel or DistilBertForSequenceClassification)
        layers_to_patch: List of layer indices to patch (e.g., [4, 5] for L4 and L5)
        kernel_config: Configuration for RFF kernel (num_features, sigma, normalize)
        
    Returns:
        nn.Module: Patched model (modified in-place, but also returned)
        
    Example:
        >>> from transformers import DistilBertForSequenceClassification
        >>> model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        >>> model = patch_distilbert_attention(model, layers_to_patch=[4, 5])
    """
    if kernel_config is None:
        kernel_config = {
            'num_features': 128,
            'sigma': 1.0,
            'normalize': False  # Important: disable for attention
        }
    
    # Get the base DistilBERT model
    if isinstance(model, DistilBertForSequenceClassification):
        base_model = model.distilbert
    elif isinstance(model, DistilBertModel):
        base_model = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    # Get model config
    config = base_model.config
    hidden_size = config.dim
    num_heads = config.n_heads
    
    # Patch specified layers
    for layer_idx in layers_to_patch:
        if layer_idx < 0 or layer_idx >= config.n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {config.n_layers})")
        
        # Get the layer
        layer = base_model.transformer.layer[layer_idx]
        
        # Create kernel attention
        kernel_attn = KernelSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=config.attention_dropout,
            kernel_config=kernel_config
        )
        
        # Wrap for compatibility
        wrapped_attn = KernelAttentionWrapper(kernel_attn)
        
        # Replace the attention module
        layer.attention = wrapped_attn
        
        print(f"✓ Patched layer {layer_idx} with KernelSelfAttention")
    
    return model


def freeze_layers(
    model: nn.Module,
    freeze_embeddings: bool = True,
    freeze_layer_indices: Optional[List[int]] = None
) -> Dict[str, int]:
    """
    Freeze specified layers and embeddings in DistilBERT.
    
    Args:
        model: DistilBERT model
        freeze_embeddings: Whether to freeze embeddings
        freeze_layer_indices: List of layer indices to freeze (e.g., [0, 1, 2, 3])
        
    Returns:
        dict: Summary with 'frozen_params' and 'trainable_params' counts
        
    Example:
        >>> model = patch_distilbert_attention(model, layers_to_patch=[4, 5])
        >>> summary = freeze_layers(model, freeze_embeddings=True, freeze_layer_indices=[0, 1, 2, 3])
    """
    if freeze_layer_indices is None:
        freeze_layer_indices = []
    
    # Get the base DistilBERT model
    if isinstance(model, DistilBertForSequenceClassification):
        base_model = model.distilbert
    elif isinstance(model, DistilBertModel):
        base_model = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    frozen_count = 0
    trainable_count = 0
    
    # Freeze embeddings
    if freeze_embeddings:
        for param in base_model.embeddings.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        print(f"✓ Froze embeddings")
    
    # Freeze specified layers
    for layer_idx in freeze_layer_indices:
        layer = base_model.transformer.layer[layer_idx]
        for param in layer.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        print(f"✓ Froze layer {layer_idx}")
    
    # Count trainable parameters
    for param in model.parameters():
        if param.requires_grad:
            trainable_count += param.numel()
    
    return {
        'frozen_params': frozen_count,
        'trainable_params': trainable_count
    }


def verify_patch(
    model: nn.Module,
    expected_patched_layers: List[int]
) -> Dict[str, Any]:
    """
    Verify that patching was successful.
    
    Args:
        model: Patched DistilBERT model
        expected_patched_layers: List of layer indices that should be patched
        
    Returns:
        dict: Verification report with 'success', 'patched_layers', 'issues'
        
    Example:
        >>> report = verify_patch(model, expected_patched_layers=[4, 5])
        >>> assert report['success'], f"Patch verification failed: {report['issues']}"
    """
    # Get the base DistilBERT model
    if isinstance(model, DistilBertForSequenceClassification):
        base_model = model.distilbert
    elif isinstance(model, DistilBertModel):
        base_model = model
    else:
        return {
            'success': False,
            'patched_layers': [],
            'issues': [f"Unsupported model type: {type(model)}"]
        }
    
    patched_layers = []
    issues = []
    
    # Check each layer
    for layer_idx in range(base_model.config.n_layers):
        layer = base_model.transformer.layer[layer_idx]
        
        # Check if this layer has kernel attention
        is_kernel_attention = isinstance(layer.attention, KernelAttentionWrapper)
        
        if is_kernel_attention:
            patched_layers.append(layer_idx)
            
            # Verify it's expected
            if layer_idx not in expected_patched_layers:
                issues.append(f"Layer {layer_idx} is patched but not expected")
        else:
            # Verify it's not expected
            if layer_idx in expected_patched_layers:
                issues.append(f"Layer {layer_idx} should be patched but is not")
    
    # Check if all expected layers are patched
    for expected_idx in expected_patched_layers:
        if expected_idx not in patched_layers:
            issues.append(f"Expected layer {expected_idx} is not patched")
    
    success = len(issues) == 0
    
    return {
        'success': success,
        'patched_layers': patched_layers,
        'issues': issues
    }


def get_trainable_params_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get a summary of trainable vs frozen parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Summary with parameter counts and percentages
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_percentage': trainable_pct
    }
