"""
DistilBERT Patching Utilities

This module provides helpers to replace DistilBERT's native self-attention
with the project-specific KernelSelfAttention while preserving the *exact*
forward interface expected by Hugging Face's DistilBERT TransformerBlock.

Key requirement:
- DistilBertSelfAttention.forward(hidden_states, attention_mask, head_mask, output_attentions)
  returns:
    - (context,) if output_attentions is False
    - (context, attn_probs) if output_attentions is True
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

from transformers import DistilBertModel, DistilBertForSequenceClassification

from qra_attention.attention.kernel_attention import KernelSelfAttention


class KernelAttentionWrapper(nn.Module):
    """Adapter that makes KernelSelfAttention look like DistilBERT attention.

    Handles multiple DistilBERT calling conventions:
    - Old: forward(query, key, value, mask, head_mask, output_attentions)
    - New: forward(hidden_states, attention_mask=None, head_mask=None, output_attentions=False)

    Where mask is (batch, seq_len) with 0=attend, 1=mask (DistilBERT convention).
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
        output_attentions: bool = False,
        **kwargs
    ):
        """
        Forward pass matching DistilBERT MultiHeadSelfAttention interface.
        
        Handles both old signature (query, key, value, mask) and new signature
        where query=hidden_states and mask might be passed as attention_mask kwarg.
        
        Args:
            query: (batch, seq_len, hidden_size) - for self-attn, this is hidden_states
            key: (batch, seq_len, hidden_size) - usually same as query, or None
            value: (batch, seq_len, hidden_size) - usually same as query, or None
            mask: (batch, seq_len) with 0=attend, 1=mask, or None
            head_mask: Not used (for compatibility)
            output_attentions: Whether to return attention weights
            **kwargs: Catches 'attention_mask' if passed as kwarg
            
        Returns:
            tuple: (attention_output, attention_weights) or (attention_output, None)
        """
        # Handle attention_mask passed as kwarg (newer transformers versions)
        if mask is None and 'attention_mask' in kwargs:
            mask = kwargs['attention_mask']
        
        # Reshape mask to (batch, 1, 1, seq_len) if needed
        attention_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # (batch, seq) -> (batch, 1, 1, seq)
                attention_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 4:
                # Already 4D - check if it's (batch, 1, 1, seq) or (batch, 1, seq, seq)
                if mask.shape[2] == 1:
                    # (batch, 1, 1, seq) - use as is
                    attention_mask = mask
                else:
                    # (batch, 1, seq, seq) - this is a full attention matrix, take diagonal or first row
                    # Actually, for DistilBERT this shouldn't happen. Use first row as the mask.
                    attention_mask = mask[:, :, :1, :]  # (batch, 1, 1, seq)
            else:
                # Unexpected shape - just use as is and let KernelSelfAttention raise error
                attention_mask = mask
        
        # Call kernel attention (query is used as hidden_states for self-attention)
        context, attn_probs = self.kernel_attention(
            query,  # hidden_states
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        # DistilBERT TransformerBlock always unpacks 2 values
        if output_attentions:
            return (context, attn_probs)
        return (context, None)




def patch_distilbert_attention(
    model: nn.Module,
    layers_to_patch: List[int],
    kernel_config: Optional[Dict[str, Any]] = None,
    alpha: float = 0.9,
) -> nn.Module:
    """Replace attention in specified DistilBERT layers with kernel attention."""
    if kernel_config is None:
        kernel_config = {"num_features": 128, "sigma": 1.0, "normalize": True}

    if isinstance(model, DistilBertForSequenceClassification):
        base_model = model.distilbert
    elif isinstance(model, DistilBertModel):
        base_model = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    config = base_model.config
    hidden_size = config.dim
    num_heads = config.n_heads

    for layer_idx in layers_to_patch:
        if layer_idx < 0 or layer_idx >= config.n_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {config.n_layers})")

        layer = base_model.transformer.layer[layer_idx]

        kernel_attn = KernelSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=config.attention_dropout,
            alpha=alpha,
            kernel_config=kernel_config,
        )

        # Copy weights from original DistilBERT attention projections
        print(f"  Copying weights from layer {layer_idx}...")
        with torch.no_grad():
            kernel_attn.q_proj.weight.copy_(layer.attention.q_lin.weight)
            kernel_attn.q_proj.bias.copy_(layer.attention.q_lin.bias)
            kernel_attn.k_proj.weight.copy_(layer.attention.k_lin.weight)
            kernel_attn.k_proj.bias.copy_(layer.attention.k_lin.bias)
            kernel_attn.v_proj.weight.copy_(layer.attention.v_lin.weight)
            kernel_attn.v_proj.bias.copy_(layer.attention.v_lin.bias)
            kernel_attn.out_proj.weight.copy_(layer.attention.out_lin.weight)
            kernel_attn.out_proj.bias.copy_(layer.attention.out_lin.bias)

        layer.attention = KernelAttentionWrapper(kernel_attn)
        print(f"Patched layer {layer_idx} with KernelSelfAttention")

    return model


def freeze_layers(
    model: nn.Module,
    freeze_embeddings: bool = True,
    freeze_layer_indices: Optional[List[int]] = None,
) -> Dict[str, int]:
    """Freeze specified layers and embeddings in DistilBERT."""
    if freeze_layer_indices is None:
        freeze_layer_indices = []

    if isinstance(model, DistilBertForSequenceClassification):
        base_model = model.distilbert
    elif isinstance(model, DistilBertModel):
        base_model = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    frozen_count = 0
    trainable_count = 0

    if freeze_embeddings:
        for param in base_model.embeddings.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        print("Froze embeddings")

    for layer_idx in freeze_layer_indices:
        layer = base_model.transformer.layer[layer_idx]
        for param in layer.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        print(f"Froze layer {layer_idx}")

    for param in model.parameters():
        if param.requires_grad:
            trainable_count += param.numel()

    return {"frozen_params": frozen_count, "trainable_params": trainable_count}


def verify_patch(model: nn.Module, expected_patched_layers: List[int]) -> Dict[str, Any]:
    """Verify that patching was successful."""
    if isinstance(model, DistilBertForSequenceClassification):
        base_model = model.distilbert
    elif isinstance(model, DistilBertModel):
        base_model = model
    else:
        return {
            "success": False,
            "patched_layers": [],
            "issues": [f"Unsupported model type: {type(model)}"],
        }

    patched_layers: List[int] = []
    issues: List[str] = []

    for layer_idx in range(base_model.config.n_layers):
        layer = base_model.transformer.layer[layer_idx]
        is_kernel_attention = isinstance(layer.attention, KernelAttentionWrapper)

        if is_kernel_attention:
            patched_layers.append(layer_idx)
            if layer_idx not in expected_patched_layers:
                issues.append(f"Layer {layer_idx} is patched but not expected")
        else:
            if layer_idx in expected_patched_layers:
                issues.append(f"Layer {layer_idx} should be patched but is not")

    for expected_idx in expected_patched_layers:
        if expected_idx not in patched_layers:
            issues.append(f"Expected layer {expected_idx} is not patched")

    return {
        "success": len(issues) == 0,
        "patched_layers": patched_layers,
        "issues": issues,
    }


def get_trainable_params_summary(model: nn.Module) -> Dict[str, Any]:
    """Get a summary of trainable vs frozen parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_percentage": trainable_pct,
    }