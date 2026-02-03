"""
Patching utilities for integrating kernel attention into transformers.
"""

from qra_attention.patching.patch_distilbert import (
    patch_distilbert_attention,
    freeze_layers,
    verify_patch,
    get_trainable_params_summary
)

__all__ = [
    "patch_distilbert_attention",
    "freeze_layers",
    "verify_patch",
    "get_trainable_params_summary"
]
