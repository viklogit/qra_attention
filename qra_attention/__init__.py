"""
QRA-Attention: Quantum-Ready Attention via RFF Kernels

A PyTorch implementation of Random Fourier Features (RFF) kernel attention
as a drop-in replacement for standard dot-product attention in Transformers.
"""

__version__ = "0.1.0"
__author__ = "Victor Martinez"

from qra_attention.kernels.rff import RFFKernel

__all__ = ["RFFKernel"]
