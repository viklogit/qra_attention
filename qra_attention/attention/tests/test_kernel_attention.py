"""
Unit tests for Kernel Self-Attention implementation.

These tests verify:
1. Output shapes are correct
2. Attention masking works properly
3. Multi-head mechanism functions correctly
4. Gradients flow through trainable parameters
5. Structural compatibility with standard attention
"""

import torch
import torch.nn as nn
import math
from qra_attention.attention.kernel_attention import KernelSelfAttention


def test_output_shape():
    """Test that output shape matches input shape."""
    print("Running test_output_shape...")
    
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 8
    
    attention = KernelSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kernel_config={'num_features': 64, 'sigma': 1.0}
    )
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output = attention(hidden_states)[0]
    
    # Check shape
    expected_shape = (batch_size, seq_len, hidden_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✓ Output shape test passed: {output.shape}")


def test_attention_mask():
    """Test that attention masking works correctly."""
    print("\nRunning test_attention_mask...")
    
    batch_size = 2
    seq_len = 8
    hidden_size = 128
    num_heads = 4
    
    attention = KernelSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=0.0,  # Disable dropout for deterministic test
        kernel_config={'num_features': 64, 'sigma': 1.0}
    )
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create mask: mask out last 2 positions
    # Shape: (batch, 1, 1, seq_len)
    attention_mask = torch.zeros(batch_size, 1, 1, seq_len)
    attention_mask[:, :, :, -2:] = 1  # Mask last 2 positions
    
    # Forward pass with attention weights
    output, attn_weights = attention(hidden_states, attention_mask, output_attentions=True)
    
    # Check that masked positions have near-zero attention
    # attn_weights shape: (batch, num_heads, seq_len, seq_len)
    masked_attention = attn_weights[:, :, :, -2:]  # Attention to masked positions
    max_masked_attn = torch.max(masked_attention).item()
    
    assert max_masked_attn < 1e-5, f"Masked attention should be near zero, got {max_masked_attn}"
    
    print(f"✓ Attention mask test passed: max masked attention = {max_masked_attn:.2e}")


def test_multi_head():
    """Test that multi-head mechanism works correctly."""
    print("\nRunning test_multi_head...")
    
    batch_size = 2
    seq_len = 6
    hidden_size = 192
    num_heads = 6
    
    attention = KernelSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=0.0,  # Disable dropout for deterministic test
        kernel_config={'num_features': 32, 'sigma': 1.0, 'normalize': False}  # Disable normalization
    )
    
    # Verify head_dim calculation
    expected_head_dim = hidden_size // num_heads
    assert attention.head_dim == expected_head_dim, f"Expected head_dim={expected_head_dim}, got {attention.head_dim}"
    
    # Verify we have one kernel per head
    assert len(attention.kernels) == num_heads, f"Expected {num_heads} kernels, got {len(attention.kernels)}"
    
    # Create input and run forward
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    output, attn_weights = attention(hidden_states, output_attentions=True)
    
    # Check attention weights shape
    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, f"Expected {expected_attn_shape}, got {attn_weights.shape}"
    
    # Check that attention weights sum to 1 for each query position
    attn_sums = attn_weights.sum(dim=-1)  # Sum over key positions
    expected_sums = torch.ones_like(attn_sums)
    max_diff = torch.max(torch.abs(attn_sums - expected_sums)).item()
    
    assert max_diff < 1e-5, f"Attention weights should sum to 1, max diff = {max_diff}"
    
    print(f"✓ Multi-head test passed:")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {attention.head_dim}")
    print(f"  Attention weights sum check: max diff = {max_diff:.2e}")


def test_gradient_flow():
    """Test that gradients flow through trainable parameters."""
    print("\nRunning test_gradient_flow...")
    
    batch_size = 2
    seq_len = 4
    hidden_size = 64
    num_heads = 2
    
    attention = KernelSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kernel_config={'num_features': 32, 'sigma': 1.0}
    )
    
    # Create input with requires_grad
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    
    # Forward pass
    output = attention(hidden_states)[0]
    
    # Compute loss and backward
    loss = output.sum()
    loss.backward()
    
    # Check that Q, K, V, out projections have gradients
    trainable_params = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'out_proj.weight']
    for param_name in trainable_params:
        param = dict(attention.named_parameters())[param_name]
        assert param.grad is not None, f"{param_name} should have gradients"
        assert not torch.all(param.grad == 0), f"{param_name} gradients should be non-zero"
    
    # Check that RFF kernel W and b do NOT have gradients (they're buffers)
    for head_idx, kernel in enumerate(attention.kernels):
        assert not kernel.W.requires_grad, f"Kernel {head_idx} W should not require gradients"
        assert not kernel.b.requires_grad, f"Kernel {head_idx} b should not require gradients"
    
    print(f"✓ Gradient flow test passed:")
    print(f"  Trainable parameters have gradients: {trainable_params}")
    print(f"  RFF kernels W and b are fixed (no gradients)")


def test_vs_standard_attention():
    """Compare structure with standard attention."""
    print("\nRunning test_vs_standard_attention...")
    
    batch_size = 2
    seq_len = 8
    hidden_size = 128
    num_heads = 4
    
    kernel_attention = KernelSelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        kernel_config={'num_features': 64, 'sigma': 1.0}
    )
    
    # Count trainable parameters (should be Q, K, V, out projections)
    trainable_params = sum(p.numel() for p in kernel_attention.parameters() if p.requires_grad)
    
    # Expected: 4 linear layers (Q, K, V, out) each with weight and bias
    # Each layer: hidden_size x hidden_size + hidden_size
    expected_params = 4 * (hidden_size * hidden_size + hidden_size)
    
    assert trainable_params == expected_params, f"Expected {expected_params} trainable params, got {trainable_params}"
    
    # Count buffers (should be W and b for each head)
    num_buffers = sum(1 for _ in kernel_attention.buffers())
    expected_buffers = 2 * num_heads  # W and b for each head
    
    assert num_buffers == expected_buffers, f"Expected {expected_buffers} buffers, got {num_buffers}"
    
    print(f"✓ Structure comparison test passed:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Buffers (RFF W, b): {num_buffers}")


def test_different_kernel_configs():
    """Test that different kernel configurations work."""
    print("\nRunning test_different_kernel_configs...")
    
    batch_size = 2
    seq_len = 6
    hidden_size = 96
    num_heads = 3
    
    configs = [
        {'num_features': 32, 'sigma': 0.5, 'normalize': True},
        {'num_features': 128, 'sigma': 2.0, 'normalize': False},
        {'num_features': 64, 'sigma': 1.0, 'normalize': True},
    ]
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    for i, config in enumerate(configs):
        attention = KernelSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_config=config
        )
        
        output = attention(hidden_states)[0]
        assert output.shape == hidden_states.shape, f"Config {i} failed shape check"
    
    print(f"✓ Different kernel configs test passed: tested {len(configs)} configurations")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running Kernel Self-Attention Unit Tests")
    print("=" * 60)
    
    test_output_shape()
    test_attention_mask()
    test_multi_head()
    test_gradient_flow()
    test_vs_standard_attention()
    test_different_kernel_configs()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
