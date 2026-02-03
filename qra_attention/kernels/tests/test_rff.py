"""
Unit tests for Random Fourier Features (RFF) kernel implementation.

These tests verify:
1. Feature map output shapes are correct
2. Kernel symmetry: k(x, y) = k(y, x)
3. Positive semi-definiteness of kernel matrices
4. RFF approximation quality vs true RBF kernel
"""

import torch
import numpy as np
import math
from qra_attention.kernels.rff import RFFKernel, compute_rbf_kernel


def test_feature_map_shape():
    """Test that feature map φ(x) produces correct output dimensions."""
    print("Running test_feature_map_shape...")
    
    input_dim = 64
    num_features = 128
    batch_size = 2
    num_heads = 8
    seq_len = 10
    
    kernel = RFFKernel(input_dim=input_dim, num_features=num_features, sigma=1.0)
    
    # Create random input
    x = torch.randn(batch_size, num_heads, seq_len, input_dim)
    
    # Compute feature map
    phi_x = kernel.phi(x)
    
    # Check shape
    expected_shape = (batch_size, num_heads, seq_len, num_features)
    assert phi_x.shape == expected_shape, f"Expected shape {expected_shape}, got {phi_x.shape}"
    
    print(f"✓ Feature map shape test passed: {phi_x.shape}")


def test_kernel_symmetry():
    """Test that kernel matrix is symmetric when x == y: K(x,x) = K(x,x)ᵀ."""
    print("\nRunning test_kernel_symmetry...")
    
    input_dim = 32
    num_features = 64
    
    kernel = RFFKernel(input_dim=input_dim, num_features=num_features, sigma=1.0)
    
    # Create random input
    batch_size = 2
    num_heads = 4
    seq_len = 8
    
    x = torch.randn(batch_size, num_heads, seq_len, input_dim)
    
    # Compute k(x, x)
    K = kernel.forward(x, x)
    
    # Check symmetry: K should equal Kᵀ
    K_transposed = K.transpose(-2, -1)
    
    max_diff = torch.max(torch.abs(K - K_transposed)).item()
    assert max_diff < 1e-5, f"Kernel not symmetric: max difference = {max_diff}"
    
    print(f"✓ Kernel symmetry test passed: max difference = {max_diff:.2e}")


def test_positive_definite():
    """Test that kernel matrix is positive semi-definite."""
    print("\nRunning test_positive_definite...")
    
    input_dim = 32
    num_features = 64
    
    kernel = RFFKernel(input_dim=input_dim, num_features=num_features, sigma=1.0)
    
    # Create random input
    batch_size = 1
    num_heads = 1
    seq_len = 10
    
    x = torch.randn(batch_size, num_heads, seq_len, input_dim)
    
    # Compute kernel matrix k(x, x)
    K = kernel.forward(x, x)
    
    # Extract matrix for eigenvalue computation
    K_matrix = K[0, 0].detach().cpu().numpy()
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(K_matrix)
    
    # Check all eigenvalues are non-negative (allowing small numerical errors)
    min_eigenvalue = np.min(eigenvalues)
    assert min_eigenvalue > -1e-5, f"Negative eigenvalue found: {min_eigenvalue}"
    
    print(f"✓ Positive definite test passed: min eigenvalue = {min_eigenvalue:.2e}")


def test_rbf_approximation():
    """Test that RFF approximates the true RBF kernel."""
    print("\nRunning test_rbf_approximation...")
    
    input_dim = 16
    num_features = 1024  # Increase for better demonstration
    # Use median-style heuristic for sigma to avoid tiny kernel values
    sigma = math.sqrt(input_dim) 
    
    kernel = RFFKernel(
        input_dim=input_dim,
        num_features=num_features,
        sigma=sigma,
        normalize=False
    )
    
    # Test case 1: Identical vectors (should be near 1.0)
    x = torch.randn(1, 1, 10, input_dim)
    K_diag = kernel.forward(x, x)[0, 0]
    K_diag_true = compute_rbf_kernel(x[0, 0], x[0, 0], sigma)
    
    # Test case 2: Different vectors
    y = torch.randn(1, 1, 10, input_dim)
    K_cross = kernel.forward(x, y)[0, 0]
    K_cross_true = compute_rbf_kernel(x[0, 0], y[0, 0], sigma)
    
    # Metrics
    abs_err_diag = torch.abs(K_diag - K_diag_true).mean().item()
    abs_err_cross = torch.abs(K_cross - K_cross_true).mean().item()
    
    print(f"✓ RBF approximation test results (m={num_features}, sigma={sigma:.2f}):")
    print(f"  Mean Absolute Error (Diagonal K(x,x)): {abs_err_diag:.4f}")
    print(f"  Mean Absolute Error (Cross K(x,y)):    {abs_err_cross:.4f}")
    
    # Expected error is O(1/sqrt(m))
    threshold = 2.0 / math.sqrt(num_features)
    assert abs_err_cross < threshold, f"Error {abs_err_cross:.4f} exceeds threshold {threshold:.4f}"

def test_batch_consistency():
    """Test that batched and unbatched computations are equivalent."""
    print("\nRunning test_batch_consistency...")
    
    input_dim = 32
    num_features = 64
    
    kernel = RFFKernel(input_dim=input_dim, num_features=num_features, sigma=1.0)
    
    # Create batched input
    x_batch = torch.randn(4, 2, 10, input_dim)  # (batch=4, heads=2, seq=10, dim=32)
    
    # Compute batched
    K_batch = kernel.forward(x_batch, x_batch)  # (4, 2, 10, 10)
    
    # Compute unbatched (first item)
    x_single = x_batch[0:1]  # (1, 2, 10, 32)
    K_single = kernel.forward(x_single, x_single)  # (1, 2, 10, 10)
    
    # Should be identical
    max_diff = torch.max(torch.abs(K_batch[0] - K_single[0])).item()
    assert max_diff < 1e-6, f"Batch inconsistency: {max_diff}"
    
    print(f"✓ Batch consistency test passed: max difference = {max_diff:.2e}")


def test_device_handling():
    """Test that kernel works on both CPU and CUDA (if available)."""
    print("\nRunning test_device_handling...")
    
    input_dim = 32
    num_features = 64
    
    # Test CPU
    kernel_cpu = RFFKernel(input_dim=input_dim, num_features=num_features, device="cpu")
    x_cpu = torch.randn(1, 1, 5, input_dim)
    K_cpu = kernel_cpu.forward(x_cpu, x_cpu)
    assert K_cpu.device.type == "cpu", "CPU kernel output not on CPU"
    print("✓ CPU device test passed")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        kernel_cuda = RFFKernel(input_dim=input_dim, num_features=num_features, device="cuda")
        x_cuda = torch.randn(1, 1, 5, input_dim, device="cuda")
        K_cuda = kernel_cuda.forward(x_cuda, x_cuda)
        assert K_cuda.device.type == "cuda", "CUDA kernel output not on CUDA"
        print("✓ CUDA device test passed")
    else:
        print("⊘ CUDA not available, skipping CUDA test")


def test_normalization_effect():
    """Test that normalization prevents extreme values."""
    print("\nRunning test_normalization_effect...")
    
    input_dim = 32
    num_features = 64
    
    # Without normalization
    kernel_no_norm = RFFKernel(
        input_dim=input_dim,
        num_features=num_features,
        normalize=False
    )
    
    # With normalization
    kernel_norm = RFFKernel(
        input_dim=input_dim,
        num_features=num_features,
        normalize=True
    )
    
    # Create input with large values
    x = torch.randn(1, 1, 10, input_dim) * 10
    
    K_no_norm = kernel_no_norm.forward(x, x)
    K_norm = kernel_norm.forward(x, x)
    
    max_no_norm = torch.max(torch.abs(K_no_norm)).item()
    max_norm = torch.max(torch.abs(K_norm)).item()
    
    print(f"✓ Normalization test passed:")
    print(f"  Max value without normalization: {max_no_norm:.4f}")
    print(f"  Max value with normalization: {max_norm:.4f}")
    
    # Normalized version should have smaller max values
    assert max_norm < max_no_norm, "Normalization did not reduce max values"

def test_fixed_random_features():
    """Test that W and b are not trainable (registered as buffers)."""
    print("\nRunning test_fixed_random_features...")
    
    input_dim = 32
    num_features = 64
    
    kernel = RFFKernel(input_dim=input_dim, num_features=num_features, sigma=1.0)
    
    # Check that W and b have requires_grad=False
    assert not kernel.W.requires_grad, "W should not require gradients"
    assert not kernel.b.requires_grad, "b should not require gradients"
    
    # Verify they're registered as buffers, not parameters
    param_names = [name for name, _ in kernel.named_parameters()]
    assert 'W' not in param_names, "W should be a buffer, not a parameter"
    assert 'b' not in param_names, "b should be a buffer, not a parameter"
    
    buffer_names = [name for name, _ in kernel.named_buffers()]
    assert 'W' in buffer_names, "W should be registered as a buffer"
    assert 'b' in buffer_names, "b should be registered as a buffer"
    
    print("✓ Fixed random features test passed")
    print(f"  Parameters: {param_names}")
    print(f"  Buffers: {buffer_names}")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running RFF Kernel Unit Tests")
    print("=" * 60)
    
    test_feature_map_shape()
    test_kernel_symmetry()
    test_positive_definite()
    test_rbf_approximation()
    test_batch_consistency()
    test_device_handling()
    test_normalization_effect()
    test_fixed_random_features()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
