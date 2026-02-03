"""
Unit tests for DistilBERT patching utilities.

These tests verify:
1. Patching single and multiple layers works correctly
2. Layer freezing works as expected
3. Verification function detects patches correctly
4. Parameter counts are accurate
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertConfig

from qra_attention.patching.patch_distilbert import (
    patch_distilbert_attention,
    freeze_layers,
    verify_patch,
    get_trainable_params_summary,
    KernelAttentionWrapper
)


def create_tiny_distilbert():
    """Create a tiny DistilBERT for testing."""
    config = DistilBertConfig(
        vocab_size=1000,
        dim=128,
        n_layers=6,
        n_heads=4,
        hidden_dim=512,
        max_position_embeddings=512
    )
    model = DistilBertModel(config)
    return model


def test_patch_single_layer():
    """Test patching a single layer."""
    print("Running test_patch_single_layer...")
    
    model = create_tiny_distilbert()
    
    # Patch layer 5
    patched_model = patch_distilbert_attention(
        model,
        layers_to_patch=[5],
        kernel_config={'num_features': 32, 'sigma': 1.0, 'normalize': False}
    )
    
    # Verify patch
    report = verify_patch(patched_model, expected_patched_layers=[5])
    
    assert report['success'], f"Patch verification failed: {report['issues']}"
    assert report['patched_layers'] == [5], f"Expected [5], got {report['patched_layers']}"
    
    # Check that layer 5 has KernelAttentionWrapper
    layer_5 = model.transformer.layer[5]
    assert isinstance(layer_5.attention, KernelAttentionWrapper), "Layer 5 should have KernelAttentionWrapper"
    
    # Check that other layers don't
    layer_0 = model.transformer.layer[0]
    assert not isinstance(layer_0.attention, KernelAttentionWrapper), "Layer 0 should not have KernelAttentionWrapper"
    
    print("✓ Single layer patch test passed")


def test_patch_multiple_layers():
    """Test patching multiple layers (L4 and L5)."""
    print("\nRunning test_patch_multiple_layers...")
    
    model = create_tiny_distilbert()
    
    # Patch layers 4 and 5 (protocol specification)
    patched_model = patch_distilbert_attention(
        model,
        layers_to_patch=[4, 5],
        kernel_config={'num_features': 64, 'sigma': 1.0, 'normalize': False}
    )
    
    # Verify patch
    report = verify_patch(patched_model, expected_patched_layers=[4, 5])
    
    assert report['success'], f"Patch verification failed: {report['issues']}"
    assert set(report['patched_layers']) == {4, 5}, f"Expected [4, 5], got {report['patched_layers']}"
    
    # Check that L4 and L5 have kernel attention
    for layer_idx in [4, 5]:
        layer = model.transformer.layer[layer_idx]
        assert isinstance(layer.attention, KernelAttentionWrapper), f"Layer {layer_idx} should have KernelAttentionWrapper"
    
    # Check that L0-L3 don't
    for layer_idx in [0, 1, 2, 3]:
        layer = model.transformer.layer[layer_idx]
        assert not isinstance(layer.attention, KernelAttentionWrapper), f"Layer {layer_idx} should not have KernelAttentionWrapper"
    
    print("✓ Multiple layers patch test passed")


def test_freeze_layers():
    """Test layer freezing functionality."""
    print("\nRunning test_freeze_layers...")
    
    model = create_tiny_distilbert()
    
    # Patch L4 and L5
    patch_distilbert_attention(model, layers_to_patch=[4, 5])
    
    # Freeze embeddings and L0-L3
    summary = freeze_layers(
        model,
        freeze_embeddings=True,
        freeze_layer_indices=[0, 1, 2, 3]
    )
    
    # Check that embeddings are frozen
    for param in model.embeddings.parameters():
        assert not param.requires_grad, "Embeddings should be frozen"
    
    # Check that L0-L3 are frozen
    for layer_idx in [0, 1, 2, 3]:
        layer = model.transformer.layer[layer_idx]
        for param in layer.parameters():
            assert not param.requires_grad, f"Layer {layer_idx} should be frozen"
    
    # Check that L4 and L5 are trainable
    for layer_idx in [4, 5]:
        layer = model.transformer.layer[layer_idx]
        has_trainable = any(p.requires_grad for p in layer.parameters())
        assert has_trainable, f"Layer {layer_idx} should have trainable parameters"
    
    # Check summary
    assert summary['frozen_params'] > 0, "Should have frozen parameters"
    assert summary['trainable_params'] > 0, "Should have trainable parameters"
    
    print(f"✓ Freeze layers test passed:")
    print(f"  Frozen params: {summary['frozen_params']:,}")
    print(f"  Trainable params: {summary['trainable_params']:,}")


def test_verify_patch():
    """Test the verification function."""
    print("\nRunning test_verify_patch...")
    
    model = create_tiny_distilbert()
    
    # Patch L4 and L5
    patch_distilbert_attention(model, layers_to_patch=[4, 5])
    
    # Correct verification
    report = verify_patch(model, expected_patched_layers=[4, 5])
    assert report['success'], "Verification should succeed"
    assert len(report['issues']) == 0, "Should have no issues"
    
    # Incorrect verification (expect L3 to be patched, but it's not)
    report = verify_patch(model, expected_patched_layers=[3, 4, 5])
    assert not report['success'], "Verification should fail"
    assert len(report['issues']) > 0, "Should have issues"
    
    print("✓ Verify patch test passed")


def test_parameter_counts():
    """Test parameter counting functions."""
    print("\nRunning test_parameter_counts...")
    
    model = create_tiny_distilbert()
    
    # Get initial counts
    initial_summary = get_trainable_params_summary(model)
    assert initial_summary['trainable_params'] == initial_summary['total_params'], "All params should be trainable initially"
    
    # Patch and freeze
    patch_distilbert_attention(model, layers_to_patch=[4, 5])
    freeze_layers(model, freeze_embeddings=True, freeze_layer_indices=[0, 1, 2, 3])
    
    # Get new counts
    final_summary = get_trainable_params_summary(model)
    
    assert final_summary['frozen_params'] > 0, "Should have frozen parameters"
    assert final_summary['trainable_params'] < initial_summary['trainable_params'], "Should have fewer trainable params"
    assert final_summary['total_params'] == initial_summary['total_params'], "Total params should be unchanged"
    
    # Check percentage
    assert 0 < final_summary['trainable_percentage'] < 100, "Trainable percentage should be between 0 and 100"
    
    print(f"✓ Parameter counts test passed:")
    print(f"  Total params: {final_summary['total_params']:,}")
    print(f"  Trainable: {final_summary['trainable_params']:,} ({final_summary['trainable_percentage']:.1f}%)")
    print(f"  Frozen: {final_summary['frozen_params']:,}")


def test_forward_pass():
    """Test that forward pass works after patching."""
    print("\nRunning test_forward_pass...")
    
    model = create_tiny_distilbert()
    
    # Patch L4 and L5
    patch_distilbert_attention(model, layers_to_patch=[4, 5])
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check output shape
    expected_shape = (batch_size, seq_len, model.config.dim)
    assert outputs.last_hidden_state.shape == expected_shape, f"Expected {expected_shape}, got {outputs.last_hidden_state.shape}"
    
    print("✓ Forward pass test passed")


def test_with_sequence_classification():
    """Test patching with DistilBertForSequenceClassification."""
    print("\nRunning test_with_sequence_classification...")
    
    config = DistilBertConfig(
        vocab_size=1000,
        dim=128,
        n_layers=6,
        n_heads=4,
        hidden_dim=512,
        num_labels=2
    )
    model = DistilBertForSequenceClassification(config)
    
    # Patch L4 and L5
    patch_distilbert_attention(model, layers_to_patch=[4, 5])
    
    # Verify
    report = verify_patch(model, expected_patched_layers=[4, 5])
    assert report['success'], "Patch should succeed for sequence classification model"
    
    # Forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check logits shape
    expected_shape = (batch_size, 2)
    assert outputs.logits.shape == expected_shape, f"Expected {expected_shape}, got {outputs.logits.shape}"
    
    print("✓ Sequence classification test passed")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Running DistilBERT Patching Unit Tests")
    print("=" * 60)
    
    test_patch_single_layer()
    test_patch_multiple_layers()
    test_freeze_layers()
    test_verify_patch()
    test_parameter_counts()
    test_forward_pass()
    test_with_sequence_classification()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
