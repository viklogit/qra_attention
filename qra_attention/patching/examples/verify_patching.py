"""
Manual verification script for DistilBERT patching.

This script demonstrates how to use the patching utilities and verifies
that they work correctly with a real DistilBERT model.

Run this script to verify the patching implementation:
    python examples/verify_patching.py
"""

import torch
from transformers import DistilBertForSequenceClassification

from qra_attention.patching import (
    patch_distilbert_attention,
    freeze_layers,
    verify_patch,
    get_trainable_params_summary
)


def main():
    print("=" * 70)
    print("DistilBERT Patching Verification")
    print("=" * 70)
    
    # Load DistilBERT model
    print("\n1. Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    print(f"✓ Loaded model with {model.config.n_layers} layers")
    
    # Get initial parameter summary
    print("\n2. Initial parameter summary:")
    initial_summary = get_trainable_params_summary(model)
    print(f"   Total parameters: {initial_summary['total_params']:,}")
    print(f"   Trainable: {initial_summary['trainable_params']:,} (100%)")
    
    # Patch layers L4 and L5
    print("\n3. Patching layers L4 and L5 with kernel attention...")
    model = patch_distilbert_attention(
        model,
        layers_to_patch=[4, 5],
        kernel_config={
            'num_features': 128,
            'sigma': 1.0,
            'normalize': False
        }
    )
    
    # Verify patch
    print("\n4. Verifying patch...")
    report = verify_patch(model, expected_patched_layers=[4, 5])
    if report['success']:
        print(f"✓ Patch verification successful!")
        print(f"   Patched layers: {report['patched_layers']}")
    else:
        print(f"✗ Patch verification failed!")
        print(f"   Issues: {report['issues']}")
        return
    
    # Freeze layers L0-L3 and embeddings
    print("\n5. Freezing embeddings and layers L0-L3...")
    freeze_summary = freeze_layers(
        model,
        freeze_embeddings=True,
        freeze_layer_indices=[0, 1, 2, 3]
    )
    print(f"   Frozen parameters: {freeze_summary['frozen_params']:,}")
    print(f"   Trainable parameters: {freeze_summary['trainable_params']:,}")
    
    # Get final parameter summary
    print("\n6. Final parameter summary:")
    final_summary = get_trainable_params_summary(model)
    print(f"   Total parameters: {final_summary['total_params']:,}")
    print(f"   Trainable: {final_summary['trainable_params']:,} ({final_summary['trainable_percentage']:.1f}%)")
    print(f"   Frozen: {final_summary['frozen_params']:,} ({100 - final_summary['trainable_percentage']:.1f}%)")
    
    # Test forward pass
    print("\n7. Testing forward pass...")
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"✓ Forward pass successful!")
    print(f"   Output logits shape: {outputs.logits.shape}")
    
    print("\n" + "=" * 70)
    print("All verifications passed! ✓")
    print("=" * 70)
    print("\nThe patched model is ready for training:")
    print("  - Layers L4 and L5 use kernel attention")
    print("  - Layers L0-L3 and embeddings are frozen")
    print(f"  - {final_summary['trainable_percentage']:.1f}% of parameters are trainable")


if __name__ == "__main__":
    main()
