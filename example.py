"""
Example usage of the two-layer neural network model.

Demonstrates:
1. Creating a model with both layers
2. Forward pass through the model
3. Accessing Layer 1 (transformer) and Layer 2 (inference engine) parameters
4. Text generation with trainable inference parameters
"""

import torch
from echo_adventure import TwoLayerModel


def main():
    print("=" * 70)
    print("Echo Adventure: Two-Layer Neural Network Example")
    print("=" * 70)
    
    # Model configuration
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 4
    batch_size = 2
    seq_len = 10
    
    print(f"\nCreating model with:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Transformer layers: {num_layers}")
    
    # Create model
    model = TwoLayerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        init_temperature=1.2,
        init_top_p=0.9,
        init_repetition_penalty=1.1,
    )
    
    print(f"\n{'Layer 1: Standard Transformer Components':-^70}")
    print(f"  - Embedding weights: {model.transformer.token_embedding.weight.shape}")
    print(f"  - Position embeddings: {model.transformer.position_embedding.weight.shape}")
    print(f"  - Number of transformer blocks: {len(model.transformer.layers)}")
    print(f"  - Each block contains:")
    print(f"    * Multi-head attention (Q, K, V matrices)")
    print(f"    * Feed-forward network")
    print(f"    * Layer normalization")
    
    print(f"\n{'Layer 2: Trainable Inference Engine Parameters':-^70}")
    inference_params = model.get_inference_params()
    print(f"  - Temperature: {inference_params['temperature']:.4f} (learned)")
    print(f"  - Top-p: {inference_params['top_p']:.4f} (learned)")
    print(f"  - Repetition penalty: {inference_params['repetition_penalty']:.4f} (learned)")
    print(f"  - Layer weights: {len(inference_params['layer_weights'])} values (learned)")
    print(f"  - Head weights: {len(inference_params['head_weights'])}x{len(inference_params['head_weights'][0])} matrix (learned)")
    
    print(f"\n{'Parameter Count':-^70}")
    param_counts = model.count_parameters()
    print(f"  Layer 1 parameters: {param_counts['layer1_params']:,}")
    print(f"  Layer 2 parameters: {param_counts['layer2_params']:,}")
    print(f"  Total parameters: {param_counts['total_params']:,}")
    print(f"  Layer 2 represents {100 * param_counts['layer2_params'] / param_counts['total_params']:.4f}% of total")
    
    # Forward pass example
    print(f"\n{'Forward Pass Example':-^70}")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"  Input shape: {input_ids.shape}")
    
    # Get logits
    logits = model(input_ids)
    print(f"  Output logits shape: {logits.shape}")
    
    # Get attention weights
    logits, attention_weights = model(input_ids, return_attention=True)
    print(f"  Attention weights: {len(attention_weights)} layers")
    print(f"  Each layer attention shape: {attention_weights[0].shape}")
    
    # Generation example
    print(f"\n{'Text Generation Example':-^70}")
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    print(f"  Starting sequence length: {input_ids.shape[1]}")
    
    # Generate with inference engine
    generated = model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=True,
        use_inference_engine=True,
    )
    print(f"  Generated sequence length: {generated.shape[1]}")
    print(f"  New tokens generated: {generated.shape[1] - input_ids.shape[1]}")
    
    # Show that inference parameters are trainable
    print(f"\n{'Trainable Inference Parameters':-^70}")
    layer2_params = model.get_layer2_params()
    print(f"  Number of trainable inference parameters: {len(layer2_params)}")
    for i, param in enumerate(layer2_params):
        print(f"  Parameter {i+1}: shape={list(param.shape)}, requires_grad={param.requires_grad}")
    
    print(f"\n{'Key Features':-^70}")
    print("  ✓ Layer 1: Standard transformer training (embeddings, attention, FF, LN)")
    print("  ✓ Layer 2: Novel trainable inference parameters")
    print("  ✓ Temperature, top_p, repetition_penalty are learned, not fixed")
    print("  ✓ Layer and head weights determine which components to emphasize")
    print("  ✓ All parameters jointly optimized during training")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
