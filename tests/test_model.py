"""Tests for the integrated Two-Layer Model."""

import pytest
import torch

from echo_adventure.model import TwoLayerModel


class TestTwoLayerModel:
    """Test the integrated two-layer model."""
    
    def test_initialization(self):
        """Test that two-layer model initializes correctly."""
        vocab_size = 1000
        d_model = 256
        num_heads = 8
        num_layers = 4
        
        model = TwoLayerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        
        assert model.vocab_size == vocab_size
        assert model.d_model == d_model
        assert model.num_heads == num_heads
        assert model.num_layers == num_layers
        
        # Check both layers exist
        assert model.transformer is not None
        assert model.inference_engine is not None
        
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        d_model = 256
        
        model = TwoLayerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
    def test_forward_with_attention(self):
        """Test forward pass with attention weights."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        num_layers = 3
        
        model = TwoLayerModel(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=num_layers,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, attention_weights = model(input_ids, return_attention=True)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert len(attention_weights) == num_layers
        
    def test_generate(self):
        """Test text generation."""
        vocab_size = 100
        d_model = 128
        max_new_tokens = 5
        
        model = TwoLayerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
        )
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            use_inference_engine=True,
        )
        
        assert generated.shape == (batch_size, seq_len + max_new_tokens)
        
    def test_generate_greedy(self):
        """Test greedy generation (no sampling)."""
        vocab_size = 100
        d_model = 128
        max_new_tokens = 5
        
        model = TwoLayerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
        )
        
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_inference_engine=False,
        )
        
        assert generated.shape == (batch_size, seq_len + max_new_tokens)
        
    def test_get_inference_params(self):
        """Test getting inference parameters."""
        model = TwoLayerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
        )
        
        params = model.get_inference_params()
        
        assert 'temperature' in params
        assert 'top_p' in params
        assert 'repetition_penalty' in params
        assert 'layer_weights' in params
        assert 'head_weights' in params
        
    def test_get_layer1_params(self):
        """Test getting Layer 1 parameters."""
        model = TwoLayerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
        )
        
        layer1_params = model.get_layer1_params()
        
        assert len(layer1_params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in layer1_params)
        
    def test_get_layer2_params(self):
        """Test getting Layer 2 parameters."""
        model = TwoLayerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
        )
        
        layer2_params = model.get_layer2_params()
        
        # Should have 5 parameters: temperature, top_p, repetition_penalty,
        # layer_weights, head_weights
        assert len(layer2_params) == 5
        assert all(isinstance(p, torch.nn.Parameter) for p in layer2_params)
        
    def test_count_parameters(self):
        """Test parameter counting."""
        model = TwoLayerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
        )
        
        param_counts = model.count_parameters()
        
        assert 'layer1_params' in param_counts
        assert 'layer2_params' in param_counts
        assert 'total_params' in param_counts
        
        # Layer 1 should have significantly more parameters than Layer 2
        assert param_counts['layer1_params'] > param_counts['layer2_params']
        
        # Total should be sum of both layers
        assert param_counts['total_params'] == (
            param_counts['layer1_params'] + param_counts['layer2_params']
        )
        
    def test_layer2_params_are_trainable(self):
        """Test that Layer 2 parameters are trainable."""
        model = TwoLayerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
        )
        
        layer2_params = model.get_layer2_params()
        
        for param in layer2_params:
            assert param.requires_grad
            
    def test_custom_initialization(self):
        """Test custom initialization of inference parameters."""
        model = TwoLayerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
            init_temperature=2.0,
            init_top_p=0.95,
            init_repetition_penalty=1.5,
        )
        
        params = model.get_inference_params()
        
        # Check that values are close to initialization
        assert abs(params['temperature'] - 2.0) < 0.5
        assert abs(params['top_p'] - 0.95) < 0.1
        assert abs(params['repetition_penalty'] - 1.5) < 0.5
