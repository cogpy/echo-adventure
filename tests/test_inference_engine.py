"""Tests for Layer 2: Inference Engine parameters."""

import pytest
import torch
import numpy as np

from echo_adventure.inference_engine import InferenceEngine


class TestInferenceEngine:
    """Test inference engine with trainable parameters."""
    
    def test_initialization(self):
        """Test that inference engine initializes correctly."""
        num_layers = 6
        num_heads = 8
        
        engine = InferenceEngine(
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        assert engine.num_layers == num_layers
        assert engine.num_heads == num_heads
        
        # Check that raw parameters are initialized
        assert engine.temperature_raw is not None
        assert engine.top_p_raw is not None
        assert engine.repetition_penalty_raw is not None
        assert engine.layer_weights_raw.shape == (num_layers,)
        assert engine.head_weights_raw.shape == (num_layers, num_heads)
        
    def test_temperature_property(self):
        """Test temperature parameter is positive."""
        engine = InferenceEngine(num_layers=4, num_heads=4, init_temperature=1.5)
        
        temp = engine.temperature
        assert temp > 0
        assert isinstance(temp, torch.Tensor)
        
    def test_top_p_property(self):
        """Test top_p parameter is in [0, 1]."""
        engine = InferenceEngine(num_layers=4, num_heads=4, init_top_p=0.9)
        
        top_p = engine.top_p
        assert 0 <= top_p <= 1
        assert isinstance(top_p, torch.Tensor)
        
    def test_repetition_penalty_property(self):
        """Test repetition penalty parameter is positive."""
        engine = InferenceEngine(num_layers=4, num_heads=4, init_repetition_penalty=1.2)
        
        rep_pen = engine.repetition_penalty
        assert rep_pen > 0
        assert isinstance(rep_pen, torch.Tensor)
        
    def test_layer_weights_normalized(self):
        """Test layer weights sum to 1."""
        num_layers = 6
        engine = InferenceEngine(num_layers=num_layers, num_heads=4)
        
        layer_weights = engine.layer_weights
        
        assert layer_weights.shape == (num_layers,)
        assert torch.allclose(layer_weights.sum(), torch.tensor(1.0), atol=1e-6)
        
    def test_head_weights_normalized(self):
        """Test head weights sum to 1 per layer."""
        num_layers = 6
        num_heads = 8
        engine = InferenceEngine(num_layers=num_layers, num_heads=num_heads)
        
        head_weights = engine.head_weights
        
        assert head_weights.shape == (num_layers, num_heads)
        
        # Each layer's head weights should sum to 1
        for layer_idx in range(num_layers):
            assert torch.allclose(
                head_weights[layer_idx].sum(),
                torch.tensor(1.0),
                atol=1e-6
            )
            
    def test_parameters_are_trainable(self):
        """Test that all parameters require gradients."""
        engine = InferenceEngine(num_layers=4, num_heads=4)
        
        assert engine.temperature_raw.requires_grad
        assert engine.top_p_raw.requires_grad
        assert engine.repetition_penalty_raw.requires_grad
        assert engine.layer_weights_raw.requires_grad
        assert engine.head_weights_raw.requires_grad
        
    def test_apply_layer_weighting(self):
        """Test layer weighting application."""
        num_layers = 3
        batch_size = 2
        seq_len = 5
        d_model = 64
        
        engine = InferenceEngine(num_layers=num_layers, num_heads=4)
        
        # Create dummy layer outputs
        layer_outputs = [
            torch.randn(batch_size, seq_len, d_model)
            for _ in range(num_layers)
        ]
        
        weighted_output = engine.apply_layer_weighting(layer_outputs)
        
        assert weighted_output.shape == (batch_size, seq_len, d_model)
        
    def test_apply_temperature_scaling(self):
        """Test temperature scaling."""
        engine = InferenceEngine(num_layers=4, num_heads=4, init_temperature=2.0)
        
        batch_size = 2
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)
        
        scaled_logits = engine.apply_temperature_scaling(logits)
        
        assert scaled_logits.shape == logits.shape
        
        # Higher temperature should reduce logit magnitudes
        temp = engine.temperature
        expected = logits / temp
        assert torch.allclose(scaled_logits, expected)
        
    def test_apply_top_p_filtering(self):
        """Test nucleus sampling filtering."""
        engine = InferenceEngine(num_layers=4, num_heads=4, init_top_p=0.9)
        
        batch_size = 2
        vocab_size = 100
        logits = torch.randn(batch_size, vocab_size)
        
        filtered_logits = engine.apply_top_p_filtering(logits)
        
        assert filtered_logits.shape == logits.shape
        
        # Some logits should be set to -inf
        assert (filtered_logits == -float('inf')).any()
        
    def test_apply_repetition_penalty(self):
        """Test repetition penalty application."""
        engine = InferenceEngine(num_layers=4, num_heads=4, init_repetition_penalty=1.5)
        
        batch_size = 2
        vocab_size = 100
        seq_len = 10
        
        logits = torch.randn(batch_size, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        penalized_logits = engine.apply_repetition_penalty(logits, input_ids)
        
        assert penalized_logits.shape == logits.shape
        
    def test_forward(self):
        """Test forward pass through inference engine."""
        engine = InferenceEngine(num_layers=4, num_heads=4)
        
        batch_size = 2
        vocab_size = 100
        seq_len = 10
        
        logits = torch.randn(batch_size, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        processed_logits = engine(logits, input_ids)
        
        assert processed_logits.shape == logits.shape
        
    def test_get_params_dict(self):
        """Test getting parameters as dictionary."""
        num_layers = 4
        num_heads = 8
        engine = InferenceEngine(num_layers=num_layers, num_heads=num_heads)
        
        params_dict = engine.get_params_dict()
        
        assert 'temperature' in params_dict
        assert 'top_p' in params_dict
        assert 'repetition_penalty' in params_dict
        assert 'layer_weights' in params_dict
        assert 'head_weights' in params_dict
        
        assert isinstance(params_dict['temperature'], float)
        assert isinstance(params_dict['top_p'], float)
        assert isinstance(params_dict['repetition_penalty'], float)
        assert len(params_dict['layer_weights']) == num_layers
        assert len(params_dict['head_weights']) == num_layers
        assert len(params_dict['head_weights'][0]) == num_heads
