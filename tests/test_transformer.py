"""Tests for Layer 1: Transformer components."""

import pytest
import torch
import torch.nn as nn

from echo_adventure.transformer import (
    MultiHeadAttention,
    FeedForward,
    TransformerLayer,
)


class TestMultiHeadAttention:
    """Test multi-head attention mechanism."""
    
    def test_initialization(self):
        """Test that multi-head attention initializes correctly."""
        d_model = 512
        num_heads = 8
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        assert mha.d_model == d_model
        assert mha.num_heads == num_heads
        assert mha.head_dim == d_model // num_heads
        
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, attention_weights = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        batch_size = 2
        seq_len = 10
        d_model = 64
        num_heads = 4
        
        mha = MultiHeadAttention(d_model, num_heads)
        mha.eval()  # Use eval mode to disable dropout
        
        x = torch.randn(batch_size, seq_len, d_model)
        _, attention_weights = mha(x, x, x)
        
        # Sum over last dimension (attention over keys)
        sums = attention_weights.sum(dim=-1)
        
        # Should be close to 1
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestFeedForward:
    """Test feed-forward network."""
    
    def test_initialization(self):
        """Test that feed-forward network initializes correctly."""
        d_model = 512
        d_ff = 2048
        
        ff = FeedForward(d_model, d_ff)
        
        assert isinstance(ff.linear1, nn.Linear)
        assert isinstance(ff.linear2, nn.Linear)
        assert ff.linear1.out_features == d_ff
        assert ff.linear2.out_features == d_model
        
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048
        
        ff = FeedForward(d_model, d_ff)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = ff(x)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerLayer:
    """Test complete transformer layer."""
    
    def test_initialization(self):
        """Test that transformer layer initializes correctly."""
        vocab_size = 1000
        d_model = 512
        num_heads = 8
        num_layers = 6
        
        transformer = TransformerLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        
        assert transformer.d_model == d_model
        assert transformer.num_layers == num_layers
        assert transformer.num_heads == num_heads
        assert len(transformer.layers) == num_layers
        
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        d_model = 256
        
        transformer = TransformerLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = transformer(input_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)
        
    def test_return_attention(self):
        """Test that attention weights are returned when requested."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        num_layers = 3
        num_heads = 4
        
        transformer = TransformerLayer(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output, attention_weights = transformer(input_ids, return_attention=True)
        
        assert len(attention_weights) == num_layers
        for attn in attention_weights:
            assert attn.shape == (batch_size, num_heads, seq_len, seq_len)
            
    def test_embeddings(self):
        """Test that embeddings are created correctly."""
        vocab_size = 1000
        d_model = 128
        max_seq_len = 512
        
        transformer = TransformerLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
        )
        
        assert transformer.token_embedding.num_embeddings == vocab_size
        assert transformer.token_embedding.embedding_dim == d_model
        assert transformer.position_embedding.num_embeddings == max_seq_len
        assert transformer.position_embedding.embedding_dim == d_model
