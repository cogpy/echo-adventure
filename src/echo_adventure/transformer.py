"""
Layer 1: Standard Transformer Components

Implements standard transformer training components:
- Embedding weights
- Attention matrices (Q, K, V)
- Feed-forward layers
- Layer normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with Q, K, V matrices."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V projection matrices
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, num_heads, seq_len, head_dim)
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """Feed-forward network with two linear layers and activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of the feed-forward layer
            dropout: Dropout rate
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    """
    Complete transformer layer with:
    - Embeddings
    - Multi-head attention
    - Feed-forward network
    - Layer normalization
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_layers: int = 6
    ):
        """
        Initialize transformer layer.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            num_layers: Number of transformer blocks
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embedding weights
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(d_model, num_heads, dropout),
                'feed_forward': FeedForward(d_model, d_ff, dropout),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout1': nn.Dropout(dropout),
                'dropout2': nn.Dropout(dropout),
            })
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Forward pass of transformer.
        
        Args:
            x: Input token indices of shape (batch, seq_len)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
            If return_attention=True, also returns list of attention weights
        """
        batch_size, seq_len = x.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        attention_weights_list = []
        
        # Apply transformer blocks
        for layer in self.layers:
            # Multi-head attention with residual connection
            attn_out, attn_weights = layer['attention'](x, x, x, mask)
            x = layer['norm1'](x + layer['dropout1'](attn_out))
            
            # Feed-forward with residual connection
            ff_out = layer['feed_forward'](x)
            x = layer['norm2'](x + layer['dropout2'](ff_out))
            
            if return_attention:
                attention_weights_list.append(attn_weights)
        
        if return_attention:
            return x, attention_weights_list
        return x
