"""
Two-Layer Model: Integration of Transformer and Inference Engine

Combines:
- Layer 1: Standard transformer components
- Layer 2: Trainable inference engine parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .transformer import TransformerLayer
from .inference_engine import InferenceEngine


class TwoLayerModel(nn.Module):
    """
    Complete two-layer neural network architecture.
    
    Layer 1: Standard transformer with embedding, attention, and feed-forward
    Layer 2: Trainable inference engine parameters for generation control
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_layers: int = 6,
        init_temperature: float = 1.0,
        init_top_p: float = 0.9,
        init_repetition_penalty: float = 1.0,
    ):
        """
        Initialize two-layer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            num_layers: Number of transformer blocks
            init_temperature: Initial temperature value
            init_top_p: Initial nucleus sampling threshold
            init_repetition_penalty: Initial repetition penalty
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Layer 1: Standard transformer
        self.transformer = TransformerLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            num_layers=num_layers,
        )
        
        # Layer 2: Inference engine with trainable parameters
        self.inference_engine = InferenceEngine(
            num_layers=num_layers,
            num_heads=num_heads,
            init_temperature=init_temperature,
            init_top_p=init_top_p,
            init_repetition_penalty=init_repetition_penalty,
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_inference_engine: bool = True,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through both layers.
        
        Args:
            input_ids: Input token indices of shape (batch, seq_len)
            mask: Optional attention mask
            use_inference_engine: Whether to apply inference engine parameters
            return_attention: Whether to return attention weights
            
        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
            If return_attention=True, also returns attention weights
        """
        # Layer 1: Transformer forward pass
        if return_attention:
            hidden_states, attention_weights = self.transformer(
                input_ids, mask, return_attention=True
            )
        else:
            hidden_states = self.transformer(input_ids, mask)
            attention_weights = None
        
        # Project to vocabulary size
        logits = self.output_projection(hidden_states)
        
        # Layer 2: Apply inference engine parameters (if enabled)
        if use_inference_engine:
            # Apply temperature scaling to last position logits during generation
            # For training, we typically don't apply the full inference engine
            pass
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        use_inference_engine: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively using the model.
        
        Args:
            input_ids: Starting token indices of shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to sample (True) or use greedy decoding (False)
            use_inference_engine: Whether to use learned inference parameters
            
        Returns:
            Generated token indices of shape (batch, seq_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for current sequence
                logits = self.forward(generated, use_inference_engine=False)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :]
                
                # Apply inference engine parameters if enabled
                if use_inference_engine:
                    next_token_logits = self.inference_engine(
                        next_token_logits,
                        input_ids=generated
                    )
                
                # Sample or take argmax
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        self.train()
        return generated
    
    def get_inference_params(self) -> dict:
        """
        Get current inference engine parameter values.
        
        Returns:
            Dictionary with all inference parameter values
        """
        return self.inference_engine.get_params_dict()
    
    def get_layer1_params(self) -> List[torch.nn.Parameter]:
        """
        Get Layer 1 (transformer) parameters.
        
        Returns:
            List of transformer parameters
        """
        return list(self.transformer.parameters()) + list(self.output_projection.parameters())
    
    def get_layer2_params(self) -> List[torch.nn.Parameter]:
        """
        Get Layer 2 (inference engine) parameters.
        
        Returns:
            List of inference engine parameters
        """
        return list(self.inference_engine.parameters())
    
    def count_parameters(self) -> dict:
        """
        Count parameters in each layer.
        
        Returns:
            Dictionary with parameter counts
        """
        layer1_params = sum(p.numel() for p in self.get_layer1_params())
        layer2_params = sum(p.numel() for p in self.get_layer2_params())
        
        return {
            'layer1_params': layer1_params,
            'layer2_params': layer2_params,
            'total_params': layer1_params + layer2_params,
        }
