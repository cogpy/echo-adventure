"""
Layer 2: Inference Engine with Trainable Parameters

Implements novel trainable parameters for inference:
- temperature: Controls randomness (learned)
- top_p: Nucleus sampling threshold (learned)
- repetition_penalty: Prevents repetition (learned)
- layer_weights: Which layers to emphasize (learned)
- head_weights: Which attention heads to use (learned)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceEngine(nn.Module):
    """
    Trainable inference engine parameters that control generation behavior.
    
    Unlike traditional fixed hyperparameters, these are learned during training
    to optimize for specific generation objectives.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        init_temperature: float = 1.0,
        init_top_p: float = 0.9,
        init_repetition_penalty: float = 1.0,
    ):
        """
        Initialize inference engine parameters.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            init_temperature: Initial temperature value
            init_top_p: Initial nucleus sampling threshold
            init_repetition_penalty: Initial repetition penalty
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Trainable temperature parameter (controls randomness)
        # Will be constrained to positive values via softplus
        self.temperature_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(init_temperature))
        )
        
        # Trainable top_p parameter (nucleus sampling threshold)
        # Will be constrained to [0, 1] via sigmoid
        self.top_p_raw = nn.Parameter(
            torch.tensor(self._inverse_sigmoid(init_top_p))
        )
        
        # Trainable repetition penalty
        # Will be constrained to positive values via softplus
        self.repetition_penalty_raw = nn.Parameter(
            torch.tensor(self._inverse_softplus(init_repetition_penalty))
        )
        
        # Trainable layer weights (which layers to emphasize)
        # Shape: (num_layers,) - will be normalized via softmax
        self.layer_weights_raw = nn.Parameter(
            torch.ones(num_layers)
        )
        
        # Trainable head weights (which attention heads to use)
        # Shape: (num_layers, num_heads) - will be normalized via softmax
        self.head_weights_raw = nn.Parameter(
            torch.ones(num_layers, num_heads)
        )
    
    @staticmethod
    def _inverse_softplus(x: float) -> float:
        """Inverse of softplus for initialization."""
        return torch.log(torch.exp(torch.tensor(x)) - 1.0).item()
    
    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Inverse of sigmoid for initialization."""
        x = max(0.001, min(0.999, x))  # Clip to valid range
        return torch.log(torch.tensor(x) / (1.0 - x)).item()
    
    @property
    def temperature(self) -> torch.Tensor:
        """
        Get temperature parameter (constrained to be positive).
        
        Returns:
            Temperature value (scalar tensor)
        """
        return F.softplus(self.temperature_raw)
    
    @property
    def top_p(self) -> torch.Tensor:
        """
        Get top_p parameter (constrained to [0, 1]).
        
        Returns:
            Top-p value (scalar tensor)
        """
        return torch.sigmoid(self.top_p_raw)
    
    @property
    def repetition_penalty(self) -> torch.Tensor:
        """
        Get repetition penalty parameter (constrained to be positive).
        
        Returns:
            Repetition penalty value (scalar tensor)
        """
        return F.softplus(self.repetition_penalty_raw)
    
    @property
    def layer_weights(self) -> torch.Tensor:
        """
        Get layer weights (normalized to sum to 1).
        
        Returns:
            Layer weights tensor of shape (num_layers,)
        """
        return F.softmax(self.layer_weights_raw, dim=0)
    
    @property
    def head_weights(self) -> torch.Tensor:
        """
        Get head weights (normalized per layer to sum to 1).
        
        Returns:
            Head weights tensor of shape (num_layers, num_heads)
        """
        return F.softmax(self.head_weights_raw, dim=1)
    
    def apply_layer_weighting(self, layer_outputs: list) -> torch.Tensor:
        """
        Apply learned layer weights to combine multiple layer outputs.
        
        Args:
            layer_outputs: List of tensors from each transformer layer
                          Each tensor has shape (batch, seq_len, d_model)
        
        Returns:
            Weighted combination of layer outputs
        """
        # Stack layer outputs: (num_layers, batch, seq_len, d_model)
        stacked = torch.stack(layer_outputs, dim=0)
        
        # Get normalized weights: (num_layers, 1, 1, 1)
        weights = self.layer_weights.view(-1, 1, 1, 1)
        
        # Weighted sum: (batch, seq_len, d_model)
        weighted_output = (stacked * weights).sum(dim=0)
        
        return weighted_output
    
    def apply_head_weighting(self, attention_weights: list) -> list:
        """
        Apply learned head weights to attention weights.
        
        Args:
            attention_weights: List of attention weight tensors, one per layer
                              Each tensor has shape (batch, num_heads, seq_len, seq_len)
        
        Returns:
            List of reweighted attention tensors
        """
        reweighted = []
        
        for layer_idx, attn in enumerate(attention_weights):
            # Get head weights for this layer: (num_heads,)
            head_w = self.head_weights[layer_idx]
            
            # Reshape for broadcasting: (1, num_heads, 1, 1)
            head_w = head_w.view(1, -1, 1, 1)
            
            # Apply weights to attention
            weighted_attn = attn * head_w
            
            # Renormalize across heads
            weighted_attn = weighted_attn / weighted_attn.sum(dim=1, keepdim=True).clamp(min=1e-9)
            
            reweighted.append(weighted_attn)
        
        return reweighted
    
    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply learned temperature scaling to logits.
        
        Args:
            logits: Logits tensor of shape (batch, seq_len, vocab_size)
        
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def apply_top_p_filtering(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply nucleus (top-p) sampling filter using learned threshold.
        
        Args:
            logits: Logits tensor of shape (batch, vocab_size)
        
        Returns:
            Filtered logits with low-probability tokens masked
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        
        # Shift to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('inf')
        
        return filtered_logits
    
    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply learned repetition penalty to discourage repeating tokens.
        
        Args:
            logits: Logits tensor of shape (batch, vocab_size)
            input_ids: Previously generated token IDs (batch, seq_len)
        
        Returns:
            Logits with repetition penalty applied
        """
        batch_size, vocab_size = logits.shape
        
        # Create penalty mask
        penalty = self.repetition_penalty
        
        for i in range(batch_size):
            for token_id in input_ids[i]:
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty
        
        return logits
    
    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Apply all inference parameters to logits.
        
        Args:
            logits: Logits tensor of shape (batch, vocab_size)
            input_ids: Optional previous tokens for repetition penalty
        
        Returns:
            Processed logits ready for sampling
        """
        # Apply repetition penalty if input_ids provided
        if input_ids is not None:
            logits = self.apply_repetition_penalty(logits, input_ids)
        
        # Apply temperature scaling
        logits = self.apply_temperature_scaling(logits)
        
        # Apply top-p filtering
        logits = self.apply_top_p_filtering(logits)
        
        return logits
    
    def get_params_dict(self) -> dict:
        """
        Get current parameter values as a dictionary.
        
        Returns:
            Dictionary with all inference parameter values
        """
        return {
            'temperature': self.temperature.item(),
            'top_p': self.top_p.item(),
            'repetition_penalty': self.repetition_penalty.item(),
            'layer_weights': self.layer_weights.detach().cpu().numpy().tolist(),
            'head_weights': self.head_weights.detach().cpu().numpy().tolist(),
        }
