#!/usr/bin/env python3.11
"""
Dual-Layer Meta-Learning Trainer for Deep Tree Echo

This implements a proof-of-concept dual-layer training system where:
- Layer 1: Neural network weights (transformer)
- Layer 2: Inference engine parameters

Both layers co-evolve during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple
import numpy as np

class DeepTreeEchoInferenceEngine(nn.Module):
    """
    Trainable inference engine with learnable parameters
    
    This is Phase 1: Parameterized inference engine
    Future phases can add discrete program structure
    """
    def __init__(self, num_layers=12, num_heads=12, hidden_dim=768):
        super().__init__()
        
        # Trainable inference parameters
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.top_p = nn.Parameter(torch.tensor(0.9))
        self.repetition_penalty = nn.Parameter(torch.tensor(1.0))
        
        # Layer importance weights (which layers to emphasize)
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # Attention head importance weights
        self.head_weights = nn.Parameter(torch.ones(num_heads))
        
        # Post-processing parameters
        self.coherence_threshold = nn.Parameter(torch.tensor(0.5))
        
    def get_inference_params(self) -> Dict[str, float]:
        """
        Get current inference parameters (clamped to valid ranges)
        """
        return {
            'temperature': F.softplus(self.temperature).item() + 0.1,  # Min 0.1
            'top_p': torch.sigmoid(self.top_p).item(),  # 0 to 1
            'repetition_penalty': F.softplus(self.repetition_penalty).item() + 1.0,  # Min 1.0
        }
    
    def get_layer_weights_normalized(self) -> torch.Tensor:
        """
        Get normalized layer importance weights
        """
        return F.softmax(self.layer_weights, dim=0)
    
    def get_head_weights_normalized(self) -> torch.Tensor:
        """
        Get normalized attention head importance weights
        """
        return F.softmax(self.head_weights, dim=0)
    
    def forward(self, model, input_ids, attention_mask=None, max_length=100):
        """
        Execute inference using learned parameters
        
        Args:
            model: The neural network (transformer)
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            
        Returns:
            generated_ids: Generated token IDs
        """
        params = self.get_inference_params()
        
        # Generate using learned parameters
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=params['temperature'],
                top_p=params['top_p'],
                repetition_penalty=params['repetition_penalty'],
                do_sample=True,
                pad_token_id=model.config.eos_token_id,
            )
        
        return generated
    
    def compute_weighted_hidden_states(self, model, input_ids, attention_mask=None):
        """
        Compute weighted combination of hidden states from different layers
        
        This allows the inference engine to emphasize certain layers
        """
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Stack hidden states from all layers
        hidden_states = torch.stack(outputs.hidden_states)  # [num_layers, batch, seq, hidden]
        
        # Apply learned layer weights
        layer_weights = self.get_layer_weights_normalized()
        weighted_hidden = torch.einsum('l,lbsh->bsh', layer_weights, hidden_states)
        
        return weighted_hidden


class DualLayerDataset(Dataset):
    """
    Dataset for dual-layer training
    """
    def __init__(self, data_path: str, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input and output
        input_text = item['input']
        output_text = item['output']
        
        # Combine for language modeling
        full_text = input_text + " " + output_text
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'input_text': input_text,
            'output_text': output_text,
        }


class DualLayerTrainer:
    """
    Trainer for dual-layer meta-learning
    
    Alternates between:
    1. Training neural network weights (Layer 1)
    2. Training inference engine parameters (Layer 2)
    """
    def __init__(
        self,
        model: GPT2LMHeadModel,
        inference_engine: DeepTreeEchoInferenceEngine,
        tokenizer: GPT2Tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.inference_engine = inference_engine.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Separate optimizers for each layer
        self.model_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01
        )
        
        self.engine_optimizer = torch.optim.AdamW(
            inference_engine.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        
        # Training statistics
        self.stats = {
            'model_losses': [],
            'engine_rewards': [],
            'inference_params_history': [],
        }
    
    def train_neural_network(self, batch: Dict, num_steps: int = 1):
        """
        Train Layer 1: Neural network weights
        
        Uses standard language modeling loss
        """
        self.model.train()
        self.inference_engine.eval()
        
        total_loss = 0.0
        
        for _ in range(num_steps):
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids']
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.model_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.model_optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_steps
        self.stats['model_losses'].append(avg_loss)
        
        return avg_loss
    
    def train_inference_engine(self, batch: Dict, num_samples: int = 5):
        """
        Train Layer 2: Inference engine parameters
        
        Uses reinforcement learning (REINFORCE algorithm)
        Reward is based on output quality
        """
        self.model.eval()
        self.inference_engine.train()
        
        # Sample multiple outputs with current inference parameters
        rewards = []
        log_probs = []
        
        for _ in range(num_samples):
            # Generate output using current inference engine
            with torch.no_grad():
                generated = self.inference_engine(
                    self.model,
                    batch['input_ids'],
                    batch['attention_mask'],
                    max_length=batch['input_ids'].size(1) + 50
                )
            
            # Compute reward
            reward = self.compute_reward(generated, batch)
            rewards.append(reward)
        
        # Average reward
        avg_reward = torch.stack(rewards).mean()
        
        # Compute policy gradient loss
        # We want to maximize reward, so minimize negative reward
        # Add entropy bonus to encourage exploration
        params = self.inference_engine.get_inference_params()
        
        # Simple loss: negative reward (we want to maximize reward)
        # In practice, use REINFORCE with baseline
        loss = -avg_reward
        
        # Add regularization to keep parameters reasonable
        reg_loss = 0.01 * (
            (self.inference_engine.temperature - 1.0) ** 2 +
            (self.inference_engine.top_p - 0.9) ** 2
        )
        
        total_loss = loss + reg_loss
        
        # Backward pass
        self.engine_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inference_engine.parameters(), 1.0)
        self.engine_optimizer.step()
        
        self.stats['engine_rewards'].append(avg_reward.item())
        self.stats['inference_params_history'].append(params.copy())
        
        return avg_reward.item()
    
    def compute_reward(self, generated_ids: torch.Tensor, batch: Dict) -> torch.Tensor:
        """
        Compute reward for generated output
        
        Reward is based on:
        1. Perplexity (lower is better)
        2. Length (penalize too short or too long)
        3. Diversity (penalize repetition)
        
        Returns:
            reward: Scalar tensor
        """
        with torch.no_grad():
            # Compute perplexity
            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=(generated_ids != self.tokenizer.pad_token_id).long(),
                labels=generated_ids
            )
            
            perplexity = torch.exp(outputs.loss)
            
            # Perplexity reward (lower is better)
            perplexity_reward = -perplexity / 100.0  # Normalize
            
            # Length reward (penalize extremes)
            gen_length = (generated_ids != self.tokenizer.pad_token_id).sum(dim=1).float()
            target_length = batch['input_ids'].size(1)
            length_penalty = -torch.abs(gen_length - target_length) / target_length
            length_reward = length_penalty.mean()
            
            # Diversity reward (penalize repetition)
            unique_tokens = []
            for seq in generated_ids:
                unique_tokens.append(len(torch.unique(seq)))
            diversity_reward = torch.tensor(unique_tokens).float().mean() / generated_ids.size(1)
            
            # Combined reward
            reward = perplexity_reward + 0.1 * length_reward + 0.1 * diversity_reward
        
        return reward
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        neural_steps_per_batch: int = 1,
        engine_update_freq: int = 10
    ):
        """
        Train one epoch with dual-layer updates
        
        Args:
            train_loader: DataLoader for training data
            neural_steps_per_batch: Number of gradient steps for neural network per batch
            engine_update_freq: Update inference engine every N batches
        """
        epoch_stats = {
            'model_loss': 0.0,
            'engine_reward': 0.0,
            'num_batches': 0,
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Phase 1: Train neural network
            model_loss = self.train_neural_network(batch, neural_steps_per_batch)
            epoch_stats['model_loss'] += model_loss
            
            # Phase 2: Train inference engine (less frequently)
            if batch_idx % engine_update_freq == 0:
                engine_reward = self.train_inference_engine(batch)
                epoch_stats['engine_reward'] += engine_reward
            
            epoch_stats['num_batches'] += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                params = self.inference_engine.get_inference_params()
                print(f"Batch {batch_idx}: "
                      f"Loss={model_loss:.4f}, "
                      f"Temp={params['temperature']:.3f}, "
                      f"TopP={params['top_p']:.3f}")
        
        # Average statistics
        epoch_stats['model_loss'] /= epoch_stats['num_batches']
        epoch_stats['engine_reward'] /= (epoch_stats['num_batches'] // engine_update_freq)
        
        return epoch_stats
    
    def save_checkpoint(self, path: str):
        """
        Save both model and inference engine
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'inference_engine_state_dict': self.inference_engine.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'engine_optimizer_state_dict': self.engine_optimizer.state_dict(),
            'stats': self.stats,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load both model and inference engine
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.inference_engine.load_state_dict(checkpoint['inference_engine_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.engine_optimizer.load_state_dict(checkpoint['engine_optimizer_state_dict'])
        self.stats = checkpoint['stats']
        print(f"Checkpoint loaded from {path}")


def main():
    """
    Main training script
    """
    print("=" * 60)
    print("Dual-Layer Meta-Learning Trainer for Deep Tree Echo")
    print("=" * 60)
    
    # Configuration
    config = {
        'model_name': 'gpt2',  # Start with small model
        'data_path': '/home/ubuntu/training_dataset_5_fixed.jsonl',
        'batch_size': 4,
        'num_epochs': 10,
        'neural_steps_per_batch': 1,
        'engine_update_freq': 10,
        'max_length': 512,
    }
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print("Initializing neural network...")
    model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    
    # Initialize inference engine
    print("Initializing inference engine...")
    inference_engine = DeepTreeEchoInferenceEngine(
        num_layers=model.config.n_layer,
        num_heads=model.config.n_head,
        hidden_dim=model.config.n_embd
    )
    
    # Initialize dataset
    print(f"Loading dataset from {config['data_path']}...")
    dataset = DualLayerDataset(
        config['data_path'],
        tokenizer,
        max_length=config['max_length']
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)} examples")
    
    # Initialize trainer
    print("Initializing dual-layer trainer...")
    trainer = DualLayerTrainer(model, inference_engine, tokenizer)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting dual-layer training")
    print("=" * 60)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 60)
        
        epoch_stats = trainer.train_epoch(
            train_loader,
            neural_steps_per_batch=config['neural_steps_per_batch'],
            engine_update_freq=config['engine_update_freq']
        )
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Model Loss: {epoch_stats['model_loss']:.4f}")
        print(f"  Average Engine Reward: {epoch_stats['engine_reward']:.4f}")
        
        # Print current inference parameters
        params = trainer.inference_engine.get_inference_params()
        print(f"  Current Inference Parameters:")
        print(f"    Temperature: {params['temperature']:.3f}")
        print(f"    Top-p: {params['top_p']:.3f}")
        print(f"    Repetition Penalty: {params['repetition_penalty']:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
    
    # Final save
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    trainer.save_checkpoint('deep_tree_echo_dual_layer_final.pt')
    
    # Print final statistics
    print("\nFinal Inference Parameters:")
    final_params = trainer.inference_engine.get_inference_params()
    for key, value in final_params.items():
        print(f"  {key}: {value:.3f}")
    
    print("\nLayer Importance Weights:")
    layer_weights = trainer.inference_engine.get_layer_weights_normalized()
    for i, weight in enumerate(layer_weights):
        print(f"  Layer {i}: {weight.item():.4f}")


if __name__ == '__main__':
    main()
