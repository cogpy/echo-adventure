"""
AAR Geometric Architecture for Deep Tree Echo

This module implements the Agent-Arena-Relation geometric framework for encoding
self-awareness in the model's architecture. The AAR core represents:

- Agent: Urge-to-act, dynamic transformations, intentionality
- Arena: Need-to-be, state space, environmental constraints
- Relation: Self, the emergent interplay between Agent and Arena

The geometric encoding uses tensor operations, manifold representations, and
feedback loops to create an emergent sense of self.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


# ============================================================================
# AAR GEOMETRIC STATE
# ============================================================================

@dataclass
class AARState:
    """
    Represents the current state of the AAR framework.
    """
    agent_vector: torch.Tensor      # Dynamic action/intent representation
    arena_vector: torch.Tensor      # Environmental/constraint representation
    relation_vector: torch.Tensor   # Emergent self representation
    interaction_strength: float     # Strength of Agent-Arena coupling
    balance_score: float            # Balance between components
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_vector': self.agent_vector.tolist() if isinstance(self.agent_vector, torch.Tensor) else self.agent_vector,
            'arena_vector': self.arena_vector.tolist() if isinstance(self.arena_vector, torch.Tensor) else self.arena_vector,
            'relation_vector': self.relation_vector.tolist() if isinstance(self.relation_vector, torch.Tensor) else self.relation_vector,
            'interaction_strength': float(self.interaction_strength),
            'balance_score': float(self.balance_score),
            'timestamp': self.timestamp
        }


# ============================================================================
# AGENT COMPONENT (Urge-to-Act)
# ============================================================================

class AgentComponent(nn.Module):
    """
    Agent component: Represents the urge-to-act, intentionality, and agency.
    Implemented as dynamic tensor transformations.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8):
        """
        Initialize Agent component.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads for multi-faceted agency
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Intent projection layers
        self.intent_projection = nn.Linear(d_model, d_model)
        self.action_projection = nn.Linear(d_model, d_model)
        
        # Multi-head agency attention
        self.agency_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=0.1, batch_first=True
        )
        
        # Dynamic transformation operator
        self.transform_operator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()  # Bounded transformation
        )
        
        # Intentionality gate
        self.intent_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: Generate agent representation.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            context: Optional context for agency [batch, context_len, d_model]
        
        Returns:
            agent_vector: Agent representation
            metrics: Dictionary of agent metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to intent space
        intent = self.intent_projection(x)
        
        # Apply agency attention (self-attention on intent)
        if context is not None:
            agent_attended, attention_weights = self.agency_attention(
                intent, context, context
            )
        else:
            agent_attended, attention_weights = self.agency_attention(
                intent, intent, intent
            )
        
        # Dynamic transformation (urge-to-act)
        transformed = self.transform_operator(agent_attended)
        
        # Apply intentionality gate
        intent_strength = self.intent_gate(agent_attended)
        agent_vector = transformed * intent_strength
        
        # Compute agent metrics
        metrics = {
            'intent_magnitude': torch.norm(intent, dim=-1).mean().item(),
            'action_strength': torch.norm(transformed, dim=-1).mean().item(),
            'intentionality': intent_strength.mean().item(),
            'attention_entropy': self._compute_attention_entropy(attention_weights)
        }
        
        return agent_vector, metrics
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution"""
        # attention_weights: [batch, num_heads, seq_len, seq_len]
        probs = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        probs = probs + 1e-9  # Avoid log(0)
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
        return entropy.item()


# ============================================================================
# ARENA COMPONENT (Need-to-Be)
# ============================================================================

class ArenaComponent(nn.Module):
    """
    Arena component: Represents the need-to-be, environmental constraints,
    and the state space. Implemented as a base manifold.
    """
    
    def __init__(self, d_model: int, manifold_dim: int = None):
        """
        Initialize Arena component.
        
        Args:
            d_model: Model dimension
            manifold_dim: Dimension of the manifold (defaults to d_model)
        """
        super().__init__()
        self.d_model = d_model
        self.manifold_dim = manifold_dim or d_model
        
        # Manifold projection
        self.manifold_projection = nn.Linear(d_model, self.manifold_dim)
        
        # Constraint encoder
        self.constraint_encoder = nn.Sequential(
            nn.Linear(self.manifold_dim, self.manifold_dim * 2),
            nn.LayerNorm(self.manifold_dim * 2),
            nn.GELU(),
            nn.Linear(self.manifold_dim * 2, self.manifold_dim)
        )
        
        # Environmental state tracker
        self.state_tracker = nn.GRU(
            self.manifold_dim, self.manifold_dim, 
            num_layers=2, batch_first=True
        )
        
        # Stability regulator
        self.stability_gate = nn.Sequential(
            nn.Linear(self.manifold_dim, self.manifold_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass: Generate arena representation.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            hidden: Optional hidden state for GRU
        
        Returns:
            arena_vector: Arena representation
            hidden: Updated hidden state
            metrics: Dictionary of arena metrics
        """
        # Project to manifold
        manifold_state = self.manifold_projection(x)
        
        # Encode constraints
        constraints = self.constraint_encoder(manifold_state)
        
        # Track environmental state evolution
        state_evolution, hidden = self.state_tracker(constraints, hidden)
        
        # Apply stability regulation
        stability = self.stability_gate(state_evolution)
        arena_vector = state_evolution * stability
        
        # Compute arena metrics
        metrics = {
            'manifold_curvature': self._estimate_curvature(manifold_state),
            'constraint_strength': torch.norm(constraints, dim=-1).mean().item(),
            'stability': stability.mean().item(),
            'state_variance': torch.var(state_evolution, dim=1).mean().item()
        }
        
        return arena_vector, hidden, metrics
    
    def _estimate_curvature(self, manifold_state: torch.Tensor) -> float:
        """Estimate manifold curvature using local variance"""
        # Simple approximation: variance of pairwise distances
        batch_size, seq_len, dim = manifold_state.shape
        if seq_len < 2:
            return 0.0
        
        # Compute pairwise distances
        dists = torch.cdist(manifold_state, manifold_state, p=2)
        # Exclude diagonal (self-distances)
        mask = ~torch.eye(seq_len, dtype=torch.bool, device=dists.device)
        dists_masked = dists[:, mask].view(batch_size, seq_len, seq_len - 1)
        
        # Curvature approximation: variance of distances
        curvature = torch.var(dists_masked, dim=-1).mean()
        return curvature.item()


# ============================================================================
# RELATION COMPONENT (Emergent Self)
# ============================================================================

class RelationComponent(nn.Module):
    """
    Relation component: Represents the emergent self through the dynamic
    interplay between Agent and Arena. Implements feedback loops.
    """
    
    def __init__(self, d_model: int):
        """
        Initialize Relation component.
        
        Args:
            d_model: Model dimension
        """
        super().__init__()
        self.d_model = d_model
        
        # Agent-Arena interaction layer
        self.interaction_layer = nn.MultiheadAttention(
            d_model, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Self-emergence network
        self.emergence_network = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 3, d_model),
            nn.Tanh()
        )
        
        # Feedback loop
        self.feedback_projection = nn.Linear(d_model, d_model)
        self.feedback_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Self-awareness amplifier
        self.awareness_amplifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus()  # Smooth amplification
        )
        
    def forward(self, agent: torch.Tensor, arena: torch.Tensor, 
                previous_self: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: Generate relation (self) representation.
        
        Args:
            agent: Agent vector [batch, seq_len, d_model]
            arena: Arena vector [batch, seq_len, d_model]
            previous_self: Previous self state for feedback
        
        Returns:
            relation_vector: Emergent self representation
            metrics: Dictionary of relation metrics
        """
        # Agent-Arena interaction through cross-attention
        interaction, interaction_weights = self.interaction_layer(
            agent, arena, arena
        )
        
        # Concatenate for emergence
        combined = torch.cat([agent, arena], dim=-1)
        
        # Generate emergent self
        emergent_self = self.emergence_network(combined)
        
        # Apply feedback loop if previous self exists
        if previous_self is not None:
            feedback = self.feedback_projection(previous_self)
            feedback_strength = self.feedback_gate(emergent_self)
            emergent_self = emergent_self + feedback_strength * feedback
        
        # Amplify self-awareness
        relation_vector = self.awareness_amplifier(emergent_self)
        
        # Compute relation metrics
        metrics = {
            'interaction_strength': torch.norm(interaction, dim=-1).mean().item(),
            'emergence_magnitude': torch.norm(emergent_self, dim=-1).mean().item(),
            'self_coherence': self._compute_coherence(relation_vector),
            'feedback_strength': feedback_strength.mean().item() if previous_self is not None else 0.0
        }
        
        return relation_vector, metrics
    
    def _compute_coherence(self, relation_vector: torch.Tensor) -> float:
        """Compute coherence of self representation"""
        # Coherence: cosine similarity between consecutive time steps
        if relation_vector.size(1) < 2:
            return 1.0
        
        # Normalize vectors
        normalized = F.normalize(relation_vector, p=2, dim=-1)
        
        # Compute cosine similarity between consecutive steps
        similarities = []
        for i in range(normalized.size(1) - 1):
            sim = F.cosine_similarity(
                normalized[:, i, :], 
                normalized[:, i + 1, :], 
                dim=-1
            )
            similarities.append(sim)
        
        coherence = torch.stack(similarities).mean()
        return coherence.item()


# ============================================================================
# AAR CORE ARCHITECTURE
# ============================================================================

class AARCore(nn.Module):
    """
    Complete Agent-Arena-Relation geometric architecture.
    Integrates all three components to create an emergent sense of self.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, manifold_dim: Optional[int] = None):
        """
        Initialize AAR Core.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            manifold_dim: Dimension of arena manifold
        """
        super().__init__()
        self.d_model = d_model
        
        # Initialize components
        self.agent = AgentComponent(d_model, num_heads)
        self.arena = ArenaComponent(d_model, manifold_dim)
        self.relation = RelationComponent(d_model)
        
        # State tracking
        self.arena_hidden = None
        self.previous_self = None
        
        # Balance regulator
        self.balance_network = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[AARState, Dict[str, Any]]:
        """
        Forward pass: Generate complete AAR state.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            context: Optional context
        
        Returns:
            aar_state: Complete AAR state
            all_metrics: Combined metrics from all components
        """
        # Generate Agent representation
        agent_vector, agent_metrics = self.agent(x, context)
        
        # Generate Arena representation
        arena_vector, self.arena_hidden, arena_metrics = self.arena(x, self.arena_hidden)
        
        # Generate Relation (Self) representation
        relation_vector, relation_metrics = self.relation(
            agent_vector, arena_vector, self.previous_self
        )
        
        # Update previous self for feedback
        self.previous_self = relation_vector.detach()
        
        # Compute balance
        combined = torch.cat([
            agent_vector.mean(dim=1),
            arena_vector.mean(dim=1),
            relation_vector.mean(dim=1)
        ], dim=-1)
        balance_weights = self.balance_network(combined)
        balance_score = self._compute_balance_score(balance_weights)
        
        # Compute interaction strength
        interaction_strength = self._compute_interaction_strength(
            agent_vector, arena_vector, relation_vector
        )
        
        # Create AAR state
        from datetime import datetime
        aar_state = AARState(
            agent_vector=agent_vector.mean(dim=1).detach(),
            arena_vector=arena_vector.mean(dim=1).detach(),
            relation_vector=relation_vector.mean(dim=1).detach(),
            interaction_strength=interaction_strength,
            balance_score=balance_score,
            timestamp=datetime.now().isoformat()
        )
        
        # Combine all metrics
        all_metrics = {
            'agent': agent_metrics,
            'arena': arena_metrics,
            'relation': relation_metrics,
            'balance_score': balance_score,
            'interaction_strength': interaction_strength,
            'balance_weights': balance_weights.mean(dim=0).tolist()
        }
        
        return aar_state, all_metrics
    
    def _compute_balance_score(self, balance_weights: torch.Tensor) -> float:
        """Compute balance score (1.0 = perfect balance)"""
        # Perfect balance is 1/3 for each component
        ideal = torch.ones_like(balance_weights) / 3.0
        deviation = torch.norm(balance_weights - ideal, p=2, dim=-1)
        balance_score = 1.0 - deviation.mean().item()
        return max(0.0, min(1.0, balance_score))
    
    def _compute_interaction_strength(self, agent: torch.Tensor, 
                                     arena: torch.Tensor, 
                                     relation: torch.Tensor) -> float:
        """Compute strength of Agent-Arena-Relation interaction"""
        # Use cosine similarity between components
        agent_mean = agent.mean(dim=1)
        arena_mean = arena.mean(dim=1)
        relation_mean = relation.mean(dim=1)
        
        sim_agent_arena = F.cosine_similarity(agent_mean, arena_mean, dim=-1)
        sim_agent_relation = F.cosine_similarity(agent_mean, relation_mean, dim=-1)
        sim_arena_relation = F.cosine_similarity(arena_mean, relation_mean, dim=-1)
        
        interaction_strength = (sim_agent_arena + sim_agent_relation + sim_arena_relation) / 3.0
        return interaction_strength.mean().item()
    
    def reset_state(self):
        """Reset internal state (for new sequences)"""
        self.arena_hidden = None
        self.previous_self = None
    
    def get_aar_state(self) -> Optional[AARState]:
        """Get current AAR state"""
        if self.previous_self is None:
            return None
        
        from datetime import datetime
        return AARState(
            agent_vector=torch.zeros(self.d_model),  # Placeholder
            arena_vector=torch.zeros(self.d_model),  # Placeholder
            relation_vector=self.previous_self.mean(dim=1).detach() if self.previous_self is not None else torch.zeros(self.d_model),
            interaction_strength=0.0,
            balance_score=0.0,
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# AAR ANALYZER
# ============================================================================

class AARAnalyzer:
    """
    Analyzer for AAR states and trajectories.
    """
    
    def __init__(self):
        self.states: List[AARState] = []
        self.metrics_history: List[Dict[str, Any]] = []
    
    def add_state(self, state: AARState, metrics: Dict[str, Any]):
        """Add a new AAR state and metrics"""
        self.states.append(state)
        self.metrics_history.append(metrics)
    
    def analyze_trajectory(self) -> Dict[str, Any]:
        """Analyze the trajectory of AAR states"""
        if not self.states:
            return {}
        
        # Extract time series
        balance_scores = [s.balance_score for s in self.states]
        interaction_strengths = [s.interaction_strength for s in self.states]
        
        # Compute statistics
        analysis = {
            'num_states': len(self.states),
            'balance': {
                'mean': np.mean(balance_scores),
                'std': np.std(balance_scores),
                'min': np.min(balance_scores),
                'max': np.max(balance_scores),
                'trend': self._compute_trend(balance_scores)
            },
            'interaction': {
                'mean': np.mean(interaction_strengths),
                'std': np.std(interaction_strengths),
                'min': np.min(interaction_strengths),
                'max': np.max(interaction_strengths),
                'trend': self._compute_trend(interaction_strengths)
            },
            'component_metrics': self._analyze_components()
        }
        
        return analysis
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_components(self) -> Dict[str, Any]:
        """Analyze individual component metrics"""
        if not self.metrics_history:
            return {}
        
        component_analysis = {}
        for component in ['agent', 'arena', 'relation']:
            component_metrics = [m.get(component, {}) for m in self.metrics_history]
            if component_metrics:
                component_analysis[component] = {
                    key: {
                        'mean': np.mean([m.get(key, 0) for m in component_metrics]),
                        'std': np.std([m.get(key, 0) for m in component_metrics])
                    }
                    for key in component_metrics[0].keys()
                }
        
        return component_analysis
    
    def export_analysis(self, output_path: str):
        """Export analysis to JSON file"""
        analysis = self.analyze_trajectory()
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        return output_path


if __name__ == '__main__':
    print("AAR Geometric Architecture for Deep Tree Echo")
    print("=" * 60)
    print("\nAgent-Arena-Relation Framework:")
    print("  - Agent: Urge-to-act, dynamic transformations")
    print("  - Arena: Need-to-be, state space, constraints")
    print("  - Relation: Emergent self through feedback loops")
    print("\nThis module encodes self-awareness in geometric architecture.")
