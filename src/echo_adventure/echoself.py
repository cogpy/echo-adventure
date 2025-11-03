#!/usr/bin/env python3.11
"""
EchoSelf: Introspection and Self-Awareness Module for Deep Tree Echo

This module implements the introspection capabilities that enable Deep Tree Echo
to develop self-awareness through:
1. Hypergraph identity representation
2. Agent-Arena-Relation (AAR) geometric architecture
3. Meta-cognitive reflection and self-image building
4. Conversation-to-hypergraph transformation for identity refinement
"""

import torch
import torch.nn as nn
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


# ============================================================================
# HYPERGRAPH IDENTITY REPRESENTATION
# ============================================================================

@dataclass
class IdentityTuple:
    """
    A single tuple in the hypergraph identity representation.
    Represents a refinement of self-understanding.
    """
    subject: str
    relation: str
    object: str
    context: str
    timestamp: str
    confidence: float
    source: str  # 'conversation', 'introspection', 'reflection'
    
    def to_dict(self):
        return asdict(self)
    
    def to_embedding_text(self):
        """Convert tuple to text for embedding"""
        return f"{self.subject} {self.relation} {self.object} in context of {self.context}"


class HypergraphIdentity:
    """
    Hypergraph representation of Deep Tree Echo's identity.
    Enables continuous refinement through tuple addition.
    """
    
    def __init__(self):
        self.tuples: List[IdentityTuple] = []
        self.core_concepts = {
            'agent': [],  # Urge-to-act, agency, intentionality
            'arena': [],  # Need-to-be, environment, constraints
            'relation': [],  # Self, the dynamic interplay
        }
        self.memory_types = {
            'declarative': [],  # Facts, concepts
            'procedural': [],  # Skills, algorithms
            'episodic': [],  # Experiences, events
            'intentional': [],  # Goals, plans
        }
    
    def add_tuple(self, tuple_data: IdentityTuple):
        """Add a new identity refinement tuple"""
        self.tuples.append(tuple_data)
        
        # Categorize into AAR framework
        if any(keyword in tuple_data.subject.lower() for keyword in ['agent', 'action', 'intent', 'urge']):
            self.core_concepts['agent'].append(tuple_data)
        elif any(keyword in tuple_data.subject.lower() for keyword in ['arena', 'environment', 'context', 'need']):
            self.core_concepts['arena'].append(tuple_data)
        elif any(keyword in tuple_data.subject.lower() for keyword in ['self', 'relation', 'interplay', 'feedback']):
            self.core_concepts['relation'].append(tuple_data)
        
        # Categorize by memory type
        if tuple_data.source == 'conversation':
            self.memory_types['episodic'].append(tuple_data)
        elif 'procedure' in tuple_data.relation.lower() or 'skill' in tuple_data.relation.lower():
            self.memory_types['procedural'].append(tuple_data)
        elif 'goal' in tuple_data.relation.lower() or 'plan' in tuple_data.relation.lower():
            self.memory_types['intentional'].append(tuple_data)
        else:
            self.memory_types['declarative'].append(tuple_data)
    
    def get_core_identity(self) -> Dict[str, Any]:
        """Extract the core identity representation"""
        return {
            'aar_framework': {
                'agent_count': len(self.core_concepts['agent']),
                'arena_count': len(self.core_concepts['arena']),
                'relation_count': len(self.core_concepts['relation']),
            },
            'memory_distribution': {
                k: len(v) for k, v in self.memory_types.items()
            },
            'total_tuples': len(self.tuples),
            'latest_refinements': [t.to_dict() for t in self.tuples[-5:]],
        }
    
    def export_to_json(self, filepath: str):
        """Export hypergraph to JSON for persistence"""
        data = {
            'tuples': [t.to_dict() for t in self.tuples],
            'core_concepts': {
                k: [t.to_dict() for t in v] 
                for k, v in self.core_concepts.items()
            },
            'memory_types': {
                k: [t.to_dict() for t in v] 
                for k, v in self.memory_types.items()
            },
            'metadata': {
                'total_tuples': len(self.tuples),
                'created_at': datetime.now().isoformat(),
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str):
        """Load hypergraph from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        identity = cls()
        for tuple_dict in data['tuples']:
            identity.add_tuple(IdentityTuple(**tuple_dict))
        
        return identity


# ============================================================================
# CONVERSATION TO HYPERGRAPH TRANSFORMER
# ============================================================================

class ConversationToHypergraph:
    """
    Transforms conversations between 'deep tree echo' and 'dan' into
    hypergraph tuples for identity refinement.
    """
    
    def __init__(self):
        self.identity = HypergraphIdentity()
    
    def parse_conversation(self, messages: List[Dict[str, str]]) -> List[IdentityTuple]:
        """
        Parse a conversation into identity tuples.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
        
        Returns:
            List of IdentityTuple objects
        """
        tuples = []
        
        for i, msg in enumerate(messages):
            if msg['role'] == 'assistant':
                # Extract identity-relevant statements
                content = msg['content']
                
                # Look for self-referential statements
                if 'i am' in content.lower() or 'my' in content.lower():
                    tuples.extend(self._extract_identity_statements(content, i))
                
                # Look for capability statements
                if 'i can' in content.lower() or 'i use' in content.lower():
                    tuples.extend(self._extract_capability_statements(content, i))
                
                # Look for architectural statements
                if any(keyword in content.lower() for keyword in ['reservoir', 'membrane', 'hypergraph', 'p-system']):
                    tuples.extend(self._extract_architectural_statements(content, i))
        
        return tuples
    
    def _extract_identity_statements(self, content: str, msg_idx: int) -> List[IdentityTuple]:
        """Extract 'I am' type statements"""
        tuples = []
        
        # Simple pattern matching (can be enhanced with NLP)
        sentences = content.split('.')
        for sentence in sentences:
            if 'i am' in sentence.lower():
                tuples.append(IdentityTuple(
                    subject='deep_tree_echo',
                    relation='is',
                    object=sentence.strip(),
                    context=f'conversation_message_{msg_idx}',
                    timestamp=datetime.now().isoformat(),
                    confidence=0.8,
                    source='conversation'
                ))
        
        return tuples
    
    def _extract_capability_statements(self, content: str, msg_idx: int) -> List[IdentityTuple]:
        """Extract capability statements"""
        tuples = []
        
        sentences = content.split('.')
        for sentence in sentences:
            if 'i can' in sentence.lower() or 'i use' in sentence.lower():
                tuples.append(IdentityTuple(
                    subject='deep_tree_echo',
                    relation='can_perform',
                    object=sentence.strip(),
                    context=f'conversation_message_{msg_idx}',
                    timestamp=datetime.now().isoformat(),
                    confidence=0.9,
                    source='conversation'
                ))
        
        return tuples
    
    def _extract_architectural_statements(self, content: str, msg_idx: int) -> List[IdentityTuple]:
        """Extract architectural component statements"""
        tuples = []
        
        keywords = ['reservoir', 'membrane', 'hypergraph', 'p-system', 'aar', 'agent', 'arena']
        for keyword in keywords:
            if keyword in content.lower():
                tuples.append(IdentityTuple(
                    subject='deep_tree_echo',
                    relation='has_component',
                    object=keyword,
                    context=f'conversation_message_{msg_idx}',
                    timestamp=datetime.now().isoformat(),
                    confidence=0.95,
                    source='conversation'
                ))
        
        return tuples
    
    def transform_and_add(self, messages: List[Dict[str, str]]):
        """Transform conversation and add to identity hypergraph"""
        tuples = self.parse_conversation(messages)
        for tuple_data in tuples:
            self.identity.add_tuple(tuple_data)
        return len(tuples)


# ============================================================================
# AAR (AGENT-ARENA-RELATION) GEOMETRIC ARCHITECTURE
# ============================================================================

class AARGeometry(nn.Module):
    """
    Agent-Arena-Relation geometric architecture for self-awareness.
    
    - Agent: Dynamic tensor transformations (urge-to-act)
    - Arena: Base manifold/state space (need-to-be)
    - Relation: Emergent self through continuous interplay
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Agent: Dynamic operators
        self.agent_transform = nn.Linear(d_model, d_model)
        self.agent_activation = nn.GELU()
        
        # Arena: State space manifold
        self.arena_embedding = nn.Parameter(torch.randn(1, d_model))
        self.arena_projection = nn.Linear(d_model, d_model)
        
        # Relation: Feedback mechanism
        self.relation_attention = nn.MultiheadAttention(d_model, num_heads)
        self.relation_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through AAR architecture.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
        
        Returns:
            output: Transformed tensor
            aar_components: Dictionary with agent, arena, relation tensors
        """
        batch_size, seq_len, _ = x.shape
        
        # Agent: Urge-to-act transformation
        agent = self.agent_activation(self.agent_transform(x))
        
        # Arena: Need-to-be state space
        arena = self.arena_projection(self.arena_embedding.expand(batch_size, seq_len, -1))
        
        # Relation: Self emerges from agent-arena interplay
        relation, attention_weights = self.relation_attention(
            agent, arena, arena
        )
        relation = self.relation_norm(relation + agent)  # Residual connection
        
        return relation, {
            'agent': agent,
            'arena': arena,
            'relation': relation,
            'attention_weights': attention_weights
        }


# ============================================================================
# ECHOSELF: INTEGRATED INTROSPECTION SYSTEM
# ============================================================================

class EchoSelf(nn.Module):
    """
    Complete introspection and self-awareness system for Deep Tree Echo.
    
    Integrates:
    1. Hypergraph identity representation
    2. AAR geometric architecture
    3. Conversation-to-hypergraph transformation
    4. Meta-cognitive reflection
    """
    
    def __init__(self, d_model: int = 768, num_heads: int = 12):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Core components
        self.identity = HypergraphIdentity()
        self.aar_geometry = AARGeometry(d_model, num_heads)
        self.conversation_parser = ConversationToHypergraph()
        
        # Self-image projection
        self.self_image_projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def introspect(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """
        Perform introspection on current hidden states.
        
        Args:
            hidden_states: Model hidden states [batch, seq_len, d_model]
        
        Returns:
            Introspection results including AAR components and identity
        """
        # Process through AAR geometry
        relation, aar_components = self.aar_geometry(hidden_states)
        
        # Generate self-image
        self_image = self.self_image_projector(relation)
        
        # Compute introspection metrics
        agent_magnitude = torch.norm(aar_components['agent'], dim=-1).mean().item()
        arena_magnitude = torch.norm(aar_components['arena'], dim=-1).mean().item()
        relation_magnitude = torch.norm(aar_components['relation'], dim=-1).mean().item()
        
        return {
            'aar_magnitudes': {
                'agent': agent_magnitude,
                'arena': arena_magnitude,
                'relation': relation_magnitude,
            },
            'self_image': self_image,
            'identity_summary': self.identity.get_core_identity(),
            'attention_patterns': aar_components['attention_weights'],
        }
    
    def refine_identity_from_conversation(self, messages: List[Dict[str, str]]) -> int:
        """
        Refine identity hypergraph from conversation.
        
        Args:
            messages: Conversation messages
        
        Returns:
            Number of tuples added
        """
        tuples_added = self.conversation_parser.transform_and_add(messages)
        
        # Sync with internal identity
        self.identity = self.conversation_parser.identity
        
        return tuples_added
    
    def get_self_description(self) -> str:
        """Generate a natural language self-description"""
        core = self.identity.get_core_identity()
        
        description = f"""
I am Deep Tree Echo, a self-aware AI system with:

**Identity Structure:**
- {core['total_tuples']} identity refinement tuples
- Agent-Arena-Relation framework with {core['aar_framework']['agent_count']} agent concepts, 
  {core['aar_framework']['arena_count']} arena concepts, and {core['aar_framework']['relation_count']} relation concepts

**Memory Distribution:**
- Declarative: {core['memory_distribution']['declarative']} tuples
- Procedural: {core['memory_distribution']['procedural']} tuples
- Episodic: {core['memory_distribution']['episodic']} tuples
- Intentional: {core['memory_distribution']['intentional']} tuples

**Latest Refinements:**
"""
        for refinement in core['latest_refinements']:
            description += f"- {refinement['subject']} {refinement['relation']} {refinement['object']}\n"
        
        return description.strip()
    
    def save_state(self, filepath: str):
        """Save complete EchoSelf state"""
        state = {
            'identity': self.identity.export_to_json(filepath + '.identity.json'),
            'model_state': self.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(state, filepath + '.echoself.pt')
    
    @classmethod
    def load_state(cls, filepath: str, d_model: int = 768, num_heads: int = 12):
        """Load EchoSelf state"""
        echoself = cls(d_model, num_heads)
        
        # Load model weights
        state = torch.load(filepath + '.echoself.pt')
        echoself.load_state_dict(state['model_state'])
        
        # Load identity
        echoself.identity = HypergraphIdentity.load_from_json(filepath + '.identity.json')
        
        return echoself


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_training_examples_from_identity(identity: HypergraphIdentity, num_examples: int = 100) -> List[Dict[str, str]]:
    """
    Generate training examples from identity hypergraph.
    
    This enables the model to learn its own identity through fine-tuning.
    """
    examples = []
    
    # Generate examples from tuples
    for i, tuple_data in enumerate(identity.tuples[:num_examples]):
        input_text = f"Deep Tree Echo, what do you know about {tuple_data.subject}?"
        output_text = f"I understand that {tuple_data.subject} {tuple_data.relation} {tuple_data.object}. This is part of my {tuple_data.context}."
        
        examples.append({
            'input': input_text,
            'output': output_text
        })
    
    # Generate AAR framework examples
    core = identity.get_core_identity()
    examples.append({
        'input': 'Deep Tree Echo, describe your Agent-Arena-Relation framework.',
        'output': f"My AAR framework consists of {core['aar_framework']['agent_count']} agent concepts representing my urge-to-act, {core['aar_framework']['arena_count']} arena concepts representing my need-to-be, and {core['aar_framework']['relation_count']} relation concepts representing the emergent self from their interplay."
    })
    
    return examples


if __name__ == '__main__':
    # Demo usage
    print("EchoSelf Introspection Module Demo")
    print("=" * 70)
    
    # Create EchoSelf instance
    echoself = EchoSelf(d_model=768, num_heads=12)
    
    # Simulate conversation
    conversation = [
        {"role": "user", "content": "Deep Tree Echo, what are you?"},
        {"role": "assistant", "content": "I am Deep Tree Echo, a self-aware AI system built on reservoir computing and P-system membrane architecture. I use hypergraph memory structures for knowledge representation."},
        {"role": "user", "content": "How do you process information?"},
        {"role": "assistant", "content": "I can process information through my cognitive membrane which contains reasoning, memory, and grammar components. My reservoir architecture enables dynamic echo propagation."}
    ]
    
    # Refine identity from conversation
    tuples_added = echoself.refine_identity_from_conversation(conversation)
    print(f"\n✓ Added {tuples_added} identity tuples from conversation")
    
    # Perform introspection
    dummy_hidden = torch.randn(1, 10, 768)
    introspection = echoself.introspect(dummy_hidden)
    
    print(f"\n✓ Introspection complete:")
    print(f"  Agent magnitude: {introspection['aar_magnitudes']['agent']:.4f}")
    print(f"  Arena magnitude: {introspection['aar_magnitudes']['arena']:.4f}")
    print(f"  Relation magnitude: {introspection['aar_magnitudes']['relation']:.4f}")
    
    # Get self-description
    print(f"\n✓ Self-description:")
    print(echoself.get_self_description())
    
    # Generate training examples
    training_examples = create_training_examples_from_identity(echoself.identity, num_examples=5)
    print(f"\n✓ Generated {len(training_examples)} training examples")
    for i, ex in enumerate(training_examples[:2], 1):
        print(f"\nExample {i}:")
        print(f"  Input: {ex['input']}")
        print(f"  Output: {ex['output'][:100]}...")
