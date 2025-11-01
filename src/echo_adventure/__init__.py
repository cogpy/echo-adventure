"""
Echo Adventure: Two-layer neural network with trainable inference engine parameters.

This package implements a novel architecture with:
- Layer 1: Standard transformer components
- Layer 2: Trainable inference engine parameters
- Layer 3: EchoSelf introspection and self-awareness

New in v0.2.0:
- EchoSelf module for self-awareness and introspection
- Hypergraph identity representation
- Agent-Arena-Relation (AAR) geometric architecture
- Conversation-to-hypergraph transformation
- Meta-cognitive reflection capabilities
"""

__version__ = "0.2.0"

from .transformer import TransformerLayer
from .inference_engine import InferenceEngine
from .model import TwoLayerModel

# EchoSelf components (optional import, requires torch)
try:
    from .echoself import (
        EchoSelf,
        HypergraphIdentity,
        IdentityTuple,
        ConversationToHypergraph,
        AARGeometry,
        create_training_examples_from_identity
    )
    __all__ = [
        "TransformerLayer", 
        "InferenceEngine", 
        "TwoLayerModel",
        "EchoSelf",
        "HypergraphIdentity",
        "IdentityTuple",
        "ConversationToHypergraph",
        "AARGeometry",
        "create_training_examples_from_identity"
    ]
except ImportError:
    # Torch not available, only export base components
    __all__ = ["TransformerLayer", "InferenceEngine", "TwoLayerModel"]
