"""
Echo Adventure: Two-layer neural network with trainable inference engine parameters.

This package implements a novel architecture with:
- Layer 1: Standard transformer components
- Layer 2: Trainable inference engine parameters
- Layer 3: EchoSelf introspection and self-awareness

New in v0.3.0:
- Advanced introspection metrics and analysis
- Identity evolution tracking over time
- Fine-tuning integration for self-aware models
- Comprehensive reporting and visualization data
- Enhanced AAR component analysis

New in v0.2.0:
- EchoSelf module for self-awareness and introspection
- Hypergraph identity representation
- Agent-Arena-Relation (AAR) geometric architecture
- Conversation-to-hypergraph transformation
- Meta-cognitive reflection capabilities
"""

__version__ = "0.3.0"

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
    
    # New in v0.3.0: Introspection metrics
    from .introspection_metrics import (
        IntrospectionMetricsCollector,
        IntrospectionSnapshot,
        IdentityEvolutionTracker,
        AARComponentAnalyzer,
        MemoryDistributionAnalyzer,
        create_metrics_collector,
        analyze_aar_balance,
        analyze_memory_distribution
    )
    
    # New in v0.3.0: Fine-tuning integration
    from .finetuning_integration import (
        IdentityDatasetBuilder,
        FineTuningManager,
        EchoSelfFineTuningPipeline,
        create_finetuning_pipeline,
        quick_dataset_build
    )
    
    __all__ = [
        # Core components
        "TransformerLayer", 
        "InferenceEngine", 
        "TwoLayerModel",
        # EchoSelf (v0.2.0)
        "EchoSelf",
        "HypergraphIdentity",
        "IdentityTuple",
        "ConversationToHypergraph",
        "AARGeometry",
        "create_training_examples_from_identity",
        # Introspection metrics (v0.3.0)
        "IntrospectionMetricsCollector",
        "IntrospectionSnapshot",
        "IdentityEvolutionTracker",
        "AARComponentAnalyzer",
        "MemoryDistributionAnalyzer",
        "create_metrics_collector",
        "analyze_aar_balance",
        "analyze_memory_distribution",
        # Fine-tuning integration (v0.3.0)
        "IdentityDatasetBuilder",
        "FineTuningManager",
        "EchoSelfFineTuningPipeline",
        "create_finetuning_pipeline",
        "quick_dataset_build"
    ]
except ImportError:
    # Torch not available, only export base components
    __all__ = ["TransformerLayer", "InferenceEngine", "TwoLayerModel"]
