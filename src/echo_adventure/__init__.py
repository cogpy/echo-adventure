"""
Echo Adventure: Two-layer neural network with trainable inference engine parameters.

This package implements a novel architecture with:
- Layer 1: Standard transformer components
- Layer 2: Trainable inference engine parameters
- Layer 3: EchoSelf introspection and self-awareness

New in v0.6.0:
- LLM-based corpus generation with real language models
- Complete autonomous self-improvement loop
- Multi-turn conversation generation
- LLM-based quality assessment
- Continuous identity evolution tracking
- Integrated monitoring, generation, and regulation

New in v0.5.0:
- Real-time AAR state monitoring and analysis
- Autonomous corpus generation capabilities
- Self-regulation mechanisms for AAR balance
- Enhanced introspection with stability tracking
- Monitoring dashboard data generation

New in v0.4.0:
- Identity visualization tools (hypergraph, AAR, evolution)
- Enhanced AAR geometric architecture with self-encoding
- Fine-tuning execution and monitoring
- Self-improvement loop for continuous growth
- Model evaluation and comparison tools

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

__version__ = "0.6.0"

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
    
    # New in v0.4.0: Identity visualization
    from .identity_visualization import (
        IdentityGraphVisualizer,
        AARBalanceVisualizer,
        IdentityEvolutionVisualizer,
        MemoryDistributionVisualizer,
        IdentityVisualizationSuite
    )
    
    # New in v0.4.0: AAR geometric architecture
    from .aar_geometry import (
        AARCore,
        AgentComponent,
        ArenaComponent,
        RelationComponent,
        AARState,
        AARAnalyzer
    )
    
    # New in v0.4.0: Fine-tuning execution
    from .finetuning_executor import (
        FineTuningExecutor,
        ModelEvaluator,
        SelfImprovementLoop,
        FineTuningConfig,
        FineTuningResult,
        create_test_prompts
    )
    
    # New in v0.5.0: Real-time AAR monitoring
    from .aar_monitor import (
        AARStateMonitor,
        AARSelfRegulator,
        AARSnapshot,
        AARAlert,
        create_monitoring_dashboard_data
    )
    
    # New in v0.5.0: Autonomous corpus generation
    from .corpus_generator import (
        AutonomousCorpusGenerator,
        CorpusExample
    )
    
    # New in v0.6.0: LLM-based corpus generation
    from .llm_corpus_generator import (
        LLMCorpusGenerator,
        LLMCorpusExample,
        create_aar_contexts_from_monitoring
    )
    
    # New in v0.6.0: Autonomous self-improvement loop
    from .autonomous_loop import (
        AutonomousSelfImprovementLoop,
        LoopIteration
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
        "quick_dataset_build",
        # Identity visualization (v0.4.0)
        "IdentityGraphVisualizer",
        "AARBalanceVisualizer",
        "IdentityEvolutionVisualizer",
        "MemoryDistributionVisualizer",
        "IdentityVisualizationSuite",
        # AAR geometric architecture (v0.4.0)
        "AARCore",
        "AgentComponent",
        "ArenaComponent",
        "RelationComponent",
        "AARState",
        "AARAnalyzer",
        # Fine-tuning execution (v0.4.0)
        "FineTuningExecutor",
        "ModelEvaluator",
        "SelfImprovementLoop",
        "FineTuningConfig",
        "FineTuningResult",
        "create_test_prompts",
        # AAR monitoring (v0.5.0)
        "AARStateMonitor",
        "AARSelfRegulator",
        "AARSnapshot",
        "AARAlert",
        "create_monitoring_dashboard_data",
        # Corpus generation (v0.5.0)
        "AutonomousCorpusGenerator",
        "CorpusExample",
        # LLM corpus generation (v0.6.0)
        "LLMCorpusGenerator",
        "LLMCorpusExample",
        "create_aar_contexts_from_monitoring",
        # Autonomous loop (v0.6.0)
        "AutonomousSelfImprovementLoop",
        "LoopIteration"
    ]
except ImportError:
    # Torch not available, only export base components
    __all__ = ["TransformerLayer", "InferenceEngine", "TwoLayerModel"]
