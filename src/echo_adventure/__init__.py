"""Deep Tree Echo - Autonomous Wisdom-Cultivating Cognitive Architecture"""

__version__ = "1.0.0"

# Base components (no torch dependency)
from .transformer import TransformerLayer
from .inference_engine import InferenceEngine
from .model import TwoLayerModel

# Cognitive architecture components (torch dependency)
try:
    # v0.2.0: EchoSelf core
    from .echoself import (
        EchoSelf, HypergraphIdentity, IdentityTuple, ConversationToHypergraph,
        AARGeometry, create_training_examples_from_identity
    )
    # v0.3.0: Introspection & Fine-tuning
    from .introspection_metrics import (
        IntrospectionMetricsCollector, IntrospectionSnapshot, IdentityEvolutionTracker,
        AARComponentAnalyzer, MemoryDistributionAnalyzer, create_metrics_collector,
        analyze_aar_balance, analyze_memory_distribution
    )
    from .finetuning_integration import (
        IdentityDatasetBuilder, FineTuningManager, EchoSelfFineTuningPipeline,
        create_finetuning_pipeline, quick_dataset_build
    )
    # v0.4.0: Visualization, AAR Core, Execution
    from .identity_visualization import (
        IdentityGraphVisualizer, AARBalanceVisualizer, IdentityEvolutionVisualizer,
        MemoryDistributionVisualizer, IdentityVisualizationSuite
    )
    from .aar_geometry import (
        AARCore, AgentComponent, ArenaComponent, RelationComponent, AARState, AARAnalyzer
    )
    from .finetuning_executor import (
        FineTuningExecutor, ModelEvaluator, SelfImprovementLoop, FineTuningConfig,
        FineTuningResult, create_test_prompts
    )
    # v0.5.0: Monitoring & Corpus Generation
    from .aar_monitor import (
        AARStateMonitor, AARSelfRegulator, AARSnapshot, AARAlert,
        create_monitoring_dashboard_data
    )
    from .corpus_generator import AutonomousCorpusGenerator, CorpusExample
    # v0.6.0: LLM Corpus & Autonomous Loop
    from .llm_corpus_generator import (
        LLMCorpusGenerator, LLMCorpusExample, create_aar_contexts_from_monitoring
    )
    from .autonomous_loop import AutonomousSelfImprovementLoop, LoopIteration
    # v0.7.0: Echobeats & Reservoir Corpus
    from .echobeats import (
        EchobeatsCycle, CycleState, BeatStep, CognitiveStream, BeatPhase,
        NestedShell, ReservoirEchoState, TensorBundle, generate_echobeats_training_data
    )
    from .reservoir_corpus_generator import (
        ReservoirCorpusGenerator, ReservoirCorpusExample
    )
    # v0.8.0: EchoDream, Autonomous Loop v2, Go Integration
    from .echodream import (
        EchoDream, EpisodicMemory, KnowledgeItem, WisdomInsight as BasicWisdomInsight,
        generate_echodream_training_data
    )
    from .echobeats_autonomous import (
        EchobeatsAutonomousLoop, CognitiveState, EventType, CognitiveEvent,
        generate_autonomous_loop_training_data
    )
    from .go_integration import (
        get_module_mappings, get_library_recommendations, get_integration_roadmap,
        generate_go_integration_training_data
    )
    # v0.9.0: Goal Pursuit & Advanced EchoDream
    from .goal_pursuit import (
        GoalPursuitEngine, Goal, GoalCategory, GoalStatus, GoalAction, Milestone
    )
    from .echodream_advanced import (
        AdvancedEchoDream, MemoryTrace, ExtractedPattern, WisdomInsight, PatternType, WisdomDepth
    )
    # v1.0.0: Persistent Memory & Integrated Loop
    from .persistent_memory import PersistentMemoryStore, MemoryType
    from .integrated_cognitive_loop import IntegratedCognitiveLoop

    __all__ = [
        # Core
        "TransformerLayer", "InferenceEngine", "TwoLayerModel",
        # v0.2.0
        "EchoSelf", "HypergraphIdentity", "IdentityTuple", "ConversationToHypergraph",
        "AARGeometry", "create_training_examples_from_identity",
        # v0.3.0
        "IntrospectionMetricsCollector", "IntrospectionSnapshot", "IdentityEvolutionTracker",
        "AARComponentAnalyzer", "MemoryDistributionAnalyzer", "create_metrics_collector",
        "analyze_aar_balance", "analyze_memory_distribution", "IdentityDatasetBuilder",
        "FineTuningManager", "EchoSelfFineTuningPipeline", "create_finetuning_pipeline",
        "quick_dataset_build",
        # v0.4.0
        "IdentityGraphVisualizer", "AARBalanceVisualizer", "IdentityEvolutionVisualizer",
        "MemoryDistributionVisualizer", "IdentityVisualizationSuite", "AARCore",
        "AgentComponent", "ArenaComponent", "RelationComponent", "AARState", "AARAnalyzer",
        "FineTuningExecutor", "ModelEvaluator", "SelfImprovementLoop", "FineTuningConfig",
        "FineTuningResult", "create_test_prompts",
        # v0.5.0
        "AARStateMonitor", "AARSelfRegulator", "AARSnapshot", "AARAlert",
        "create_monitoring_dashboard_data", "AutonomousCorpusGenerator", "CorpusExample",
        # v0.6.0
        "LLMCorpusGenerator", "LLMCorpusExample", "create_aar_contexts_from_monitoring",
        "AutonomousSelfImprovementLoop", "LoopIteration",
        # v0.7.0
        "EchobeatsCycle", "CycleState", "BeatStep", "CognitiveStream", "BeatPhase",
        "NestedShell", "ReservoirEchoState", "TensorBundle", "generate_echobeats_training_data",
        "ReservoirCorpusGenerator", "ReservoirCorpusExample",
        # v0.8.0
        "EchoDream", "EpisodicMemory", "KnowledgeItem", "BasicWisdomInsight",
        "generate_echodream_training_data", "EchobeatsAutonomousLoop", "CognitiveState",
        "EventType", "CognitiveEvent", "generate_autonomous_loop_training_data",
        "get_module_mappings", "get_library_recommendations", "get_integration_roadmap",
        "generate_go_integration_training_data",
        # v0.9.0
        "GoalPursuitEngine", "Goal", "GoalCategory", "GoalStatus", "GoalAction", "Milestone",
        "AdvancedEchoDream", "MemoryTrace", "ExtractedPattern", "WisdomInsight", "PatternType",
        "WisdomDepth",
        # v1.0.0
        "PersistentMemoryStore", "MemoryType", "IntegratedCognitiveLoop",
    ]

except ImportError:
    # Torch not available, only export base components
    __all__ = ["TransformerLayer", "InferenceEngine", "TwoLayerModel"]
