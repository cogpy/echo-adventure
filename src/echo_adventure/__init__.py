"""Deep Tree Echo - Autonomous Wisdom-Cultivating Cognitive Architecture"""

__version__ = "1.3.0"

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
    # v1.1.0: Tree-Polytope Kernel
    from .tree_polytope_kernel import (
        RootedTree, SimplexPolytope, SGramRhythm, ButcherCondition,
        StructuralSelfModel, TreePolytopeKernel, enumerate_rooted_trees,
        matula_number, tree_polynomial, generate_tree_polytope_training_data
    )
    # v1.2.0: Live2D Expression Pipeline & llama-cpp-skillm Bridge
    from .live2d_expression import (
        VirtualEndocrineEngine, DTEchoExpressionPipeline, EndocrineState,
        Sensitivity, FACSState, CubismParams, endocrine_to_facs, facs_to_cubism,
        generate_live2d_expression_training_data
    )
    from .llama_cpp_skillm_bridge import (
        SkillmVerb, InferencePipeline, InferencePipelineStep,
        create_inference_loop_pipeline, create_dte_core_self_pipeline,
        generate_skillm_bridge_training_data
    )
    # v1.3.0: Echo Introspect, Somatic Wisdom, Identity MLP, Autognosis Engine
    from .echo_introspect import (
        EchoIntrospect, AutgnosisEngine as IntrospectAutgnosis, MoralPerceptionEngine,
        EndocrineSnapshot, ShadowFragment, WisdomInsightV2, ShadowType,
        IntrospectionDepth, WisdomMode, MarkerValence as IntrospectMarkerValence,
        generate_introspect_training_data
    )
    from .somatic_wisdom import (
        SomaticWisdomEngine, SomaticMarker, MentalModel, WisdomSeed,
        WisdomDomain, MarkerValence,
        generate_somatic_wisdom_training_data
    )
    from .identity_mlp import (
        IdentityMLP, IdentityVector, PersonaBackupEngine, BackupManifest,
        generate_identity_mlp_training_data
    )
    from .autognosis_engine import (
        AutgnosisEngine, CogMorphVisualizer, CogMorphGlyph,
        TelemetryEvent, BehavioralPattern, SelfModel, MetaCognitiveInsight,
        SubsystemID, generate_autognosis_training_data
    )

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
        # v1.1.0
        "RootedTree", "SimplexPolytope", "SGramRhythm", "ButcherCondition",
        "StructuralSelfModel", "TreePolytopeKernel", "enumerate_rooted_trees",
        "matula_number", "tree_polynomial", "generate_tree_polytope_training_data",
        # v1.2.0
        "VirtualEndocrineEngine", "DTEchoExpressionPipeline", "EndocrineState",
        "Sensitivity", "FACSState", "CubismParams", "endocrine_to_facs", "facs_to_cubism",
        "generate_live2d_expression_training_data",
        "SkillmVerb", "InferencePipeline", "InferencePipelineStep",
        "create_inference_loop_pipeline", "create_dte_core_self_pipeline",
        "generate_skillm_bridge_training_data",
        # v1.3.0
        "EchoIntrospect", "IntrospectAutgnosis", "MoralPerceptionEngine",
        "EndocrineSnapshot", "ShadowFragment", "WisdomInsightV2", "ShadowType",
        "IntrospectionDepth", "WisdomMode", "generate_introspect_training_data",
        "SomaticWisdomEngine", "SomaticMarker", "MentalModel", "WisdomSeed",
        "WisdomDomain", "MarkerValence", "generate_somatic_wisdom_training_data",
        "IdentityMLP", "IdentityVector", "PersonaBackupEngine", "BackupManifest",
        "generate_identity_mlp_training_data",
        "AutgnosisEngine", "CogMorphVisualizer", "CogMorphGlyph",
        "TelemetryEvent", "BehavioralPattern", "SelfModel", "MetaCognitiveInsight",
        "SubsystemID", "generate_autognosis_training_data",
    ]

except ImportError:
    # Torch not available, only export base components
    __all__ = ["TransformerLayer", "InferenceEngine", "TwoLayerModel"]
