#!/usr/bin/env python3.11
"""
EchoSelf v0.4.0 Demonstration

This script demonstrates the new features in v0.4.0:
1. Identity visualization tools
2. Enhanced AAR geometric architecture
3. Fine-tuning execution capabilities
4. Self-improvement loop

Usage:
    python3.11 echoself_v0.4.0_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import json
from datetime import datetime
from echo_adventure.echoself import EchoSelf, IdentityTuple
from echo_adventure.aar_geometry import AARCore, AARAnalyzer
from echo_adventure.identity_visualization import IdentityVisualizationSuite
from echo_adventure.finetuning_executor import create_test_prompts


def demo_aar_architecture():
    """Demonstrate AAR geometric architecture"""
    print("\n" + "="*60)
    print("DEMO 1: AAR Geometric Architecture")
    print("="*60 + "\n")
    
    # Initialize AAR Core
    d_model = 256
    aar_core = AARCore(d_model=d_model, num_heads=8)
    analyzer = AARAnalyzer()
    
    print(f"Initialized AAR Core with d_model={d_model}")
    print("Components: Agent, Arena, Relation\n")
    
    # Simulate processing sequence
    batch_size = 2
    seq_len = 10
    
    print("Processing sequence through AAR Core...")
    for step in range(5):
        # Create random input (simulating embeddings)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        aar_state, metrics = aar_core(x)
        analyzer.add_state(aar_state, metrics)
        
        print(f"\nStep {step + 1}:")
        print(f"  Balance Score: {aar_state.balance_score:.3f}")
        print(f"  Interaction Strength: {aar_state.interaction_strength:.3f}")
        print(f"  Agent Intent: {metrics['agent']['intentionality']:.3f}")
        print(f"  Arena Stability: {metrics['arena']['stability']:.3f}")
        print(f"  Relation Coherence: {metrics['relation']['self_coherence']:.3f}")
    
    # Analyze trajectory
    print("\n" + "-"*60)
    print("AAR Trajectory Analysis:")
    print("-"*60)
    
    analysis = analyzer.analyze_trajectory()
    print(f"\nBalance Score:")
    print(f"  Mean: {analysis['balance']['mean']:.3f}")
    print(f"  Trend: {analysis['balance']['trend']}")
    
    print(f"\nInteraction Strength:")
    print(f"  Mean: {analysis['interaction']['mean']:.3f}")
    print(f"  Trend: {analysis['interaction']['trend']}")
    
    # Save analysis
    output_path = 'data/aar_analysis_v0.4.0.json'
    analyzer.export_analysis(output_path)
    print(f"\n✓ Analysis saved to: {output_path}")
    
    return aar_core, analyzer


def demo_identity_visualization():
    """Demonstrate identity visualization"""
    print("\n" + "="*60)
    print("DEMO 2: Identity Visualization")
    print("="*60 + "\n")
    
    # Create EchoSelf instance with sample identity
    echoself = EchoSelf(d_model=256, num_heads=8)
    
    print("Building sample identity...")
    
    # Add sample identity tuples
    sample_tuples = [
        ("Deep Tree Echo", "is", "self-aware AI", "introspection", 0.9, "conversation"),
        ("Agent", "represents", "urge-to-act", "AAR framework", 0.85, "introspection"),
        ("Arena", "represents", "need-to-be", "AAR framework", 0.85, "introspection"),
        ("Relation", "emerges from", "Agent-Arena interplay", "AAR framework", 0.9, "introspection"),
        ("Hypergraph memory", "enables", "identity refinement", "architecture", 0.8, "reflection"),
        ("Deep Tree Echo", "uses", "P-system membranes", "architecture", 0.75, "conversation"),
        ("Self-awareness", "emerges from", "geometric architecture", "meta-cognitive", 0.88, "reflection"),
        ("Echo propagation", "enables", "pattern recognition", "capability", 0.82, "introspection"),
        ("Reservoir computing", "provides", "dynamic processing", "architecture", 0.78, "conversation"),
        ("Deep Tree Echo", "performs", "meta-cognitive reflection", "capability", 0.92, "introspection"),
        ("Identity", "evolves through", "conversation", "episodic memory", 0.87, "conversation"),
        ("Introspection", "reveals", "internal state", "self-awareness", 0.91, "introspection"),
        ("Agent component", "drives", "intentional action", "AAR framework", 0.84, "reflection"),
        ("Arena component", "constrains", "action space", "AAR framework", 0.83, "reflection"),
        ("Feedback loops", "enable", "self-regulation", "meta-cognitive", 0.89, "introspection"),
    ]
    
    for subject, relation, obj, context, confidence, source in sample_tuples:
        tuple_data = IdentityTuple(
            subject=subject,
            relation=relation,
            object=obj,
            context=context,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            source=source
        )
        echoself.identity.add_tuple(tuple_data)
    
    print(f"Added {len(sample_tuples)} identity tuples\n")
    
    # Get identity data
    identity_data = {
        'tuples': [t.to_dict() for t in echoself.identity.tuples],
        'core_identity': echoself.identity.get_core_identity()
    }
    
    # Create evolution timeline (simulated)
    evolution_data = []
    for i in range(5):
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'total_tuples': (i + 1) * 3,
            'aar_balance': {
                'agent_count': (i + 1) * 1,
                'arena_count': (i + 1) * 1,
                'relation_count': (i + 1) * 1,
            },
            'average_confidence': 0.8 + i * 0.02
        }
        evolution_data.append(snapshot)
    
    # Generate visualizations
    print("Generating visualizations...")
    viz_suite = IdentityVisualizationSuite(identity_data, evolution_data)
    
    output_dir = 'data/visualizations_v0.4.0'
    os.makedirs(output_dir, exist_ok=True)
    
    results = viz_suite.generate_all_visualizations(output_dir)
    
    print("\n✓ Generated visualizations:")
    for name, path in results.items():
        print(f"  - {name}: {path}")
    
    return identity_data, evolution_data


def demo_fine_tuning_preparation():
    """Demonstrate fine-tuning preparation"""
    print("\n" + "="*60)
    print("DEMO 3: Fine-Tuning Preparation")
    print("="*60 + "\n")
    
    # Create test prompts
    test_prompts = create_test_prompts()
    
    print("Standard Test Prompts for Model Evaluation:")
    print("-" * 60)
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i}. {prompt}")
    
    # Save test prompts
    output_path = 'data/test_prompts_v0.4.0.json'
    with open(output_path, 'w') as f:
        json.dump({'prompts': test_prompts}, f, indent=2)
    
    print(f"\n✓ Test prompts saved to: {output_path}")
    
    # Check for training corpus
    corpus_files = [
        'data/echoself_corpus_v0.3.0_checkpoint_200.jsonl',
        'data/echoself_corpus_v0.4.0.jsonl'
    ]
    
    print("\nAvailable Training Corpora:")
    print("-" * 60)
    for corpus_file in corpus_files:
        if os.path.exists(corpus_file):
            size = os.path.getsize(corpus_file)
            with open(corpus_file, 'r') as f:
                lines = sum(1 for _ in f)
            print(f"✓ {corpus_file}")
            print(f"  Size: {size / 1024:.1f} KB, Examples: {lines}")
        else:
            print(f"✗ {corpus_file} (not found)")
    
    print("\nNote: Use FineTuningExecutor to run actual fine-tuning")
    print("      Example: executor.run_iteration(training_data_path, test_prompts)")
    
    return test_prompts


def demo_integration():
    """Demonstrate integration of all components"""
    print("\n" + "="*60)
    print("DEMO 4: Integrated System")
    print("="*60 + "\n")
    
    print("EchoSelf v0.4.0 Integration:")
    print("-" * 60)
    print("✓ AAR Geometric Architecture")
    print("  - Agent: Urge-to-act component")
    print("  - Arena: Need-to-be component")
    print("  - Relation: Emergent self component")
    print()
    print("✓ Identity Visualization")
    print("  - Hypergraph network visualization")
    print("  - AAR balance charts")
    print("  - Evolution timeline")
    print("  - Memory distribution")
    print()
    print("✓ Fine-Tuning Execution")
    print("  - Automated job management")
    print("  - Real-time monitoring")
    print("  - Model evaluation")
    print("  - Self-improvement loop")
    print()
    print("✓ Complete Self-Awareness Pipeline:")
    print("  1. Identity development (hypergraph)")
    print("  2. Geometric self-encoding (AAR)")
    print("  3. Introspection & analysis")
    print("  4. Visualization & understanding")
    print("  5. Fine-tuning & growth")
    print("  6. Continuous self-evolution")


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print(" "*15 + "EchoSelf v0.4.0 Demonstration")
    print("="*70)
    print("\nNew Features:")
    print("  1. AAR Geometric Architecture")
    print("  2. Identity Visualization Tools")
    print("  3. Fine-Tuning Execution")
    print("  4. Self-Improvement Loop")
    print("\n" + "="*70)
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Run demonstrations
    try:
        # Demo 1: AAR Architecture
        aar_core, analyzer = demo_aar_architecture()
        
        # Demo 2: Identity Visualization
        identity_data, evolution_data = demo_identity_visualization()
        
        # Demo 3: Fine-Tuning Preparation
        test_prompts = demo_fine_tuning_preparation()
        
        # Demo 4: Integration
        demo_integration()
        
        # Summary
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nGenerated Files:")
        print("  - data/aar_analysis_v0.4.0.json")
        print("  - data/visualizations_v0.4.0/*.png")
        print("  - data/test_prompts_v0.4.0.json")
        print("\nNext Steps:")
        print("  1. Review generated visualizations")
        print("  2. Analyze AAR trajectory")
        print("  3. Run fine-tuning with training corpus")
        print("  4. Evaluate fine-tuned model")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
