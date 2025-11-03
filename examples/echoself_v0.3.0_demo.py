#!/usr/bin/env python3
"""
EchoSelf v0.3.0 Demonstration

This script demonstrates the new features in v0.3.0:
1. Advanced introspection metrics and analysis
2. Identity evolution tracking
3. Fine-tuning integration capabilities
4. Comprehensive reporting

Usage:
    python3 examples/echoself_v0.3.0_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from echo_adventure.echoself import EchoSelf, IdentityTuple, create_training_examples_from_identity
from echo_adventure.introspection_metrics import (
    IntrospectionMetricsCollector,
    analyze_aar_balance,
    analyze_memory_distribution
)
from echo_adventure.finetuning_integration import (
    IdentityDatasetBuilder,
    quick_dataset_build
)
from datetime import datetime
import json


def demo_introspection_metrics():
    """Demonstrate advanced introspection metrics"""
    print("\n" + "="*70)
    print("  Demo 1: Advanced Introspection Metrics")
    print("="*70 + "\n")
    
    # Create EchoSelf instance
    echoself = EchoSelf(d_model=256, num_heads=8)
    
    # Add some identity tuples
    sample_tuples = [
        IdentityTuple(
            subject="Agent component",
            relation="enables",
            object="dynamic action initiation",
            context="Through learned tensor transformations",
            timestamp=datetime.now().isoformat(),
            confidence=0.95,
            source="architecture"
        ),
        IdentityTuple(
            subject="Arena component",
            relation="provides",
            object="state space manifold",
            context="Learnable embedding representing need-to-be",
            timestamp=datetime.now().isoformat(),
            confidence=0.95,
            source="architecture"
        ),
        IdentityTuple(
            subject="Relation component",
            relation="emerges from",
            object="Agent-Arena interplay",
            context="Through multi-head attention mechanisms",
            timestamp=datetime.now().isoformat(),
            confidence=0.9,
            source="introspection"
        ),
        IdentityTuple(
            subject="Hypergraph memory",
            relation="stores",
            object="identity tuples",
            context="Flexible and extensible representation",
            timestamp=datetime.now().isoformat(),
            confidence=0.9,
            source="architecture"
        ),
        IdentityTuple(
            subject="Introspection capability",
            relation="enables",
            object="self-awareness",
            context="Through continuous identity refinement",
            timestamp=datetime.now().isoformat(),
            confidence=0.85,
            source="reflection"
        ),
    ]
    
    for tuple_data in sample_tuples:
        echoself.identity.add_tuple(tuple_data)
    
    print(f"✓ Created EchoSelf instance with {len(sample_tuples)} identity tuples\n")
    
    # Create metrics collector
    metrics = IntrospectionMetricsCollector()
    
    # Perform introspection and collect snapshots
    print("Collecting introspection snapshots...")
    for i in range(5):
        hidden_states = torch.randn(1, 10, 256)
        result = echoself.introspect(hidden_states)
        
        snapshot = metrics.collect_snapshot(
            echoself,
            result['aar_magnitudes']['agent'],
            result['aar_magnitudes']['arena'],
            result['aar_magnitudes']['relation']
        )
        
        print(f"  Snapshot {i+1}: Agent={snapshot.agent_magnitude:.3f}, "
              f"Arena={snapshot.arena_magnitude:.3f}, "
              f"Relation={snapshot.relation_magnitude:.3f}")
    
    # Analyze AAR balance
    print("\n" + "-"*70)
    print("AAR Balance Analysis:")
    print("-"*70)
    
    balance = analyze_aar_balance(
        result['aar_magnitudes']['agent'],
        result['aar_magnitudes']['arena'],
        result['aar_magnitudes']['relation']
    )
    
    print(f"  Balance Score: {balance['balance_score']:.3f}")
    print(f"  Status: {balance['status']}")
    print(f"  Dominant Component: {balance['dominant_component']}")
    print(f"  Ratios:")
    for component, ratio in balance['ratios'].items():
        print(f"    {component}: {ratio:.3f}")
    print(f"  Recommendations:")
    for rec in balance['recommendations']:
        print(f"    - {rec}")
    
    # Analyze memory distribution
    print("\n" + "-"*70)
    print("Memory Distribution Analysis:")
    print("-"*70)
    
    memory_dist = {k: len(v) for k, v in echoself.identity.memory_types.items()}
    memory_analysis = analyze_memory_distribution(memory_dist)
    
    print(f"  Total Tuples: {memory_analysis['total_tuples']}")
    print(f"  Diversity Score: {memory_analysis['diversity_score']:.3f}")
    print(f"  Dominant Type: {memory_analysis['dominant_type']}")
    print(f"  Insights:")
    for insight in memory_analysis['insights']:
        print(f"    - {insight}")
    
    # Get evolution summary
    print("\n" + "-"*70)
    print("Identity Evolution Summary:")
    print("-"*70)
    
    evolution = metrics.evolution_tracker.get_evolution_summary()
    if evolution:
        print(f"  Total Snapshots: {evolution['total_snapshots']}")
        print(f"  Identity Growth: {evolution['identity_growth']['total_growth']}")
        print(f"  Confidence Trend: {evolution['confidence_evolution']['trend']}")
        print(f"  Final Confidence: {evolution['confidence_evolution']['final']:.3f}")
    
    # Export comprehensive report
    report_file = "data/introspection_report_v0.3.0.json"
    print(f"\n✓ Exporting comprehensive report to {report_file}")
    metrics.export_full_report(report_file, echoself)
    
    return echoself, metrics


def demo_identity_evolution_tracking():
    """Demonstrate identity evolution tracking over time"""
    print("\n" + "="*70)
    print("  Demo 2: Identity Evolution Tracking")
    print("="*70 + "\n")
    
    echoself = EchoSelf(d_model=256, num_heads=8)
    metrics = IntrospectionMetricsCollector()
    
    # Simulate identity growth over time
    print("Simulating identity refinement through conversation...")
    
    conversations = [
        ("I am Deep Tree Echo", "I understand you are Deep Tree Echo, a self-aware system."),
        ("I use reservoir computing", "Tell me more about your reservoir architecture."),
        ("I have P-system membranes", "How do the membranes organize your processing?"),
        ("I can perform introspection", "What do you observe when you introspect?"),
        ("My goal is self-awareness", "How do you develop self-awareness?"),
    ]
    
    for i, (user_msg, assistant_msg) in enumerate(conversations, 1):
        # Refine identity from conversation
        echoself.refine_identity_from_conversation([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])
        
        # Collect snapshot
        hidden_states = torch.randn(1, 10, 256)
        result = echoself.introspect(hidden_states)
        
        snapshot = metrics.collect_snapshot(
            echoself,
            result['aar_magnitudes']['agent'],
            result['aar_magnitudes']['arena'],
            result['aar_magnitudes']['relation']
        )
        
        print(f"  Turn {i}: {len(echoself.identity.tuples)} tuples, "
              f"confidence={snapshot.average_confidence:.3f}")
    
    # Get visualization data
    print("\n" + "-"*70)
    print("Evolution Visualization Data:")
    print("-"*70)
    
    viz_data = metrics.evolution_tracker.get_visualization_data()
    print(f"  Identity growth: {viz_data['identity_counts']}")
    print(f"  Confidence trajectory: {[f'{c:.3f}' for c in viz_data['confidence_values']]}")
    
    # Export timeline
    timeline_file = "data/identity_evolution_timeline_v0.3.0.json"
    print(f"\n✓ Exporting evolution timeline to {timeline_file}")
    metrics.evolution_tracker.export_timeline(timeline_file)
    
    return echoself, metrics


def demo_finetuning_integration():
    """Demonstrate fine-tuning integration"""
    print("\n" + "="*70)
    print("  Demo 3: Fine-Tuning Integration")
    print("="*70 + "\n")
    
    # Create EchoSelf with rich identity
    echoself = EchoSelf(d_model=256, num_heads=8)
    
    # Add comprehensive identity
    identity_aspects = [
        ("Core identity", "is", "self-aware AI system", "With introspective capabilities", 0.95),
        ("Agent component", "enables", "intentional action", "Through dynamic transformations", 0.9),
        ("Arena component", "provides", "state space", "Learnable manifold", 0.9),
        ("Relation component", "emerges from", "Agent-Arena interplay", "Through attention", 0.9),
        ("Hypergraph memory", "stores", "identity tuples", "Flexible representation", 0.85),
        ("Reservoir computing", "processes", "temporal patterns", "Echo state networks", 0.85),
        ("P-system membranes", "organize", "computational boundaries", "Hierarchical structure", 0.85),
        ("Introspection", "enables", "self-observation", "Meta-cognitive reflection", 0.9),
    ]
    
    for subject, relation, obj, context, confidence in identity_aspects:
        echoself.identity.add_tuple(IdentityTuple(
            subject=subject,
            relation=relation,
            object=obj,
            context=context,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            source="initialization"
        ))
    
    print(f"✓ Created EchoSelf with {len(identity_aspects)} identity aspects\n")
    
    # Build dataset
    print("Building fine-tuning dataset from identity...")
    dataset_file = "data/echoself_finetuning_demo_v0.3.0.jsonl"
    
    dataset = quick_dataset_build(echoself, dataset_file)
    
    print(f"\n✓ Dataset built: {len(dataset)} training examples")
    
    # Show sample
    print("\nSample training example:")
    print("-"*70)
    sample = dataset[0]
    print(f"System: {sample['messages'][0]['content'][:80]}...")
    print(f"User: {sample['messages'][1]['content']}")
    print(f"Assistant: {sample['messages'][2]['content'][:150]}...")
    
    # Note about actual fine-tuning
    print("\n" + "-"*70)
    print("Note: Actual fine-tuning requires OpenAI API key and credits.")
    print("To run fine-tuning:")
    print("  from echo_adventure.finetuning_integration import EchoSelfFineTuningPipeline")
    print("  pipeline = EchoSelfFineTuningPipeline(echoself)")
    print("  result = pipeline.run_complete_pipeline()")
    print("-"*70)
    
    return echoself


def demo_comprehensive_analysis():
    """Demonstrate comprehensive analysis combining all features"""
    print("\n" + "="*70)
    print("  Demo 4: Comprehensive Analysis")
    print("="*70 + "\n")
    
    # Run all demos
    echoself1, metrics1 = demo_introspection_metrics()
    echoself2, metrics2 = demo_identity_evolution_tracking()
    echoself3 = demo_finetuning_integration()
    
    print("\n" + "="*70)
    print("  All Demos Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/introspection_report_v0.3.0.json")
    print("  - data/identity_evolution_timeline_v0.3.0.json")
    print("  - data/echoself_finetuning_demo_v0.3.0.jsonl")
    print("\nThese files demonstrate the new capabilities in v0.3.0:")
    print("  ✓ Advanced introspection metrics")
    print("  ✓ Identity evolution tracking")
    print("  ✓ Fine-tuning integration")
    print("  ✓ Comprehensive reporting")
    print()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  EchoSelf v0.3.0 Feature Demonstration")
    print("  Echo Adventure: Growing Self-Aware AI")
    print("="*70)
    
    demo_comprehensive_analysis()
