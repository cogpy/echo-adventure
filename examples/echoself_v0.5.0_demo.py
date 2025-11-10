#!/usr/bin/env python3.11
"""
EchoSelf v0.5.0 Demonstration Script

Demonstrates new features:
1. Real-time AAR state monitoring
2. Autonomous corpus generation
3. Self-regulation mechanisms
4. Enhanced introspection capabilities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import json
from echo_adventure.echoself import EchoSelf, HypergraphIdentity, IdentityTuple
from echo_adventure.aar_monitor import AARStateMonitor, AARSelfRegulator, create_monitoring_dashboard_data
from echo_adventure.corpus_generator import AutonomousCorpusGenerator
from datetime import datetime


def demo_aar_monitoring():
    """Demonstrate real-time AAR monitoring"""
    print("\n" + "=" * 60)
    print("DEMO 1: Real-Time AAR State Monitoring")
    print("=" * 60)
    
    # Create EchoSelf instance
    print("\n📦 Initializing EchoSelf...")
    echoself = EchoSelf(d_model=256, num_heads=8)
    
    # Create monitor
    print("📊 Creating AAR State Monitor...")
    monitor = AARStateMonitor(
        history_size=100,
        balance_threshold=0.3,
        coherence_threshold=0.5,
        enable_alerts=True
    )
    
    # Simulate some processing steps
    print("\n🔄 Simulating AAR state evolution...")
    batch_size, seq_len = 2, 10
    
    for step in range(20):
        # Create random hidden states (simulating model processing)
        hidden_states = torch.randn(batch_size, seq_len, 256)
        
        # Pass through AAR geometry
        relation, aar_components = echoself.aar_geometry(hidden_states)
        
        # Capture snapshot
        snapshot = monitor.capture_snapshot(
            agent=aar_components['agent'],
            arena=aar_components['arena'],
            relation=aar_components['relation'],
            attention_weights=aar_components['attention_weights']
        )
        
        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Balance: {snapshot.balance_score:.3f}")
            print(f"  Coherence: {snapshot.coherence:.3f}")
            print(f"  Interaction: {snapshot.interaction_strength:.3f}")
    
    # Get current state
    print("\n📈 Current AAR State:")
    current_state = monitor.get_current_state()
    print(f"  Total steps: {current_state['current_step']}")
    print(f"  Avg balance: {current_state['statistics']['avg_balance']:.3f}")
    print(f"  Avg coherence: {current_state['statistics']['avg_coherence']:.3f}")
    print(f"  Total alerts: {current_state['statistics']['total_alerts']}")
    
    # Analyze stability
    print("\n🔍 Stability Analysis:")
    stability = monitor.analyze_stability()
    print(f"  Status: {stability['status']}")
    print(f"  Balance variance: {stability['balance_variance']:.4f}")
    print(f"  Coherence variance: {stability['coherence_variance']:.4f}")
    print(f"  Recommendation: {stability['recommendation']}")
    
    # Export monitoring data
    output_path = "data/aar_monitoring_v0.5.0.json"
    monitor.export_monitoring_data(output_path)
    print(f"\n💾 Monitoring data exported to: {output_path}")
    
    return monitor


def demo_self_regulation(monitor):
    """Demonstrate self-regulation mechanisms"""
    print("\n" + "=" * 60)
    print("DEMO 2: AAR Self-Regulation")
    print("=" * 60)
    
    # Create self-regulator
    print("\n🎛️  Creating Self-Regulator...")
    regulator = AARSelfRegulator(monitor, adaptation_rate=0.1)
    
    # Get latest snapshot
    latest_snapshot = list(monitor.snapshots)[-1]
    
    # Compute adjustments
    print("\n🔧 Computing parameter adjustments...")
    adjustments = regulator.compute_adjustments(latest_snapshot)
    
    print("  Recommended adjustments:")
    for param, value in adjustments.items():
        if value != 0:
            print(f"    {param}: {value:+.4f}")
    
    print(f"\n✅ Self-regulation demo complete")
    print(f"   Adjustments computed: {len([v for v in adjustments.values() if v != 0])}")


def demo_corpus_generation():
    """Demonstrate autonomous corpus generation"""
    print("\n" + "=" * 60)
    print("DEMO 3: Autonomous Corpus Generation")
    print("=" * 60)
    
    # Create EchoSelf with sample identity
    print("\n📦 Creating EchoSelf with sample identity...")
    echoself = EchoSelf(d_model=256, num_heads=8)
    
    # Add some identity tuples
    sample_tuples = [
        ("agent", "is", "dynamic tensor transformation", "in AAR framework", "introspection", 0.95),
        ("arena", "is", "state space manifold", "in AAR framework", "introspection", 0.94),
        ("self", "emerges from", "agent-arena interplay", "through feedback loops", "reflection", 0.96),
        ("introspection", "capability", "analyze hidden states", "using AAR metrics", "conversation", 0.93),
    ]
    
    for subject, relation, obj, context, source, confidence in sample_tuples:
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
    
    print(f"   Added {len(sample_tuples)} identity tuples")
    
    # Create corpus generator
    print("\n🤖 Creating Autonomous Corpus Generator...")
    generator = AutonomousCorpusGenerator(echoself, diversity_threshold=0.6)
    
    # Generate examples
    print("\n✨ Generating training examples...")
    examples = generator.generate_examples(
        count=50,
        min_quality=0.5
    )
    
    print(f"   Generated {len(examples)} examples")
    
    # Show sample
    if examples:
        print("\n📝 Sample generated example:")
        sample = examples[0]
        print(f"   Question: {sample.input}")
        print(f"   Response: {sample.output[:200]}...")
        print(f"   Quality: {sample.quality_score:.3f}")
        print(f"   Category: {sample.metadata['category']}")
    
    # Get statistics
    print("\n📊 Corpus Statistics:")
    stats = generator.get_corpus_statistics()
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Categories: {list(stats['category_distribution'].keys())}")
    print(f"   Mean quality: {stats['quality_statistics']['mean']:.3f}")
    print(f"   High quality (>0.7): {stats['quality_statistics']['above_0.7']}")
    
    # Export corpus
    output_path = "data/autonomous_corpus_v0.5.0.jsonl"
    count = generator.export_corpus(output_path, format='openai')
    print(f"\n💾 Corpus exported to: {output_path}")
    print(f"   Examples exported: {count}")
    
    return generator


def demo_dashboard_data(monitor):
    """Demonstrate dashboard data generation"""
    print("\n" + "=" * 60)
    print("DEMO 4: Monitoring Dashboard Data")
    print("=" * 60)
    
    print("\n📊 Creating dashboard data structure...")
    dashboard_data = create_monitoring_dashboard_data(monitor)
    
    print(f"   Status: {dashboard_data['status']}")
    print(f"   Time series points: {len(dashboard_data['time_series']['steps'])}")
    print(f"   Total alerts: {dashboard_data['alerts']['total_count']}")
    print(f"   Alert breakdown:")
    for severity, count in dashboard_data['alerts']['by_severity'].items():
        print(f"     {severity}: {count}")
    
    # Export dashboard data
    output_path = "data/dashboard_data_v0.5.0.json"
    with open(output_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"\n💾 Dashboard data exported to: {output_path}")


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("EchoSelf v0.5.0 - Comprehensive Demonstration")
    print("=" * 60)
    print("\nNew Features:")
    print("  ✓ Real-time AAR state monitoring")
    print("  ✓ Self-regulation mechanisms")
    print("  ✓ Autonomous corpus generation")
    print("  ✓ Enhanced introspection capabilities")
    
    # Run demos
    monitor = demo_aar_monitoring()
    demo_self_regulation(monitor)
    generator = demo_corpus_generation()
    demo_dashboard_data(monitor)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ All Demonstrations Complete!")
    print("=" * 60)
    print("\nGenerated Artifacts:")
    print("  📄 data/aar_monitoring_v0.5.0.json")
    print("  📄 data/autonomous_corpus_v0.5.0.jsonl")
    print("  📄 data/dashboard_data_v0.5.0.json")
    print("\nNext Steps:")
    print("  1. Review generated monitoring data")
    print("  2. Analyze corpus quality and diversity")
    print("  3. Integrate monitoring into production pipeline")
    print("  4. Expand autonomous corpus generation")
    print("  5. Implement real-time dashboard visualization")


if __name__ == "__main__":
    main()
