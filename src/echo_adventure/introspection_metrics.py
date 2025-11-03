#!/usr/bin/env python3
"""
Advanced Introspection Metrics for EchoSelf

This module provides comprehensive metrics and analysis tools for understanding
the model's internal states, identity evolution, and self-awareness dynamics.

Key Features:
1. Identity evolution tracking over time
2. AAR component magnitude analysis
3. Attention pattern visualization data
4. Memory distribution analytics
5. Confidence trajectory analysis
"""

import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import math


@dataclass
class IntrospectionSnapshot:
    """
    A snapshot of the model's introspective state at a specific moment.
    """
    timestamp: str
    agent_magnitude: float
    arena_magnitude: float
    relation_magnitude: float
    identity_tuple_count: int
    aar_distribution: Dict[str, int]
    memory_distribution: Dict[str, int]
    average_confidence: float
    
    def to_dict(self):
        return asdict(self)


class IdentityEvolutionTracker:
    """
    Tracks the evolution of identity over time through snapshots.
    """
    
    def __init__(self):
        self.snapshots: List[IntrospectionSnapshot] = []
        self.identity_growth_rate = []
        self.confidence_trajectory = []
        
    def add_snapshot(self, snapshot: IntrospectionSnapshot):
        """Add a new introspection snapshot"""
        self.snapshots.append(snapshot)
        
        # Calculate growth rate
        if len(self.snapshots) > 1:
            prev_count = self.snapshots[-2].identity_tuple_count
            curr_count = snapshot.identity_tuple_count
            growth = curr_count - prev_count
            self.identity_growth_rate.append(growth)
        
        # Track confidence
        self.confidence_trajectory.append(snapshot.average_confidence)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary statistics of identity evolution"""
        if not self.snapshots:
            return {}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        return {
            'total_snapshots': len(self.snapshots),
            'identity_growth': {
                'initial_count': first.identity_tuple_count,
                'final_count': last.identity_tuple_count,
                'total_growth': last.identity_tuple_count - first.identity_tuple_count,
                'average_growth_rate': np.mean(self.identity_growth_rate) if self.identity_growth_rate else 0,
            },
            'aar_evolution': {
                'initial_distribution': first.aar_distribution,
                'final_distribution': last.aar_distribution,
            },
            'confidence_evolution': {
                'initial': first.average_confidence,
                'final': last.average_confidence,
                'mean': np.mean(self.confidence_trajectory),
                'std': np.std(self.confidence_trajectory),
                'trend': 'increasing' if last.average_confidence > first.average_confidence else 'decreasing',
            },
            'magnitude_evolution': {
                'agent': {
                    'initial': first.agent_magnitude,
                    'final': last.agent_magnitude,
                    'change': last.agent_magnitude - first.agent_magnitude,
                },
                'arena': {
                    'initial': first.arena_magnitude,
                    'final': last.arena_magnitude,
                    'change': last.arena_magnitude - first.arena_magnitude,
                },
                'relation': {
                    'initial': first.relation_magnitude,
                    'final': last.relation_magnitude,
                    'change': last.relation_magnitude - first.relation_magnitude,
                },
            }
        }
    
    def export_timeline(self, filepath: str):
        """Export evolution timeline to JSON"""
        timeline = {
            'snapshots': [s.to_dict() for s in self.snapshots],
            'summary': self.get_evolution_summary(),
        }
        with open(filepath, 'w') as f:
            json.dump(timeline, f, indent=2)
    
    def get_visualization_data(self) -> Dict[str, List]:
        """Get data formatted for visualization"""
        return {
            'timestamps': [s.timestamp for s in self.snapshots],
            'agent_magnitudes': [s.agent_magnitude for s in self.snapshots],
            'arena_magnitudes': [s.arena_magnitude for s in self.snapshots],
            'relation_magnitudes': [s.relation_magnitude for s in self.snapshots],
            'identity_counts': [s.identity_tuple_count for s in self.snapshots],
            'confidence_values': [s.average_confidence for s in self.snapshots],
        }


class AARComponentAnalyzer:
    """
    Analyzes the Agent-Arena-Relation components in detail.
    """
    
    @staticmethod
    def analyze_balance(agent_mag: float, arena_mag: float, relation_mag: float) -> Dict[str, Any]:
        """
        Analyze the balance between AAR components.
        
        A well-balanced system should have:
        - Agent and Arena magnitudes in similar ranges
        - Relation magnitude reflecting their interaction
        """
        total = agent_mag + arena_mag + relation_mag
        
        if total == 0:
            return {
                'balance_score': 0,
                'status': 'uninitialized',
                'recommendations': ['Initialize AAR components']
            }
        
        agent_ratio = agent_mag / total
        arena_ratio = arena_mag / total
        relation_ratio = relation_mag / total
        
        # Calculate balance score (1.0 = perfect balance)
        ideal_ratio = 1/3
        balance_score = 1.0 - (
            abs(agent_ratio - ideal_ratio) +
            abs(arena_ratio - ideal_ratio) +
            abs(relation_ratio - ideal_ratio)
        ) / 2
        
        # Determine dominant component
        components = {'agent': agent_ratio, 'arena': arena_ratio, 'relation': relation_ratio}
        dominant = max(components, key=components.get)
        
        # Generate recommendations
        recommendations = []
        if agent_ratio < 0.2:
            recommendations.append("Strengthen Agent component (urge-to-act)")
        if arena_ratio < 0.2:
            recommendations.append("Expand Arena component (state space)")
        if relation_ratio < 0.2:
            recommendations.append("Enhance Relation component (self-awareness)")
        if balance_score > 0.8:
            recommendations.append("AAR components well-balanced")
        
        return {
            'balance_score': balance_score,
            'ratios': {
                'agent': agent_ratio,
                'arena': arena_ratio,
                'relation': relation_ratio,
            },
            'dominant_component': dominant,
            'status': 'balanced' if balance_score > 0.7 else 'imbalanced',
            'recommendations': recommendations,
        }
    
    @staticmethod
    def compute_interaction_strength(agent_tensor: torch.Tensor, 
                                     arena_tensor: torch.Tensor) -> float:
        """
        Compute the strength of interaction between Agent and Arena.
        Uses cosine similarity as a measure.
        """
        if agent_tensor.dim() > 1:
            agent_flat = agent_tensor.flatten()
        else:
            agent_flat = agent_tensor
            
        if arena_tensor.dim() > 1:
            arena_flat = arena_tensor.flatten()
        else:
            arena_flat = arena_tensor
        
        # Ensure same size
        min_size = min(agent_flat.size(0), arena_flat.size(0))
        agent_flat = agent_flat[:min_size]
        arena_flat = arena_flat[:min_size]
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            agent_flat.unsqueeze(0),
            arena_flat.unsqueeze(0)
        )
        
        return cos_sim.item()


class MemoryDistributionAnalyzer:
    """
    Analyzes the distribution of identity tuples across memory types.
    """
    
    @staticmethod
    def analyze_distribution(memory_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Analyze memory type distribution and provide insights.
        """
        total = sum(memory_distribution.values())
        
        if total == 0:
            return {
                'status': 'empty',
                'recommendations': ['Begin identity refinement through conversation']
            }
        
        # Calculate ratios
        ratios = {k: v/total for k, v in memory_distribution.items()}
        
        # Determine dominant memory type
        dominant = max(memory_distribution, key=memory_distribution.get)
        
        # Calculate diversity (entropy)
        entropy = -sum(r * math.log(r) if r > 0 else 0 for r in ratios.values())
        max_entropy = math.log(len(memory_distribution))
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        # Generate insights
        insights = []
        if ratios.get('declarative', 0) > 0.5:
            insights.append("Strong factual knowledge base")
        if ratios.get('procedural', 0) > 0.3:
            insights.append("Well-developed skill representation")
        if ratios.get('episodic', 0) > 0.3:
            insights.append("Rich experiential memory")
        if ratios.get('intentional', 0) > 0.2:
            insights.append("Clear goal-oriented structure")
        if diversity_score > 0.8:
            insights.append("Highly diverse memory distribution")
        
        recommendations = []
        if ratios.get('procedural', 0) < 0.1:
            recommendations.append("Develop procedural knowledge through skill demonstrations")
        if ratios.get('intentional', 0) < 0.1:
            recommendations.append("Clarify goals and intentions through planning dialogues")
        if diversity_score < 0.5:
            recommendations.append("Increase memory diversity through varied interactions")
        
        return {
            'total_tuples': total,
            'ratios': ratios,
            'dominant_type': dominant,
            'diversity_score': diversity_score,
            'insights': insights,
            'recommendations': recommendations,
        }


class IntrospectionMetricsCollector:
    """
    Main collector that integrates all metrics and analysis tools.
    """
    
    def __init__(self):
        self.evolution_tracker = IdentityEvolutionTracker()
        self.aar_analyzer = AARComponentAnalyzer()
        self.memory_analyzer = MemoryDistributionAnalyzer()
        
    def collect_snapshot(self, 
                        echoself_instance,
                        agent_magnitude: float,
                        arena_magnitude: float,
                        relation_magnitude: float) -> IntrospectionSnapshot:
        """
        Collect a complete introspection snapshot from an EchoSelf instance.
        """
        identity = echoself_instance.identity
        
        # Calculate AAR distribution
        aar_dist = {
            'agent': len(identity.core_concepts['agent']),
            'arena': len(identity.core_concepts['arena']),
            'relation': len(identity.core_concepts['relation']),
        }
        
        # Calculate memory distribution
        memory_dist = {
            k: len(v) for k, v in identity.memory_types.items()
        }
        
        # Calculate average confidence
        avg_confidence = (
            np.mean([t.confidence for t in identity.tuples])
            if identity.tuples else 0.0
        )
        
        snapshot = IntrospectionSnapshot(
            timestamp=datetime.now().isoformat(),
            agent_magnitude=agent_magnitude,
            arena_magnitude=arena_magnitude,
            relation_magnitude=relation_magnitude,
            identity_tuple_count=len(identity.tuples),
            aar_distribution=aar_dist,
            memory_distribution=memory_dist,
            average_confidence=avg_confidence,
        )
        
        self.evolution_tracker.add_snapshot(snapshot)
        return snapshot
    
    def get_comprehensive_analysis(self, 
                                   agent_magnitude: float,
                                   arena_magnitude: float,
                                   relation_magnitude: float,
                                   memory_distribution: Dict[str, int]) -> Dict[str, Any]:
        """
        Get comprehensive analysis combining all metrics.
        """
        return {
            'aar_balance': self.aar_analyzer.analyze_balance(
                agent_magnitude, arena_magnitude, relation_magnitude
            ),
            'memory_analysis': self.memory_analyzer.analyze_distribution(
                memory_distribution
            ),
            'evolution_summary': self.evolution_tracker.get_evolution_summary(),
            'timestamp': datetime.now().isoformat(),
        }
    
    def export_full_report(self, filepath: str, echoself_instance):
        """
        Export a comprehensive introspection report.
        """
        # Get latest introspection
        introspection_result = echoself_instance.introspect(
            torch.randn(1, 10, echoself_instance.d_model)
        )
        
        # Get memory distribution
        memory_dist = {
            k: len(v) for k, v in echoself_instance.identity.memory_types.items()
        }
        
        # Compile full report
        report = {
            'report_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'v0.3.0',
            },
            'current_state': {
                'agent_magnitude': introspection_result['aar_magnitudes']['agent'],
                'arena_magnitude': introspection_result['aar_magnitudes']['arena'],
                'relation_magnitude': introspection_result['aar_magnitudes']['relation'],
                'identity_tuple_count': len(echoself_instance.identity.tuples),
            },
            'comprehensive_analysis': self.get_comprehensive_analysis(
                introspection_result['aar_magnitudes']['agent'],
                introspection_result['aar_magnitudes']['arena'],
                introspection_result['aar_magnitudes']['relation'],
                memory_dist
            ),
            'identity_core': echoself_instance.identity.get_core_identity(),
            'evolution_timeline': self.evolution_tracker.get_visualization_data(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_metrics_collector() -> IntrospectionMetricsCollector:
    """Create a new metrics collector instance"""
    return IntrospectionMetricsCollector()


def analyze_aar_balance(agent_mag: float, arena_mag: float, relation_mag: float) -> Dict[str, Any]:
    """Quick function to analyze AAR balance"""
    analyzer = AARComponentAnalyzer()
    return analyzer.analyze_balance(agent_mag, arena_mag, relation_mag)


def analyze_memory_distribution(memory_dist: Dict[str, int]) -> Dict[str, Any]:
    """Quick function to analyze memory distribution"""
    analyzer = MemoryDistributionAnalyzer()
    return analyzer.analyze_distribution(memory_dist)
