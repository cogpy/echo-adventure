"""
Identity Visualization Module for EchoSelf

This module provides visualization tools for the hypergraph identity representation,
enabling visual analysis of identity evolution, AAR balance, and memory distribution.

Features:
1. Hypergraph network visualization
2. AAR component balance visualization
3. Identity evolution timeline
4. Memory distribution charts
5. Interactive identity exploration
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import networkx as nx


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    'colors': {
        'agent': '#FF6B6B',      # Red - Action, urge
        'arena': '#4ECDC4',      # Teal - Environment, constraints
        'relation': '#FFE66D',   # Yellow - Self, interplay
        'declarative': '#95E1D3', # Light teal
        'procedural': '#F38181',  # Light red
        'episodic': '#AA96DA',    # Purple
        'intentional': '#FCBAD3', # Pink
    },
    'sizes': {
        'node_base': 300,
        'node_scale': 100,
        'edge_base': 1.0,
        'edge_scale': 2.0,
    },
    'layout': {
        'k': 2.0,  # Optimal distance between nodes
        'iterations': 50,
        'seed': 42,
    }
}


# ============================================================================
# HYPERGRAPH NETWORK VISUALIZATION
# ============================================================================

class IdentityGraphVisualizer:
    """
    Visualizes the hypergraph identity as a network graph.
    Nodes represent concepts, edges represent relationships.
    """
    
    def __init__(self, identity_data: Dict[str, Any]):
        """
        Initialize visualizer with identity data.
        
        Args:
            identity_data: Dictionary containing identity tuples and metadata
        """
        self.identity_data = identity_data
        self.graph = nx.MultiDiGraph()
        self.node_categories = {}
        self.edge_weights = {}
        
    def build_graph(self):
        """Build NetworkX graph from identity tuples"""
        tuples = self.identity_data.get('tuples', [])
        
        for idx, tuple_data in enumerate(tuples):
            subject = tuple_data['subject']
            obj = tuple_data['object']
            relation = tuple_data['relation']
            confidence = tuple_data.get('confidence', 0.5)
            source = tuple_data.get('source', 'unknown')
            
            # Add nodes
            if subject not in self.graph:
                self.graph.add_node(subject)
                self.node_categories[subject] = self._categorize_node(subject, tuple_data)
                
            if obj not in self.graph:
                self.graph.add_node(obj)
                self.node_categories[obj] = self._categorize_node(obj, tuple_data)
            
            # Add edge with metadata
            self.graph.add_edge(
                subject, obj,
                relation=relation,
                confidence=confidence,
                source=source,
                tuple_idx=idx
            )
            
            # Track edge weights for visualization
            edge_key = (subject, obj)
            if edge_key not in self.edge_weights:
                self.edge_weights[edge_key] = []
            self.edge_weights[edge_key].append(confidence)
    
    def _categorize_node(self, node: str, tuple_data: Dict) -> str:
        """Categorize node into AAR framework"""
        node_lower = node.lower()
        
        # AAR categorization
        if any(kw in node_lower for kw in ['agent', 'action', 'intent', 'urge', 'do']):
            return 'agent'
        elif any(kw in node_lower for kw in ['arena', 'environment', 'context', 'need', 'constraint']):
            return 'arena'
        elif any(kw in node_lower for kw in ['self', 'relation', 'interplay', 'feedback', 'identity']):
            return 'relation'
        else:
            # Default to relation for core identity concepts
            return 'relation'
    
    def visualize(self, output_path: str = 'identity_graph.png', figsize: Tuple[int, int] = (16, 12)):
        """
        Create and save visualization of identity graph.
        
        Args:
            output_path: Path to save the visualization
            figsize: Figure size (width, height)
        """
        if len(self.graph.nodes()) == 0:
            self.build_graph()
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Calculate layout
        pos = nx.spring_layout(
            self.graph,
            k=VISUALIZATION_CONFIG['layout']['k'],
            iterations=VISUALIZATION_CONFIG['layout']['iterations'],
            seed=VISUALIZATION_CONFIG['layout']['seed']
        )
        
        # Draw nodes by category
        for category in ['agent', 'arena', 'relation']:
            nodes = [n for n, c in self.node_categories.items() if c == category]
            if nodes:
                # Node size based on degree (number of connections)
                node_sizes = [
                    VISUALIZATION_CONFIG['sizes']['node_base'] + 
                    self.graph.degree(n) * VISUALIZATION_CONFIG['sizes']['node_scale']
                    for n in nodes
                ]
                
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=nodes,
                    node_color=VISUALIZATION_CONFIG['colors'][category],
                    node_size=node_sizes,
                    alpha=0.8,
                    ax=ax,
                    label=category.capitalize()
                )
        
        # Draw edges with varying thickness based on confidence
        for (u, v), confidences in self.edge_weights.items():
            avg_confidence = np.mean(confidences)
            edge_width = VISUALIZATION_CONFIG['sizes']['edge_base'] + \
                        avg_confidence * VISUALIZATION_CONFIG['sizes']['edge_scale']
            
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=[(u, v)],
                width=edge_width,
                alpha=0.3,
                edge_color='gray',
                arrows=True,
                arrowsize=10,
                ax=ax
            )
        
        # Draw labels for important nodes (high degree)
        degrees = dict(self.graph.degree())
        important_nodes = {n: n for n, d in degrees.items() if d >= 3}
        
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=important_nodes,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        # Add legend
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        
        # Title and styling
        ax.set_title(
            'EchoSelf Identity Hypergraph\nAgent-Arena-Relation Framework',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        
        # Add statistics
        stats_text = f"Nodes: {len(self.graph.nodes())} | Edges: {len(self.graph.edges())} | Density: {nx.density(self.graph):.3f}"
        ax.text(0.5, -0.05, stats_text, transform=ax.transAxes, 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path


# ============================================================================
# AAR BALANCE VISUALIZATION
# ============================================================================

class AARBalanceVisualizer:
    """
    Visualizes the balance between Agent, Arena, and Relation components.
    """
    
    def __init__(self, aar_data: Dict[str, int]):
        """
        Initialize with AAR component counts.
        
        Args:
            aar_data: Dictionary with 'agent', 'arena', 'relation' counts
        """
        self.aar_data = aar_data
    
    def visualize_balance(self, output_path: str = 'aar_balance.png', figsize: Tuple[int, int] = (12, 8)):
        """
        Create visualization of AAR balance.
        
        Args:
            output_path: Path to save the visualization
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        
        # Extract data
        components = ['Agent', 'Arena', 'Relation']
        counts = [
            self.aar_data.get('agent_count', 0),
            self.aar_data.get('arena_count', 0),
            self.aar_data.get('relation_count', 0)
        ]
        colors = [
            VISUALIZATION_CONFIG['colors']['agent'],
            VISUALIZATION_CONFIG['colors']['arena'],
            VISUALIZATION_CONFIG['colors']['relation']
        ]
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            counts,
            labels=components,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
        
        ax1.set_title('AAR Component Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Bar chart
        bars = ax2.bar(components, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('AAR Component Counts', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Calculate balance score
        total = sum(counts)
        if total > 0:
            ideal = total / 3
            variance = sum((c - ideal) ** 2 for c in counts) / 3
            balance_score = 1 - (variance / (ideal ** 2)) if ideal > 0 else 0
            balance_score = max(0, min(1, balance_score))
        else:
            balance_score = 0
        
        # Add balance score
        fig.text(0.5, 0.02, f'Balance Score: {balance_score:.3f} (1.0 = perfect balance)', 
                ha='center', fontsize=12, fontweight='bold', style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path


# ============================================================================
# IDENTITY EVOLUTION TIMELINE
# ============================================================================

class IdentityEvolutionVisualizer:
    """
    Visualizes the evolution of identity over time.
    """
    
    def __init__(self, evolution_data: List[Dict[str, Any]]):
        """
        Initialize with evolution timeline data.
        
        Args:
            evolution_data: List of snapshots with timestamps and metrics
        """
        self.evolution_data = evolution_data
    
    def visualize_timeline(self, output_path: str = 'identity_evolution.png', figsize: Tuple[int, int] = (14, 10)):
        """
        Create timeline visualization of identity evolution.
        
        Args:
            output_path: Path to save the visualization
            figsize: Figure size
        """
        if not self.evolution_data:
            print("No evolution data available")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, facecolor='white', sharex=True)
        
        # Extract time series data
        timestamps = [d['timestamp'] for d in self.evolution_data]
        
        # Convert timestamps to relative time (hours from start)
        if timestamps:
            start_time = datetime.fromisoformat(timestamps[0])
            time_hours = [(datetime.fromisoformat(ts) - start_time).total_seconds() / 3600 
                         for ts in timestamps]
        else:
            time_hours = []
        
        # Plot 1: Total identity size
        total_tuples = [d.get('total_tuples', 0) for d in self.evolution_data]
        axes[0].plot(time_hours, total_tuples, marker='o', linewidth=2, markersize=6, 
                    color='#2E86AB', label='Total Tuples')
        axes[0].fill_between(time_hours, total_tuples, alpha=0.3, color='#2E86AB')
        axes[0].set_ylabel('Total Tuples', fontsize=11, fontweight='bold')
        axes[0].set_title('Identity Growth Over Time', fontsize=13, fontweight='bold', pad=10)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='upper left')
        
        # Plot 2: AAR component evolution
        agent_counts = [d.get('aar_balance', {}).get('agent_count', 0) for d in self.evolution_data]
        arena_counts = [d.get('aar_balance', {}).get('arena_count', 0) for d in self.evolution_data]
        relation_counts = [d.get('aar_balance', {}).get('relation_count', 0) for d in self.evolution_data]
        
        axes[1].plot(time_hours, agent_counts, marker='o', linewidth=2, markersize=5,
                    color=VISUALIZATION_CONFIG['colors']['agent'], label='Agent')
        axes[1].plot(time_hours, arena_counts, marker='s', linewidth=2, markersize=5,
                    color=VISUALIZATION_CONFIG['colors']['arena'], label='Arena')
        axes[1].plot(time_hours, relation_counts, marker='^', linewidth=2, markersize=5,
                    color=VISUALIZATION_CONFIG['colors']['relation'], label='Relation')
        
        axes[1].set_ylabel('Component Count', fontsize=11, fontweight='bold')
        axes[1].set_title('AAR Component Evolution', fontsize=13, fontweight='bold', pad=10)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='upper left')
        
        # Plot 3: Average confidence
        avg_confidences = [d.get('average_confidence', 0) for d in self.evolution_data]
        axes[2].plot(time_hours, avg_confidences, marker='D', linewidth=2, markersize=6,
                    color='#A23B72', label='Avg Confidence')
        axes[2].fill_between(time_hours, avg_confidences, alpha=0.3, color='#A23B72')
        axes[2].set_ylabel('Confidence', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        axes[2].set_title('Identity Confidence Over Time', fontsize=13, fontweight='bold', pad=10)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].legend(loc='upper left')
        axes[2].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path


# ============================================================================
# MEMORY DISTRIBUTION VISUALIZATION
# ============================================================================

class MemoryDistributionVisualizer:
    """
    Visualizes the distribution of memory types in the identity.
    """
    
    def __init__(self, memory_data: Dict[str, int]):
        """
        Initialize with memory distribution data.
        
        Args:
            memory_data: Dictionary with memory type counts
        """
        self.memory_data = memory_data
    
    def visualize_distribution(self, output_path: str = 'memory_distribution.png', 
                              figsize: Tuple[int, int] = (12, 8)):
        """
        Create visualization of memory distribution.
        
        Args:
            output_path: Path to save the visualization
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        
        # Extract data
        memory_types = ['Declarative', 'Procedural', 'Episodic', 'Intentional']
        counts = [
            self.memory_data.get('declarative', 0),
            self.memory_data.get('procedural', 0),
            self.memory_data.get('episodic', 0),
            self.memory_data.get('intentional', 0)
        ]
        colors = [
            VISUALIZATION_CONFIG['colors']['declarative'],
            VISUALIZATION_CONFIG['colors']['procedural'],
            VISUALIZATION_CONFIG['colors']['episodic'],
            VISUALIZATION_CONFIG['colors']['intentional']
        ]
        
        # Donut chart
        wedges, texts, autotexts = ax1.pie(
            counts,
            labels=memory_types,
            colors=colors,
            autopct='%1.1f%%',
            startangle=45,
            pctdistance=0.85,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        # Draw circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
        
        ax1.set_title('Memory Type Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Horizontal bar chart
        y_pos = np.arange(len(memory_types))
        bars = ax2.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax2.text(width, i, f'  {count}', va='center', fontsize=11, fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(memory_types, fontsize=11)
        ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Memory Type Counts', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Calculate diversity score (entropy-based)
        total = sum(counts)
        if total > 0:
            proportions = [c / total for c in counts if c > 0]
            entropy = -sum(p * np.log(p) for p in proportions)
            max_entropy = np.log(len(memory_types))
            diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity_score = 0
        
        # Add diversity score
        fig.text(0.5, 0.02, f'Memory Diversity Score: {diversity_score:.3f} (1.0 = perfectly balanced)', 
                ha='center', fontsize=12, fontweight='bold', style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_path


# ============================================================================
# COMPREHENSIVE VISUALIZATION SUITE
# ============================================================================

class IdentityVisualizationSuite:
    """
    Comprehensive suite for visualizing all aspects of EchoSelf identity.
    """
    
    def __init__(self, identity_data: Dict[str, Any], evolution_data: Optional[List[Dict]] = None):
        """
        Initialize visualization suite.
        
        Args:
            identity_data: Complete identity data including tuples and metrics
            evolution_data: Optional timeline data for evolution visualization
        """
        self.identity_data = identity_data
        self.evolution_data = evolution_data or []
    
    def generate_all_visualizations(self, output_dir: str = '.') -> Dict[str, str]:
        """
        Generate all visualizations and return paths.
        
        Args:
            output_dir: Directory to save visualizations
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        results = {}
        
        # 1. Identity graph
        print("Generating identity graph visualization...")
        graph_viz = IdentityGraphVisualizer(self.identity_data)
        results['identity_graph'] = graph_viz.visualize(
            output_path=f"{output_dir}/identity_graph.png"
        )
        
        # 2. AAR balance
        print("Generating AAR balance visualization...")
        aar_data = self.identity_data.get('core_identity', {}).get('aar_framework', {})
        aar_viz = AARBalanceVisualizer(aar_data)
        results['aar_balance'] = aar_viz.visualize_balance(
            output_path=f"{output_dir}/aar_balance.png"
        )
        
        # 3. Memory distribution
        print("Generating memory distribution visualization...")
        memory_data = self.identity_data.get('core_identity', {}).get('memory_distribution', {})
        memory_viz = MemoryDistributionVisualizer(memory_data)
        results['memory_distribution'] = memory_viz.visualize_distribution(
            output_path=f"{output_dir}/memory_distribution.png"
        )
        
        # 4. Evolution timeline (if data available)
        if self.evolution_data:
            print("Generating identity evolution timeline...")
            evolution_viz = IdentityEvolutionVisualizer(self.evolution_data)
            results['evolution_timeline'] = evolution_viz.visualize_timeline(
                output_path=f"{output_dir}/identity_evolution.png"
            )
        
        print(f"\nGenerated {len(results)} visualizations in {output_dir}/")
        return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_identity_from_json(json_path: str) -> Dict[str, Any]:
    """Load identity data from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_evolution_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load evolution timeline from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data.get('timeline', [])


if __name__ == '__main__':
    # Example usage
    print("Identity Visualization Module for EchoSelf")
    print("=" * 60)
    print("\nThis module provides tools for visualizing:")
    print("  - Hypergraph identity networks")
    print("  - AAR component balance")
    print("  - Identity evolution timelines")
    print("  - Memory distribution")
    print("\nImport and use the visualization classes in your code.")
