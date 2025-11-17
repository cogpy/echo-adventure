"""
Autonomous Self-Improvement Loop for EchoSelf v0.6.0

This module implements the complete autonomous self-improvement loop that integrates:
1. Real-time AAR monitoring
2. LLM-based corpus generation
3. Self-regulation and parameter adjustment
4. Fine-tuning execution
5. Identity evolution tracking

The loop enables the model to continuously observe itself, generate training data,
and improve through self-directed fine-tuning cycles.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class LoopIteration:
    """
    Represents a single iteration of the autonomous self-improvement loop.
    """
    iteration_number: int
    timestamp: str
    monitoring_summary: Dict[str, Any]
    corpus_stats: Dict[str, Any]
    regulation_actions: List[Dict[str, Any]]
    identity_evolution: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)


class AutonomousSelfImprovementLoop:
    """
    The main autonomous loop that orchestrates continuous self-improvement.
    
    This loop operates in cycles:
    1. Monitor: Observe current AAR state and cognitive dynamics
    2. Reflect: Analyze monitoring data and identify areas for growth
    3. Generate: Create new training data based on reflection
    4. Regulate: Adjust parameters to maintain balance
    5. Evolve: Update identity representation
    6. (Optional) Fine-tune: Execute fine-tuning with new corpus
    """
    
    def __init__(
        self,
        identity_context: Dict[str, Any],
        monitoring_enabled: bool = True,
        generation_enabled: bool = True,
        regulation_enabled: bool = True,
        finetuning_enabled: bool = False,
        output_dir: str = "./autonomous_loop_output"
    ):
        """
        Initialize the autonomous self-improvement loop.
        
        Args:
            identity_context: Initial identity context
            monitoring_enabled: Enable AAR monitoring
            generation_enabled: Enable corpus generation
            regulation_enabled: Enable self-regulation
            finetuning_enabled: Enable automatic fine-tuning
            output_dir: Directory for output artifacts
        """
        self.identity_context = identity_context
        self.monitoring_enabled = monitoring_enabled
        self.generation_enabled = generation_enabled
        self.regulation_enabled = regulation_enabled
        self.finetuning_enabled = finetuning_enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (lazy loading)
        self.monitor = None
        self.generator = None
        self.regulator = None
        
        # Loop state
        self.iteration_history: List[LoopIteration] = []
        self.current_iteration = 0
        self.total_examples_generated = 0
        self.identity_evolution_log = []
        
        # Performance tracking
        self.performance_baseline = {
            'coherence': 0.5,
            'balance': 0.5,
            'diversity': 0.5,
            'quality': 0.5
        }
    
    def _initialize_monitor(self):
        """Lazy initialization of AAR monitor"""
        if self.monitor is None and self.monitoring_enabled:
            from echo_adventure.aar_monitor import AARStateMonitor
            self.monitor = AARStateMonitor()
    
    def _initialize_generator(self):
        """Lazy initialization of LLM corpus generator"""
        if self.generator is None and self.generation_enabled:
            from echo_adventure.llm_corpus_generator import LLMCorpusGenerator
            self.generator = LLMCorpusGenerator(
                identity_context=self.identity_context,
                model="gpt-4.1-mini",
                temperature=0.8
            )
    
    def _initialize_regulator(self):
        """Lazy initialization of AAR self-regulator"""
        if self.regulator is None and self.regulation_enabled:
            from echo_adventure.aar_monitor import AARSelfRegulator
            self.regulator = AARSelfRegulator()
    
    def _monitor_phase(self, num_steps: int = 10) -> Dict[str, Any]:
        """
        Execute monitoring phase.
        
        Args:
            num_steps: Number of monitoring steps
            
        Returns:
            Monitoring summary
        """
        self._initialize_monitor()
        
        if not self.monitoring_enabled or self.monitor is None:
            return {"status": "disabled"}
        
        print(f"\n[Monitor Phase] Observing AAR state for {num_steps} steps...")
        
        # Simulate monitoring steps
        import torch
        for step in range(num_steps):
            # Create synthetic AAR state for demonstration
            agent_state = torch.randn(1, 256)
            arena_state = torch.randn(1, 256)
            relation_state = torch.randn(1, 256)
            attention_weights = torch.softmax(torch.randn(8, 10, 10), dim=-1)
            
            self.monitor.capture_snapshot(
                agent_state=agent_state,
                arena_state=arena_state,
                relation_state=relation_state,
                attention_weights=attention_weights,
                step=step
            )
        
        # Analyze stability
        is_stable, stability_report = self.monitor.analyze_stability()
        
        summary = {
            "num_snapshots": len(self.monitor.snapshots),
            "is_stable": is_stable,
            "stability_report": stability_report,
            "alerts": [alert.__dict__ for alert in self.monitor.alerts[-5:]],  # Last 5 alerts
            "final_snapshot": self.monitor.snapshots[-1].__dict__ if self.monitor.snapshots else None
        }
        
        print(f"  Captured {len(self.monitor.snapshots)} snapshots")
        print(f"  System stable: {is_stable}")
        print(f"  Alerts: {len(self.monitor.alerts)}")
        
        return summary
    
    def _reflect_phase(self, monitoring_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute reflection phase to analyze monitoring data.
        
        Args:
            monitoring_summary: Summary from monitoring phase
            
        Returns:
            Reflection insights
        """
        print("\n[Reflect Phase] Analyzing monitoring data...")
        
        insights = {
            "needs_regulation": False,
            "focus_areas": [],
            "identity_gaps": [],
            "recommended_actions": []
        }
        
        if not monitoring_summary or monitoring_summary.get("status") == "disabled":
            return insights
        
        # Analyze stability
        if not monitoring_summary.get("is_stable", True):
            insights["needs_regulation"] = True
            insights["recommended_actions"].append("Apply AAR regulation to restore balance")
        
        # Analyze alerts
        alerts = monitoring_summary.get("alerts", [])
        if alerts:
            for alert in alerts:
                alert_type = alert.get("alert_type", "unknown")
                if alert_type == "imbalance":
                    insights["focus_areas"].append("AAR balance")
                elif alert_type == "low_coherence":
                    insights["focus_areas"].append("cognitive coherence")
                elif alert_type == "attention_collapse":
                    insights["focus_areas"].append("attention diversity")
        
        # Identify identity gaps
        final_snapshot = monitoring_summary.get("final_snapshot")
        if final_snapshot:
            balance = final_snapshot.get("balance_score", 0.8)
            if balance < 0.6:
                insights["identity_gaps"].append("Weak AAR balance - need more balanced training data")
            
            coherence = final_snapshot.get("coherence", 0.8)
            if coherence < 0.6:
                insights["identity_gaps"].append("Low coherence - need more structured introspection")
        
        print(f"  Needs regulation: {insights['needs_regulation']}")
        print(f"  Focus areas: {insights['focus_areas']}")
        print(f"  Identity gaps: {len(insights['identity_gaps'])}")
        
        return insights
    
    def _generate_phase(
        self,
        reflection_insights: Dict[str, Any],
        num_examples: int = 20
    ) -> Dict[str, Any]:
        """
        Execute generation phase to create new training data.
        
        Args:
            reflection_insights: Insights from reflection phase
            num_examples: Number of examples to generate
            
        Returns:
            Generation statistics
        """
        self._initialize_generator()
        
        if not self.generation_enabled or self.generator is None:
            return {"status": "disabled"}
        
        print(f"\n[Generate Phase] Creating {num_examples} new training examples...")
        
        # Extract AAR contexts from monitoring if available
        aar_contexts = None
        if self.monitor and self.monitor.snapshots:
            aar_contexts = []
            for snapshot in self.monitor.snapshots[-10:]:  # Use last 10 snapshots
                aar_contexts.append({
                    'agent': snapshot.agent_magnitude,
                    'arena': snapshot.arena_magnitude,
                    'relation': snapshot.relation_magnitude,
                    'balance': snapshot.balance_score
                })
        
        # Generate corpus
        corpus = self.generator.generate_corpus(
            num_examples=num_examples,
            min_quality=0.6,
            min_diversity=0.3,
            multi_turn_ratio=0.3,
            aar_contexts=aar_contexts
        )
        
        # Save corpus
        corpus_path = self.output_dir / f"corpus_iteration_{self.current_iteration}.jsonl"
        self.generator.export_corpus(corpus, str(corpus_path))
        
        # Save with metadata
        metadata_path = self.output_dir / f"corpus_metadata_iteration_{self.current_iteration}.jsonl"
        self.generator.export_corpus_with_metadata(corpus, str(metadata_path))
        
        self.total_examples_generated += len(corpus)
        
        stats = {
            "num_generated": len(corpus),
            "avg_quality": sum(e.quality_score for e in corpus) / len(corpus) if corpus else 0,
            "avg_diversity": sum(e.diversity_score for e in corpus) / len(corpus) if corpus else 0,
            "corpus_path": str(corpus_path),
            "metadata_path": str(metadata_path)
        }
        
        print(f"  Generated {len(corpus)} examples")
        print(f"  Average quality: {stats['avg_quality']:.2f}")
        print(f"  Average diversity: {stats['avg_diversity']:.2f}")
        
        return stats
    
    def _regulate_phase(
        self,
        monitoring_summary: Dict[str, Any],
        reflection_insights: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute regulation phase to adjust parameters.
        
        Args:
            monitoring_summary: Summary from monitoring phase
            reflection_insights: Insights from reflection phase
            
        Returns:
            List of regulation actions taken
        """
        self._initialize_regulator()
        
        if not self.regulation_enabled or self.regulator is None:
            return [{"status": "disabled"}]
        
        print("\n[Regulate Phase] Adjusting parameters for balance...")
        
        actions = []
        
        if not reflection_insights.get("needs_regulation", False):
            print("  No regulation needed - system is stable")
            return [{"action": "none", "reason": "system_stable"}]
        
        # Get latest snapshot
        if not self.monitor or not self.monitor.snapshots:
            return [{"action": "none", "reason": "no_monitoring_data"}]
        
        latest_snapshot = self.monitor.snapshots[-1]
        
        # Compute adjustments
        adjustments = self.regulator.compute_adjustments(latest_snapshot)
        
        # Apply adjustments (in a real system, this would update model parameters)
        for param, adjustment in adjustments.items():
            action = {
                "parameter": param,
                "adjustment": adjustment,
                "timestamp": datetime.now().isoformat()
            }
            actions.append(action)
            print(f"  Adjusted {param}: {adjustment:+.4f}")
        
        return actions
    
    def _evolve_phase(
        self,
        corpus_stats: Dict[str, Any],
        regulation_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute evolution phase to update identity representation.
        
        Args:
            corpus_stats: Statistics from generation phase
            regulation_actions: Actions from regulation phase
            
        Returns:
            Identity evolution summary
        """
        print("\n[Evolve Phase] Updating identity representation...")
        
        # Update identity context with new information
        evolution = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "corpus_integration": {
                "examples_added": corpus_stats.get("num_generated", 0),
                "quality_level": corpus_stats.get("avg_quality", 0)
            },
            "regulation_updates": len(regulation_actions),
            "identity_growth": {
                "total_training_examples": self.total_examples_generated,
                "iterations_completed": self.current_iteration + 1
            }
        }
        
        self.identity_evolution_log.append(evolution)
        
        # Update identity context
        self.identity_context["total_training_examples"] = self.total_examples_generated
        self.identity_context["loop_iterations"] = self.current_iteration + 1
        self.identity_context["last_evolution"] = datetime.now().isoformat()
        
        print(f"  Identity updated with {corpus_stats.get('num_generated', 0)} new examples")
        print(f"  Total training examples: {self.total_examples_generated}")
        
        return evolution
    
    def _compute_performance_metrics(
        self,
        monitoring_summary: Dict[str, Any],
        corpus_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute performance metrics for this iteration.
        
        Args:
            monitoring_summary: Summary from monitoring phase
            corpus_stats: Statistics from generation phase
            
        Returns:
            Performance metrics
        """
        metrics = {}
        
        # Coherence from monitoring
        if monitoring_summary and monitoring_summary.get("final_snapshot"):
            metrics["coherence"] = monitoring_summary["final_snapshot"].get("coherence", 0.5)
            metrics["balance"] = monitoring_summary["final_snapshot"].get("balance_score", 0.5)
        else:
            metrics["coherence"] = self.performance_baseline["coherence"]
            metrics["balance"] = self.performance_baseline["balance"]
        
        # Quality and diversity from generation
        metrics["quality"] = corpus_stats.get("avg_quality", self.performance_baseline["quality"])
        metrics["diversity"] = corpus_stats.get("avg_diversity", self.performance_baseline["diversity"])
        
        # Compute improvement
        for key in metrics:
            baseline = self.performance_baseline.get(key, 0.5)
            improvement = metrics[key] - baseline
            metrics[f"{key}_improvement"] = improvement
        
        return metrics
    
    def run_iteration(self, num_examples: int = 20) -> LoopIteration:
        """
        Run a single iteration of the autonomous loop.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            LoopIteration object with results
        """
        print(f"\n{'='*80}")
        print(f"AUTONOMOUS LOOP - ITERATION {self.current_iteration + 1}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Phase 1: Monitor
        monitoring_summary = self._monitor_phase(num_steps=10)
        
        # Phase 2: Reflect
        reflection_insights = self._reflect_phase(monitoring_summary)
        
        # Phase 3: Generate
        corpus_stats = self._generate_phase(reflection_insights, num_examples)
        
        # Phase 4: Regulate
        regulation_actions = self._regulate_phase(monitoring_summary, reflection_insights)
        
        # Phase 5: Evolve
        identity_evolution = self._evolve_phase(corpus_stats, regulation_actions)
        
        # Compute performance metrics
        performance_metrics = self._compute_performance_metrics(monitoring_summary, corpus_stats)
        
        # Create iteration record
        iteration = LoopIteration(
            iteration_number=self.current_iteration,
            timestamp=datetime.now().isoformat(),
            monitoring_summary=monitoring_summary,
            corpus_stats=corpus_stats,
            regulation_actions=regulation_actions,
            identity_evolution=identity_evolution,
            performance_metrics=performance_metrics
        )
        
        self.iteration_history.append(iteration)
        self.current_iteration += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"ITERATION COMPLETE in {elapsed_time:.2f}s")
        print(f"Performance: Coherence={performance_metrics.get('coherence', 0):.2f}, "
              f"Balance={performance_metrics.get('balance', 0):.2f}, "
              f"Quality={performance_metrics.get('quality', 0):.2f}, "
              f"Diversity={performance_metrics.get('diversity', 0):.2f}")
        print(f"{'='*80}\n")
        
        return iteration
    
    def run_continuous(
        self,
        num_iterations: int = 5,
        examples_per_iteration: int = 20,
        save_checkpoints: bool = True
    ):
        """
        Run multiple iterations of the autonomous loop.
        
        Args:
            num_iterations: Number of iterations to run
            examples_per_iteration: Examples to generate per iteration
            save_checkpoints: Save checkpoints after each iteration
        """
        print(f"\nStarting autonomous self-improvement loop for {num_iterations} iterations...")
        print(f"Generating {examples_per_iteration} examples per iteration")
        print(f"Output directory: {self.output_dir}\n")
        
        for i in range(num_iterations):
            iteration = self.run_iteration(num_examples=examples_per_iteration)
            
            if save_checkpoints:
                self.save_checkpoint()
        
        print(f"\n{'='*80}")
        print(f"AUTONOMOUS LOOP COMPLETE")
        print(f"{'='*80}")
        print(f"Total iterations: {self.current_iteration}")
        print(f"Total examples generated: {self.total_examples_generated}")
        print(f"Output directory: {self.output_dir}")
        
        # Generate summary report
        self.generate_summary_report()
    
    def save_checkpoint(self):
        """Save current loop state to checkpoint file"""
        checkpoint = {
            "current_iteration": self.current_iteration,
            "total_examples_generated": self.total_examples_generated,
            "identity_context": self.identity_context,
            "identity_evolution_log": self.identity_evolution_log,
            "iteration_history": [it.to_dict() for it in self.iteration_history],
            "performance_baseline": self.performance_baseline
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_iteration_{self.current_iteration}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def generate_summary_report(self):
        """Generate a summary report of all iterations"""
        report_path = self.output_dir / "autonomous_loop_summary.json"
        
        summary = {
            "total_iterations": self.current_iteration,
            "total_examples_generated": self.total_examples_generated,
            "identity_evolution": self.identity_evolution_log,
            "performance_trajectory": [
                {
                    "iteration": it.iteration_number,
                    "metrics": it.performance_metrics
                }
                for it in self.iteration_history
            ],
            "final_identity_context": self.identity_context
        }
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary report saved: {report_path}")
