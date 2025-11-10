"""
Real-Time AAR State Monitoring Module (v0.5.0)

This module provides real-time monitoring and analysis of the Agent-Arena-Relation
geometric architecture during model inference and generation.

Key Features:
1. Real-time AAR balance tracking during generation
2. Attention pattern analysis and visualization
3. State trajectory recording and analysis
4. Anomaly detection in AAR dynamics
5. Self-regulation mechanisms for maintaining balance
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import numpy as np
from collections import deque


@dataclass
class AARSnapshot:
    """A snapshot of AAR state at a specific point in time"""
    timestamp: str
    step: int
    agent_magnitude: float
    arena_magnitude: float
    relation_magnitude: float
    balance_score: float
    interaction_strength: float
    coherence: float
    attention_entropy: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class AARAlert:
    """Alert for AAR state anomalies"""
    timestamp: str
    step: int
    alert_type: str  # 'imbalance', 'low_coherence', 'attention_collapse', 'drift'
    severity: str  # 'low', 'medium', 'high'
    message: str
    metrics: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)


class AARStateMonitor:
    """
    Real-time monitor for AAR geometric architecture state.
    Tracks balance, coherence, and dynamics during generation.
    """
    
    def __init__(
        self,
        history_size: int = 100,
        balance_threshold: float = 0.3,
        coherence_threshold: float = 0.5,
        enable_alerts: bool = True
    ):
        """
        Initialize AAR state monitor.
        
        Args:
            history_size: Number of snapshots to keep in memory
            balance_threshold: Threshold for balance alerts (0-1)
            coherence_threshold: Threshold for coherence alerts (0-1)
            enable_alerts: Whether to generate alerts
        """
        self.history_size = history_size
        self.balance_threshold = balance_threshold
        self.coherence_threshold = coherence_threshold
        self.enable_alerts = enable_alerts
        
        # State tracking
        self.snapshots: deque = deque(maxlen=history_size)
        self.alerts: List[AARAlert] = []
        self.current_step = 0
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'total_alerts': 0,
            'avg_balance': 0.0,
            'avg_coherence': 0.0,
            'max_imbalance': 0.0,
            'min_coherence': 1.0
        }
    
    def capture_snapshot(
        self,
        agent: torch.Tensor,
        arena: torch.Tensor,
        relation: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None
    ) -> AARSnapshot:
        """
        Capture current AAR state as a snapshot.
        
        Args:
            agent: Agent component tensor [batch, seq, d_model]
            arena: Arena component tensor [batch, seq, d_model]
            relation: Relation component tensor [batch, seq, d_model]
            attention_weights: Optional attention weights [batch, heads, seq, seq]
        
        Returns:
            AARSnapshot object
        """
        # Compute magnitudes (L2 norms)
        agent_mag = torch.norm(agent, p=2, dim=-1).mean().item()
        arena_mag = torch.norm(arena, p=2, dim=-1).mean().item()
        relation_mag = torch.norm(relation, p=2, dim=-1).mean().item()
        
        # Compute balance score (how evenly distributed the components are)
        total_mag = agent_mag + arena_mag + relation_mag
        if total_mag > 0:
            proportions = torch.tensor([agent_mag, arena_mag, relation_mag]) / total_mag
            # Use entropy as balance measure (higher = more balanced)
            balance_score = -torch.sum(proportions * torch.log(proportions + 1e-10)).item() / np.log(3)
        else:
            balance_score = 0.0
        
        # Compute interaction strength (cosine similarity between agent and arena)
        agent_flat = agent.reshape(-1, agent.shape[-1])
        arena_flat = arena.reshape(-1, arena.shape[-1])
        interaction = torch.nn.functional.cosine_similarity(
            agent_flat.mean(0, keepdim=True),
            arena_flat.mean(0, keepdim=True)
        ).item()
        
        # Compute coherence (how well relation aligns with agent-arena interaction)
        relation_flat = relation.reshape(-1, relation.shape[-1])
        expected_relation = (agent_flat + arena_flat) / 2
        coherence = torch.nn.functional.cosine_similarity(
            relation_flat.mean(0, keepdim=True),
            expected_relation.mean(0, keepdim=True)
        ).item()
        
        # Compute attention entropy if available
        if attention_weights is not None:
            # Average over batch and heads
            attn_probs = attention_weights.mean(dim=(0, 1))  # [seq, seq]
            attn_entropy = -torch.sum(
                attn_probs * torch.log(attn_probs + 1e-10),
                dim=-1
            ).mean().item()
        else:
            attn_entropy = 0.0
        
        # Create snapshot
        snapshot = AARSnapshot(
            timestamp=datetime.now().isoformat(),
            step=self.current_step,
            agent_magnitude=agent_mag,
            arena_magnitude=arena_mag,
            relation_magnitude=relation_mag,
            balance_score=balance_score,
            interaction_strength=interaction,
            coherence=coherence,
            attention_entropy=attn_entropy
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        self.current_step += 1
        
        # Update statistics
        self._update_stats(snapshot)
        
        # Check for alerts
        if self.enable_alerts:
            self._check_alerts(snapshot)
        
        return snapshot
    
    def _update_stats(self, snapshot: AARSnapshot):
        """Update running statistics"""
        self.stats['total_steps'] += 1
        
        # Running averages
        n = self.stats['total_steps']
        self.stats['avg_balance'] = (
            (self.stats['avg_balance'] * (n - 1) + snapshot.balance_score) / n
        )
        self.stats['avg_coherence'] = (
            (self.stats['avg_coherence'] * (n - 1) + snapshot.coherence) / n
        )
        
        # Extremes
        imbalance = 1.0 - snapshot.balance_score
        if imbalance > self.stats['max_imbalance']:
            self.stats['max_imbalance'] = imbalance
        
        if snapshot.coherence < self.stats['min_coherence']:
            self.stats['min_coherence'] = snapshot.coherence
    
    def _check_alerts(self, snapshot: AARSnapshot):
        """Check for anomalies and generate alerts"""
        alerts = []
        
        # Check balance
        if snapshot.balance_score < self.balance_threshold:
            severity = 'high' if snapshot.balance_score < 0.2 else 'medium'
            alerts.append(AARAlert(
                timestamp=snapshot.timestamp,
                step=snapshot.step,
                alert_type='imbalance',
                severity=severity,
                message=f"AAR components are imbalanced (score: {snapshot.balance_score:.3f})",
                metrics={
                    'balance_score': snapshot.balance_score,
                    'agent_mag': snapshot.agent_magnitude,
                    'arena_mag': snapshot.arena_magnitude,
                    'relation_mag': snapshot.relation_magnitude
                }
            ))
        
        # Check coherence
        if snapshot.coherence < self.coherence_threshold:
            severity = 'high' if snapshot.coherence < 0.3 else 'medium'
            alerts.append(AARAlert(
                timestamp=snapshot.timestamp,
                step=snapshot.step,
                alert_type='low_coherence',
                severity=severity,
                message=f"Relation coherence is low (score: {snapshot.coherence:.3f})",
                metrics={
                    'coherence': snapshot.coherence,
                    'interaction_strength': snapshot.interaction_strength
                }
            ))
        
        # Check attention entropy (too low = collapsed attention)
        if snapshot.attention_entropy < 0.5 and snapshot.attention_entropy > 0:
            alerts.append(AARAlert(
                timestamp=snapshot.timestamp,
                step=snapshot.step,
                alert_type='attention_collapse',
                severity='medium',
                message=f"Attention patterns may be collapsing (entropy: {snapshot.attention_entropy:.3f})",
                metrics={
                    'attention_entropy': snapshot.attention_entropy
                }
            ))
        
        # Check for drift (if we have history)
        if len(self.snapshots) > 10:
            recent_balance = np.mean([s.balance_score for s in list(self.snapshots)[-10:]])
            if abs(snapshot.balance_score - recent_balance) > 0.3:
                alerts.append(AARAlert(
                    timestamp=snapshot.timestamp,
                    step=snapshot.step,
                    alert_type='drift',
                    severity='low',
                    message=f"Sudden drift in AAR balance detected",
                    metrics={
                        'current_balance': snapshot.balance_score,
                        'recent_avg_balance': recent_balance,
                        'drift': abs(snapshot.balance_score - recent_balance)
                    }
                ))
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            self.stats['total_alerts'] += 1
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current AAR state summary"""
        if not self.snapshots:
            return {'status': 'no_data'}
        
        latest = self.snapshots[-1]
        
        return {
            'status': 'active',
            'current_step': self.current_step,
            'latest_snapshot': latest.to_dict(),
            'recent_alerts': [a.to_dict() for a in self.alerts[-5:]],
            'statistics': self.stats.copy()
        }
    
    def get_trajectory(self, window: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get AAR state trajectory over time.
        
        Args:
            window: Number of recent snapshots (None = all)
        
        Returns:
            List of snapshot dictionaries
        """
        snapshots = list(self.snapshots)
        if window:
            snapshots = snapshots[-window:]
        
        return [s.to_dict() for s in snapshots]
    
    def analyze_stability(self) -> Dict[str, Any]:
        """Analyze AAR stability over recent history"""
        if len(self.snapshots) < 10:
            return {'status': 'insufficient_data', 'message': 'Need at least 10 snapshots'}
        
        recent = list(self.snapshots)[-20:]
        
        # Compute variance in key metrics
        balance_scores = [s.balance_score for s in recent]
        coherence_scores = [s.coherence for s in recent]
        
        balance_variance = np.var(balance_scores)
        coherence_variance = np.var(coherence_scores)
        
        # Compute trend (simple linear regression slope)
        steps = np.arange(len(balance_scores))
        balance_trend = np.polyfit(steps, balance_scores, 1)[0]
        coherence_trend = np.polyfit(steps, coherence_scores, 1)[0]
        
        # Determine stability
        is_stable = (
            balance_variance < 0.05 and
            coherence_variance < 0.05 and
            abs(balance_trend) < 0.01 and
            abs(coherence_trend) < 0.01
        )
        
        return {
            'status': 'stable' if is_stable else 'unstable',
            'balance_variance': balance_variance,
            'coherence_variance': coherence_variance,
            'balance_trend': balance_trend,
            'coherence_trend': coherence_trend,
            'mean_balance': np.mean(balance_scores),
            'mean_coherence': np.mean(coherence_scores),
            'recommendation': self._get_stability_recommendation(
                is_stable, balance_trend, coherence_trend
            )
        }
    
    def _get_stability_recommendation(
        self,
        is_stable: bool,
        balance_trend: float,
        coherence_trend: float
    ) -> str:
        """Get recommendation based on stability analysis"""
        if is_stable:
            return "AAR state is stable. Continue current operation."
        
        recommendations = []
        
        if balance_trend < -0.01:
            recommendations.append("Balance is decreasing. Consider adjusting layer weights.")
        elif balance_trend > 0.01:
            recommendations.append("Balance is increasing. Monitor for over-correction.")
        
        if coherence_trend < -0.01:
            recommendations.append("Coherence is decreasing. Check relation feedback mechanisms.")
        elif coherence_trend > 0.01:
            recommendations.append("Coherence is improving. Current trajectory is positive.")
        
        return " ".join(recommendations) if recommendations else "Monitor state closely."
    
    def export_monitoring_data(self, filepath: str):
        """Export all monitoring data to JSON file"""
        data = {
            'version': 'v0.5.0',
            'monitor_config': {
                'history_size': self.history_size,
                'balance_threshold': self.balance_threshold,
                'coherence_threshold': self.coherence_threshold
            },
            'statistics': self.stats,
            'trajectory': self.get_trajectory(),
            'alerts': [a.to_dict() for a in self.alerts],
            'stability_analysis': self.analyze_stability(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def reset(self):
        """Reset monitor state"""
        self.snapshots.clear()
        self.alerts.clear()
        self.current_step = 0
        self.stats = {
            'total_steps': 0,
            'total_alerts': 0,
            'avg_balance': 0.0,
            'avg_coherence': 0.0,
            'max_imbalance': 0.0,
            'min_coherence': 1.0
        }


class AARSelfRegulator:
    """
    Self-regulation mechanism for maintaining AAR balance.
    Adjusts parameters dynamically based on monitoring feedback.
    """
    
    def __init__(self, monitor: AARStateMonitor, adaptation_rate: float = 0.1):
        """
        Initialize self-regulator.
        
        Args:
            monitor: AARStateMonitor instance to read from
            adaptation_rate: Rate of parameter adjustment (0-1)
        """
        self.monitor = monitor
        self.adaptation_rate = adaptation_rate
        self.adjustments_made = 0
    
    def compute_adjustments(
        self,
        current_snapshot: AARSnapshot
    ) -> Dict[str, float]:
        """
        Compute parameter adjustments based on current state.
        
        Returns:
            Dictionary of parameter adjustments
        """
        adjustments = {
            'temperature': 0.0,
            'attention_scale': 0.0,
            'layer_weight_shift': 0.0
        }
        
        # Adjust based on balance
        if current_snapshot.balance_score < 0.4:
            # Low balance - increase temperature to add diversity
            adjustments['temperature'] = self.adaptation_rate * 0.1
            adjustments['attention_scale'] = -self.adaptation_rate * 0.05
        elif current_snapshot.balance_score > 0.9:
            # Very high balance might indicate over-smoothing
            adjustments['temperature'] = -self.adaptation_rate * 0.05
        
        # Adjust based on coherence
        if current_snapshot.coherence < 0.5:
            # Low coherence - strengthen relation feedback
            adjustments['layer_weight_shift'] = self.adaptation_rate * 0.1
        
        # Adjust based on attention entropy
        if current_snapshot.attention_entropy < 0.5 and current_snapshot.attention_entropy > 0:
            # Collapsed attention - increase diversity
            adjustments['attention_scale'] = self.adaptation_rate * 0.1
        
        return adjustments
    
    def apply_adjustments(
        self,
        model: nn.Module,
        adjustments: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply computed adjustments to model parameters.
        
        Args:
            model: Model with adjustable parameters
            adjustments: Dictionary of adjustments
        
        Returns:
            Report of applied adjustments
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'adjustments_applied': {},
            'success': True
        }
        
        try:
            # Apply temperature adjustment (if model has this parameter)
            if hasattr(model, 'temperature') and adjustments['temperature'] != 0:
                old_temp = model.temperature.item()
                new_temp = max(0.1, min(2.0, old_temp + adjustments['temperature']))
                model.temperature.data = torch.tensor(new_temp)
                report['adjustments_applied']['temperature'] = {
                    'old': old_temp,
                    'new': new_temp,
                    'delta': adjustments['temperature']
                }
            
            self.adjustments_made += 1
            
        except Exception as e:
            report['success'] = False
            report['error'] = str(e)
        
        return report


def create_monitoring_dashboard_data(monitor: AARStateMonitor) -> Dict[str, Any]:
    """
    Create data structure for monitoring dashboard visualization.
    
    Args:
        monitor: AARStateMonitor instance
    
    Returns:
        Dashboard data dictionary
    """
    trajectory = monitor.get_trajectory(window=50)
    
    if not trajectory:
        return {'status': 'no_data'}
    
    # Extract time series
    steps = [s['step'] for s in trajectory]
    balance_scores = [s['balance_score'] for s in trajectory]
    coherence_scores = [s['coherence'] for s in trajectory]
    agent_mags = [s['agent_magnitude'] for s in trajectory]
    arena_mags = [s['arena_magnitude'] for s in trajectory]
    relation_mags = [s['relation_magnitude'] for s in trajectory]
    
    # Get recent alerts
    recent_alerts = monitor.alerts[-10:] if monitor.alerts else []
    
    return {
        'status': 'active',
        'current_state': monitor.get_current_state(),
        'time_series': {
            'steps': steps,
            'balance_scores': balance_scores,
            'coherence_scores': coherence_scores,
            'agent_magnitudes': agent_mags,
            'arena_magnitudes': arena_mags,
            'relation_magnitudes': relation_mags
        },
        'alerts': {
            'recent': [a.to_dict() for a in recent_alerts],
            'total_count': len(monitor.alerts),
            'by_severity': {
                'high': len([a for a in monitor.alerts if a.severity == 'high']),
                'medium': len([a for a in monitor.alerts if a.severity == 'medium']),
                'low': len([a for a in monitor.alerts if a.severity == 'low'])
            }
        },
        'stability': monitor.analyze_stability(),
        'statistics': monitor.stats
    }
