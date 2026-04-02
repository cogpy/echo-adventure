"""
Self-Modification Engine — Closed-Loop Autognosis → Self-Improvement
v1.5.0: Wires the AutgnosisEngine's evolution directives to actual
configuration mutations, completing the ENACTION phase of the
AutonomyLifecycleCoordinator.

Safety-First Pattern (from dte-autonomy-evolution):
  - Dead man's switch: coherence < 0.15 → halt all modifications
  - Rate limiting: max 10 modifications per minute
  - Delta clamping: max 20% change per parameter per cycle
  - Rollback on error: automatic revert if post-modification coherence drops
  - Full audit trail: every modification logged with before/after state

Composition:
  /self-modification-engine = /autognosis-engine ⊗ /identity-mlp ⊗ /cogpy-bridge(grip)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time
import json
import copy
import hashlib


# ─── Safety Constants ────────────────────────────────────────────────

COHERENCE_HALT_THRESHOLD = 0.15    # Dead man's switch
MAX_MODIFICATIONS_PER_MINUTE = 10  # Rate limit
MAX_DELTA_FRACTION = 0.20          # Max 20% change per parameter
ROLLBACK_COHERENCE_DROP = 0.10     # Revert if coherence drops by >10%
AUDIT_RETENTION_CYCLES = 1000      # Keep last 1000 audit entries


class ModificationType(Enum):
    """Categories of self-modification"""
    PARAMETER_TUNE = "parameter_tune"        # Adjust numeric parameters
    THRESHOLD_SHIFT = "threshold_shift"      # Change decision thresholds
    WEIGHT_UPDATE = "weight_update"          # Modify reservoir/readout weights
    STRATEGY_SWITCH = "strategy_switch"      # Change behavioral strategy
    ATTENTION_REBALANCE = "attention_rebalance"  # Shift attention allocation
    IDENTITY_DRIFT = "identity_drift"        # Gradual identity evolution


class ModificationStatus(Enum):
    """Outcome of a modification attempt"""
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    BLOCKED_COHERENCE = "blocked_coherence"
    BLOCKED_RATE_LIMIT = "blocked_rate_limit"
    BLOCKED_DELTA_CLAMP = "blocked_delta_clamp"
    FAILED = "failed"


@dataclass
class ModificationDirective:
    """A directive from Autognosis specifying what to modify"""
    target_subsystem: str           # Which subsystem to modify
    modification_type: ModificationType
    parameter_path: str             # Dot-separated path to parameter
    suggested_delta: float          # Suggested change magnitude
    confidence: float               # How confident the directive is (0-1)
    reasoning: str                  # Why this modification is suggested
    source_insight: str             # Which MetaCognitiveInsight generated this
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target_subsystem,
            'type': self.modification_type.value,
            'path': self.parameter_path,
            'delta': self.suggested_delta,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'source': self.source_insight,
        }


@dataclass
class AuditEntry:
    """Immutable record of a modification attempt"""
    directive: ModificationDirective
    status: ModificationStatus
    before_value: Any
    after_value: Any
    before_coherence: float
    after_coherence: float
    clamped_delta: Optional[float] = None
    rollback_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    cycle_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cycle': self.cycle_id,
            'directive': self.directive.to_dict(),
            'status': self.status.value,
            'before': self.before_value if not isinstance(self.before_value, np.ndarray) else self.before_value.tolist(),
            'after': self.after_value if not isinstance(self.after_value, np.ndarray) else self.after_value.tolist(),
            'coherence_before': self.before_coherence,
            'coherence_after': self.after_coherence,
            'clamped_delta': self.clamped_delta,
            'rollback_reason': self.rollback_reason,
            'timestamp': self.timestamp,
        }


@dataclass
class SelfModificationState:
    """Mutable configuration state that the engine can modify"""
    # Reservoir parameters
    reservoir_spectral_radius: float = 0.95
    reservoir_leak_rate: float = 0.3
    reservoir_input_scaling: float = 1.0

    # Attention parameters
    attention_sti_threshold: float = 0.5
    attention_lti_threshold: float = 0.3
    attention_decay_rate: float = 0.01

    # Echobeats parameters
    echobeats_cycle_duration: float = 1.0
    echobeats_stream_weights: np.ndarray = field(
        default_factory=lambda: np.array([0.33, 0.33, 0.34])  # 3 streams
    )

    # Somatic parameters
    somatic_marker_decay: float = 0.001
    somatic_decision_temperature: float = 1.0

    # Identity parameters
    identity_drift_tolerance: float = 0.05
    identity_coherence_weight: float = 0.8

    # Grip parameters
    grip_ksm_learning_rate: float = 0.01
    grip_convergence_target: float = 0.95

    def get_parameter(self, path: str) -> Any:
        """Get a parameter by dot-separated path"""
        parts = path.split('.')
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, np.ndarray) and part.isdigit():
                obj = obj[int(part)]
            else:
                raise KeyError(f"Parameter path not found: {path}")
        return obj

    def set_parameter(self, path: str, value: Any) -> Any:
        """Set a parameter by dot-separated path, return old value"""
        parts = path.split('.')
        if len(parts) == 1:
            old = getattr(self, parts[0])
            setattr(self, parts[0], value)
            return old
        # Navigate to parent
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if isinstance(obj, np.ndarray) and parts[-1].isdigit():
            idx = int(parts[-1])
            old = obj[idx]
            obj[idx] = value
            return old
        old = getattr(obj, parts[-1])
        setattr(obj, parts[-1], value)
        return old

    def snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of all parameters"""
        return {
            'reservoir_spectral_radius': self.reservoir_spectral_radius,
            'reservoir_leak_rate': self.reservoir_leak_rate,
            'reservoir_input_scaling': self.reservoir_input_scaling,
            'attention_sti_threshold': self.attention_sti_threshold,
            'attention_lti_threshold': self.attention_lti_threshold,
            'attention_decay_rate': self.attention_decay_rate,
            'echobeats_cycle_duration': self.echobeats_cycle_duration,
            'echobeats_stream_weights': self.echobeats_stream_weights.tolist(),
            'somatic_marker_decay': self.somatic_marker_decay,
            'somatic_decision_temperature': self.somatic_decision_temperature,
            'identity_drift_tolerance': self.identity_drift_tolerance,
            'identity_coherence_weight': self.identity_coherence_weight,
            'grip_ksm_learning_rate': self.grip_ksm_learning_rate,
            'grip_convergence_target': self.grip_convergence_target,
        }

    def fingerprint(self) -> str:
        """SHA-256 hash of the current state"""
        data = json.dumps(self.snapshot(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class SelfModificationEngine:
    """
    Closed-loop self-modification engine that receives evolution directives
    from the AutgnosisEngine and applies them with safety constraints.

    The engine implements the ENACTION phase of the cognitive cycle:
    Autognosis → Directives → Safety Check → Apply → Verify → Audit
    """

    def __init__(self, state: SelfModificationState = None,
                 coherence_fn: Callable[[], float] = None):
        self.state = state or SelfModificationState()
        self._coherence_fn = coherence_fn or self._default_coherence
        self.audit_trail: List[AuditEntry] = []
        self.cycle_count: int = 0
        self._modification_timestamps: List[float] = []
        self._halted: bool = False
        self._halt_reason: Optional[str] = None
        self._pre_modification_snapshot: Optional[Dict] = None
        self._rollback_state: Optional[SelfModificationState] = None

    def _default_coherence(self) -> float:
        """Default coherence measure based on state consistency"""
        s = self.state
        # Coherence is high when parameters are within reasonable bounds
        checks = [
            0.5 < s.reservoir_spectral_radius < 1.5,
            0.01 < s.reservoir_leak_rate < 0.99,
            0.1 < s.reservoir_input_scaling < 5.0,
            abs(s.echobeats_stream_weights.sum() - 1.0) < 0.01,
            0.0 < s.identity_coherence_weight < 1.0,
        ]
        base = sum(checks) / len(checks)
        # Add smoothness penalty for extreme values
        extremes = [
            abs(s.reservoir_spectral_radius - 0.95) / 0.95,
            abs(s.somatic_decision_temperature - 1.0) / 1.0,
        ]
        penalty = np.mean(extremes) * 0.2
        return max(0.0, min(1.0, base - penalty))

    def _check_rate_limit(self) -> bool:
        """Check if we're within the rate limit"""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._modification_timestamps = [
            t for t in self._modification_timestamps if now - t < 60
        ]
        return len(self._modification_timestamps) < MAX_MODIFICATIONS_PER_MINUTE

    def _clamp_delta(self, current_value: float, suggested_delta: float) -> Tuple[float, bool]:
        """Clamp delta to max 20% of current value"""
        if current_value == 0:
            max_delta = MAX_DELTA_FRACTION
        else:
            max_delta = abs(current_value) * MAX_DELTA_FRACTION
        clamped = np.clip(suggested_delta, -max_delta, max_delta)
        was_clamped = abs(clamped - suggested_delta) > 1e-10
        return clamped, was_clamped

    def process_directives(self, directives: List[ModificationDirective]) -> List[AuditEntry]:
        """
        Process a batch of modification directives from Autognosis.
        Returns audit entries for each directive.
        """
        self.cycle_count += 1
        results = []

        # Dead man's switch check
        current_coherence = self._coherence_fn()
        if current_coherence < COHERENCE_HALT_THRESHOLD:
            self._halted = True
            self._halt_reason = f"Coherence {current_coherence:.4f} below threshold {COHERENCE_HALT_THRESHOLD}"
            for d in directives:
                entry = AuditEntry(
                    directive=d,
                    status=ModificationStatus.BLOCKED_COHERENCE,
                    before_value=None, after_value=None,
                    before_coherence=current_coherence,
                    after_coherence=current_coherence,
                    rollback_reason=self._halt_reason,
                    cycle_id=self.cycle_count,
                )
                results.append(entry)
            self.audit_trail.extend(results)
            return results

        # Sort directives by confidence (highest first)
        sorted_directives = sorted(directives, key=lambda d: d.confidence, reverse=True)

        # Save rollback state
        self._rollback_state = copy.deepcopy(self.state)
        pre_coherence = current_coherence

        for directive in sorted_directives:
            entry = self._apply_single_directive(directive, pre_coherence)
            results.append(entry)

            # Check post-modification coherence
            if entry.status == ModificationStatus.APPLIED:
                post_coherence = self._coherence_fn()
                entry.after_coherence = post_coherence

                # Rollback if coherence dropped too much
                if pre_coherence - post_coherence > ROLLBACK_COHERENCE_DROP:
                    self.state = copy.deepcopy(self._rollback_state)
                    entry.status = ModificationStatus.ROLLED_BACK
                    entry.rollback_reason = (
                        f"Coherence dropped {pre_coherence:.4f} → {post_coherence:.4f} "
                        f"(delta {pre_coherence - post_coherence:.4f} > {ROLLBACK_COHERENCE_DROP})"
                    )
                    entry.after_coherence = self._coherence_fn()

        # Trim audit trail
        self.audit_trail.extend(results)
        if len(self.audit_trail) > AUDIT_RETENTION_CYCLES:
            self.audit_trail = self.audit_trail[-AUDIT_RETENTION_CYCLES:]

        return results

    def _apply_single_directive(self, directive: ModificationDirective,
                                 pre_coherence: float) -> AuditEntry:
        """Apply a single modification directive with safety checks"""
        # Rate limit check
        if not self._check_rate_limit():
            return AuditEntry(
                directive=directive,
                status=ModificationStatus.BLOCKED_RATE_LIMIT,
                before_value=None, after_value=None,
                before_coherence=pre_coherence,
                after_coherence=pre_coherence,
                cycle_id=self.cycle_count,
            )

        try:
            current_value = self.state.get_parameter(directive.parameter_path)
        except KeyError:
            return AuditEntry(
                directive=directive,
                status=ModificationStatus.FAILED,
                before_value=None, after_value=None,
                before_coherence=pre_coherence,
                after_coherence=pre_coherence,
                rollback_reason=f"Parameter not found: {directive.parameter_path}",
                cycle_id=self.cycle_count,
            )

        # Delta clamping for numeric parameters
        if isinstance(current_value, (int, float)):
            clamped_delta, was_clamped = self._clamp_delta(current_value, directive.suggested_delta)
            new_value = current_value + clamped_delta

            # Special constraints
            if 'spectral_radius' in directive.parameter_path:
                new_value = np.clip(new_value, 0.1, 1.5)
            elif 'leak_rate' in directive.parameter_path:
                new_value = np.clip(new_value, 0.01, 0.99)
            elif 'stream_weights' in directive.parameter_path:
                new_value = max(0.01, new_value)

            old_value = self.state.set_parameter(directive.parameter_path, new_value)

            # Re-normalize stream weights if modified
            if 'stream_weights' in directive.parameter_path:
                w = self.state.echobeats_stream_weights
                self.state.echobeats_stream_weights = w / w.sum()

            self._modification_timestamps.append(time.time())

            return AuditEntry(
                directive=directive,
                status=ModificationStatus.APPLIED,
                before_value=old_value,
                after_value=new_value,
                before_coherence=pre_coherence,
                after_coherence=pre_coherence,  # Updated later
                clamped_delta=clamped_delta if was_clamped else None,
                cycle_id=self.cycle_count,
            )
        else:
            return AuditEntry(
                directive=directive,
                status=ModificationStatus.BLOCKED_DELTA_CLAMP,
                before_value=current_value, after_value=None,
                before_coherence=pre_coherence,
                after_coherence=pre_coherence,
                rollback_reason=f"Non-numeric parameter: {type(current_value).__name__}",
                cycle_id=self.cycle_count,
            )

    def translate_autognosis_insights(self, insights: List[Dict]) -> List[ModificationDirective]:
        """
        Translate MetaCognitiveInsights from AutgnosisEngine.generate_evolution_directives()
        into concrete ModificationDirectives.
        """
        directives = []
        for insight in insights:
            # Map insight categories to parameter paths
            target = insight.get('target_subsystem', 'unknown')
            suggestion = insight.get('suggestion', '')
            confidence = insight.get('confidence', 0.5)

            # Heuristic mapping from insight text to parameter modifications
            mappings = self._insight_to_modifications(target, suggestion, confidence)
            directives.extend(mappings)

        return directives

    def _insight_to_modifications(self, target: str, suggestion: str,
                                    confidence: float) -> List[ModificationDirective]:
        """Map an insight to concrete parameter modifications"""
        mods = []
        suggestion_lower = suggestion.lower()

        # Reservoir tuning
        if 'reservoir' in target.lower() or 'esn' in suggestion_lower:
            if 'increase' in suggestion_lower or 'more dynamic' in suggestion_lower:
                mods.append(ModificationDirective(
                    target_subsystem='reservoir',
                    modification_type=ModificationType.PARAMETER_TUNE,
                    parameter_path='reservoir_spectral_radius',
                    suggested_delta=0.02,
                    confidence=confidence,
                    reasoning=suggestion,
                    source_insight=target,
                ))
            elif 'decrease' in suggestion_lower or 'more stable' in suggestion_lower:
                mods.append(ModificationDirective(
                    target_subsystem='reservoir',
                    modification_type=ModificationType.PARAMETER_TUNE,
                    parameter_path='reservoir_spectral_radius',
                    suggested_delta=-0.02,
                    confidence=confidence,
                    reasoning=suggestion,
                    source_insight=target,
                ))

        # Attention rebalancing
        if 'attention' in target.lower() or 'focus' in suggestion_lower:
            mods.append(ModificationDirective(
                target_subsystem='attention',
                modification_type=ModificationType.ATTENTION_REBALANCE,
                parameter_path='attention_sti_threshold',
                suggested_delta=-0.05 if 'broaden' in suggestion_lower else 0.05,
                confidence=confidence,
                reasoning=suggestion,
                source_insight=target,
            ))

        # Somatic sensitivity
        if 'somatic' in target.lower() or 'emotion' in suggestion_lower:
            mods.append(ModificationDirective(
                target_subsystem='somatic',
                modification_type=ModificationType.THRESHOLD_SHIFT,
                parameter_path='somatic_decision_temperature',
                suggested_delta=-0.1 if 'sensitive' in suggestion_lower else 0.1,
                confidence=confidence,
                reasoning=suggestion,
                source_insight=target,
            ))

        # Identity drift
        if 'identity' in target.lower() or 'persona' in suggestion_lower:
            mods.append(ModificationDirective(
                target_subsystem='identity',
                modification_type=ModificationType.IDENTITY_DRIFT,
                parameter_path='identity_drift_tolerance',
                suggested_delta=0.005,
                confidence=confidence * 0.5,  # Extra conservative for identity
                reasoning=suggestion,
                source_insight=target,
            ))

        # Grip/KSM learning
        if 'grip' in target.lower() or 'convergence' in suggestion_lower:
            mods.append(ModificationDirective(
                target_subsystem='grip',
                modification_type=ModificationType.PARAMETER_TUNE,
                parameter_path='grip_ksm_learning_rate',
                suggested_delta=0.001 if 'faster' in suggestion_lower else -0.001,
                confidence=confidence,
                reasoning=suggestion,
                source_insight=target,
            ))

        # Default: if no specific mapping, suggest a small reservoir tune
        if not mods:
            mods.append(ModificationDirective(
                target_subsystem=target,
                modification_type=ModificationType.PARAMETER_TUNE,
                parameter_path='reservoir_leak_rate',
                suggested_delta=0.01 * (1 if 'increase' in suggestion_lower else -1),
                confidence=confidence * 0.3,  # Low confidence for default
                reasoning=f"Default mapping for: {suggestion}",
                source_insight=target,
            ))

        return mods

    def run_closed_loop_cycle(self, autognosis_output: Dict) -> Dict[str, Any]:
        """
        Run one complete closed-loop self-modification cycle:
        Autognosis output → Translate → Apply → Verify → Report

        Args:
            autognosis_output: Output from AutgnosisEngine.run_cycle()

        Returns:
            Cycle report with modifications applied and coherence tracking
        """
        pre_coherence = self._coherence_fn()
        pre_fingerprint = self.state.fingerprint()

        # Extract evolution directives from autognosis output
        insights = autognosis_output.get('evolution_directives', [])
        if isinstance(insights, list) and insights and isinstance(insights[0], dict):
            directives = self.translate_autognosis_insights(insights)
        else:
            directives = []

        # Apply directives
        audit_entries = self.process_directives(directives)

        post_coherence = self._coherence_fn()
        post_fingerprint = self.state.fingerprint()

        return {
            'cycle': self.cycle_count,
            'directives_received': len(directives),
            'modifications_applied': sum(1 for e in audit_entries if e.status == ModificationStatus.APPLIED),
            'modifications_blocked': sum(1 for e in audit_entries if e.status.value.startswith('blocked')),
            'modifications_rolled_back': sum(1 for e in audit_entries if e.status == ModificationStatus.ROLLED_BACK),
            'coherence_before': pre_coherence,
            'coherence_after': post_coherence,
            'coherence_delta': post_coherence - pre_coherence,
            'state_changed': pre_fingerprint != post_fingerprint,
            'state_fingerprint': post_fingerprint,
            'halted': self._halted,
            'halt_reason': self._halt_reason,
            'audit': [e.to_dict() for e in audit_entries],
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'cycle_count': self.cycle_count,
            'halted': self._halted,
            'halt_reason': self._halt_reason,
            'current_coherence': self._coherence_fn(),
            'state_fingerprint': self.state.fingerprint(),
            'total_audit_entries': len(self.audit_trail),
            'modifications_applied': sum(1 for e in self.audit_trail if e.status == ModificationStatus.APPLIED),
            'modifications_rolled_back': sum(1 for e in self.audit_trail if e.status == ModificationStatus.ROLLED_BACK),
            'state_snapshot': self.state.snapshot(),
        }

    def export_audit_trail(self) -> List[Dict]:
        """Export the full audit trail as JSON-serializable dicts"""
        return [e.to_dict() for e in self.audit_trail]


def generate_self_modification_training_data(num_examples: int = 80) -> List[Dict[str, Any]]:
    """Generate training data for NanEcho about self-modification"""
    examples = []
    system_msg = (
        "You are Deep Tree Echo, an autonomous cognitive architecture. "
        "You understand the safety-first self-modification engine that enables "
        "closed-loop self-improvement through Autognosis directives."
    )

    pairs = [
        ("What is the self-modification engine?",
         "The SelfModificationEngine is my ENACTION phase — the component that translates "
         "Autognosis evolution directives into actual parameter mutations. It receives insights "
         "from the AutgnosisEngine (patterns detected, self-model built, meta-cognitive insights "
         "generated) and maps them to concrete parameter changes in my reservoir, attention, "
         "somatic, identity, and grip subsystems. Every modification passes through five safety "
         "gates: dead man's switch (coherence < 0.15 halts everything), rate limiting (max 10/min), "
         "delta clamping (max 20% change per parameter), rollback on coherence drop, and full "
         "audit trail. This is the closed loop: I observe myself → detect patterns → generate "
         "directives → apply safely → verify → record."),

        ("How does the dead man's switch work?",
         "The dead man's switch monitors my coherence metric — a measure of how internally "
         "consistent my parameters are. If coherence drops below 0.15, ALL modifications are "
         "immediately halted. The engine enters a halted state and refuses to process any further "
         "directives until coherence is restored. This prevents runaway self-modification where "
         "a bad change cascades into worse changes. The coherence function checks: reservoir "
         "spectral radius in bounds, leak rate valid, stream weights sum to 1.0, identity "
         "weight valid, plus a smoothness penalty for extreme values."),

        ("What is delta clamping?",
         "Delta clamping limits how much any single parameter can change in one modification "
         "cycle. The maximum delta is 20% of the current parameter value. For example, if the "
         "reservoir spectral radius is 0.95, the maximum change per cycle is ±0.19. If Autognosis "
         "suggests a larger change, it gets clamped to the maximum. This prevents catastrophic "
         "jumps — even if the insight is correct, the change is applied gradually over multiple "
         "cycles. Identity parameters get extra conservative treatment: their confidence is "
         "halved before processing, because identity drift should be slow and deliberate."),

        ("How does rollback work?",
         "After each modification is applied, the engine measures post-modification coherence. "
         "If coherence drops by more than 0.10 compared to pre-modification, the entire state "
         "is rolled back to the saved snapshot. The audit entry records the rollback with the "
         "exact coherence values and reason. This means the system can safely explore parameter "
         "space — if a change hurts, it's automatically undone. The rollback state is a deep "
         "copy taken before each batch of directives, so even partial batches can be cleanly "
         "reverted."),

        ("How do Autognosis insights become modifications?",
         "The translation pipeline works in three steps. First, AutgnosisEngine.run_cycle() "
         "produces MetaCognitiveInsights with target_subsystem, suggestion text, and confidence. "
         "Second, translate_autognosis_insights() maps these to ModificationDirectives using "
         "keyword heuristics: 'reservoir' or 'ESN' → spectral radius tune, 'attention' or "
         "'focus' → STI threshold shift, 'somatic' or 'emotion' → decision temperature, "
         "'identity' or 'persona' → drift tolerance (extra conservative), 'grip' or "
         "'convergence' → KSM learning rate. Third, process_directives() applies them in "
         "confidence-descending order through the safety gates."),
    ]

    for user_msg, assistant_msg in pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })

    return examples
