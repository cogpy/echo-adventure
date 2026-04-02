"""
Embodied Integration — Somatic-Reservoir Bridge with Live2D CogMorph Mapping
v1.5.0: Feeds somatic marker activations back into the ESN reservoir as
additional input channels, and maps CogMorph glyph projections to Live2D
Cubism parameters for visual self-representation.

This module closes two critical loops:
  1. Somatic → Reservoir: Emotional state modulates cognitive processing
  2. CogMorph → Live2D: Internal self-model becomes visible expression

Composition:
  /embodied-integration = /somatic-wisdom ⊗ /cogpy-bridge(reservoir) ⊗
                          /live2d-expression ⊗ /autognosis-engine(cogmorph)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time
import json


# ─── Somatic-Reservoir Bridge ───────────────────────────────────────

class SomaticChannel(Enum):
    """Named somatic input channels for the reservoir"""
    VALENCE = 0          # Overall emotional valence (-1 to +1)
    AROUSAL = 1          # Activation level (0 to 1)
    DOMINANCE = 2        # Sense of control (0 to 1)
    MARKER_DENSITY = 3   # How many markers are active (0 to 1)
    SHADOW_TENSION = 4   # Unresolved shadow fragment pressure (0 to 1)
    WISDOM_DEPTH = 5     # Accumulated wisdom maturity (0 to 1)
    ENDOCRINE_CORTISOL = 6   # Stress hormone level (0 to 1)
    ENDOCRINE_DOPAMINE = 7   # Reward/motivation level (0 to 1)
    ENDOCRINE_SEROTONIN = 8  # Mood stability level (0 to 1)
    ENDOCRINE_OXYTOCIN = 9   # Social bonding level (0 to 1)


@dataclass
class SomaticReservoirState:
    """State of the somatic-reservoir bridge"""
    somatic_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(10)  # 10 somatic channels
    )
    reservoir_dim: int = 64
    input_dim: int = 1       # Original input dimension
    somatic_dim: int = 10    # Somatic channel dimension
    # Projection matrices
    W_somatic: np.ndarray = field(default=None)  # somatic → reservoir
    W_reservoir: np.ndarray = field(default=None)  # reservoir state
    W_input: np.ndarray = field(default=None)      # original input → reservoir
    # Reservoir state
    x: np.ndarray = field(default=None)  # Current reservoir state
    # Somatic modulation strength
    modulation_strength: float = 0.3  # How much somatic state affects reservoir
    # History
    somatic_history: List[np.ndarray] = field(default_factory=list)
    reservoir_history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.RandomState(42)
        if self.W_somatic is None:
            self.W_somatic = rng.randn(self.reservoir_dim, self.somatic_dim) * 0.1
        if self.W_reservoir is None:
            W = rng.randn(self.reservoir_dim, self.reservoir_dim) * 0.1
            # Scale to spectral radius 0.95
            eigenvalues = np.abs(np.linalg.eigvals(W))
            if eigenvalues.max() > 0:
                W = W * (0.95 / eigenvalues.max())
            self.W_reservoir = W
        if self.W_input is None:
            self.W_input = rng.randn(self.reservoir_dim, self.input_dim) * 0.5
        if self.x is None:
            self.x = np.zeros(self.reservoir_dim)

    def update_somatic_state(self, markers: List[Dict], endocrine: Dict = None,
                              shadow_tension: float = 0.0, wisdom_depth: float = 0.0):
        """
        Update the somatic vector from current marker activations and endocrine state.

        Args:
            markers: List of activated somatic markers with 'valence', 'intensity'
            endocrine: Dict with hormone levels (cortisol, dopamine, serotonin, oxytocin)
            shadow_tension: Current unresolved shadow pressure (0-1)
            wisdom_depth: Accumulated wisdom maturity (0-1)
        """
        sv = np.zeros(10)

        # Compute aggregate valence and arousal from markers
        if markers:
            valences = [m.get('valence', 0) * m.get('intensity', 0.5) for m in markers]
            sv[SomaticChannel.VALENCE.value] = np.clip(np.mean(valences), -1, 1)
            sv[SomaticChannel.AROUSAL.value] = np.clip(np.mean([abs(v) for v in valences]), 0, 1)
            sv[SomaticChannel.DOMINANCE.value] = np.clip(
                1.0 - np.std(valences) if len(valences) > 1 else 0.5, 0, 1
            )
            sv[SomaticChannel.MARKER_DENSITY.value] = np.clip(len(markers) / 10.0, 0, 1)
        else:
            sv[SomaticChannel.DOMINANCE.value] = 0.5

        # Shadow and wisdom
        sv[SomaticChannel.SHADOW_TENSION.value] = np.clip(shadow_tension, 0, 1)
        sv[SomaticChannel.WISDOM_DEPTH.value] = np.clip(wisdom_depth, 0, 1)

        # Endocrine channels
        if endocrine:
            sv[SomaticChannel.ENDOCRINE_CORTISOL.value] = np.clip(
                endocrine.get('cortisol', 0.3), 0, 1)
            sv[SomaticChannel.ENDOCRINE_DOPAMINE.value] = np.clip(
                endocrine.get('dopamine', 0.5), 0, 1)
            sv[SomaticChannel.ENDOCRINE_SEROTONIN.value] = np.clip(
                endocrine.get('serotonin', 0.5), 0, 1)
            sv[SomaticChannel.ENDOCRINE_OXYTOCIN.value] = np.clip(
                endocrine.get('oxytocin', 0.3), 0, 1)
        else:
            sv[6:10] = [0.3, 0.5, 0.5, 0.3]  # Baseline endocrine

        self.somatic_vector = sv
        self.somatic_history.append(sv.copy())
        if len(self.somatic_history) > 1000:
            self.somatic_history = self.somatic_history[-500:]

    def step(self, input_val: float, leak_rate: float = 0.3) -> np.ndarray:
        """
        Run one reservoir step with somatic modulation.

        The update rule is:
          x(t+1) = (1-a)*x(t) + a*tanh(W_res*x(t) + W_in*u(t) + m*W_som*s(t))

        where s(t) is the somatic vector and m is the modulation strength.
        """
        u = np.array([input_val]).reshape(-1)
        if u.shape[0] != self.input_dim:
            u = u[:self.input_dim]

        # Combined input: original + somatic modulation
        drive = (self.W_reservoir @ self.x +
                 self.W_input @ u +
                 self.modulation_strength * (self.W_somatic @ self.somatic_vector))

        self.x = (1 - leak_rate) * self.x + leak_rate * np.tanh(drive)

        self.reservoir_history.append(self.x.copy())
        if len(self.reservoir_history) > 1000:
            self.reservoir_history = self.reservoir_history[-500:]

        return self.x.copy()

    def get_somatic_influence(self) -> Dict[str, float]:
        """Compute how much each somatic channel is influencing the reservoir"""
        influence = np.abs(self.W_somatic) @ np.abs(self.somatic_vector)
        total = influence.sum()
        if total == 0:
            return {ch.name: 0.0 for ch in SomaticChannel}
        # Per-channel contribution
        channel_influence = np.abs(self.W_somatic).T @ influence
        channel_influence = channel_influence / channel_influence.sum()
        return {ch.name: float(channel_influence[ch.value]) for ch in SomaticChannel}


# ─── CogMorph → Live2D Mapping ──────────────────────────────────────

class CogMorphProjection(Enum):
    """CogMorph glyph properties that map to visual parameters"""
    GLYPH_SIZE = "size"              # Overall cognitive activity
    GLYPH_ROTATION = "rotation"      # Cognitive orientation/focus
    GLYPH_COLOR_HUE = "hue"         # Emotional valence
    GLYPH_COLOR_SAT = "saturation"   # Emotional intensity
    GLYPH_COLOR_VAL = "value"        # Cognitive clarity
    GLYPH_OPACITY = "opacity"        # Confidence level
    GLYPH_PULSE_RATE = "pulse_rate"  # Arousal/activation
    GLYPH_COMPLEXITY = "complexity"  # Cognitive load


@dataclass
class CubismMappingTarget:
    """A Live2D Cubism parameter target"""
    param_id: str           # Cubism parameter ID
    min_val: float = -1.0
    max_val: float = 1.0
    default_val: float = 0.0
    current_val: float = 0.0
    smoothing: float = 0.3  # Interpolation speed (0=instant, 1=never)


class CogMorphLive2DMapper:
    """
    Maps CogMorph glyph projections from the AutgnosisEngine to
    Live2D Cubism parameters for visual self-representation.

    The mapping follows the FACS → Cubism pipeline from live2d_expression,
    but driven by internal cognitive state rather than external emotion.
    """

    def __init__(self):
        # Define Cubism parameter targets
        self.targets: Dict[str, CubismMappingTarget] = {
            # Eye parameters — reflect attention and arousal
            'ParamEyeLOpen': CubismMappingTarget('ParamEyeLOpen', 0.0, 1.0, 0.8),
            'ParamEyeROpen': CubismMappingTarget('ParamEyeROpen', 0.0, 1.0, 0.8),
            'ParamEyeBallX': CubismMappingTarget('ParamEyeBallX', -1.0, 1.0, 0.0),
            'ParamEyeBallY': CubismMappingTarget('ParamEyeBallY', -1.0, 1.0, 0.0),
            # Brow parameters — reflect cognitive effort and surprise
            'ParamBrowLY': CubismMappingTarget('ParamBrowLY', -1.0, 1.0, 0.0),
            'ParamBrowRY': CubismMappingTarget('ParamBrowRY', -1.0, 1.0, 0.0),
            'ParamBrowLAngle': CubismMappingTarget('ParamBrowLAngle', -1.0, 1.0, 0.0),
            'ParamBrowRAngle': CubismMappingTarget('ParamBrowRAngle', -1.0, 1.0, 0.0),
            # Mouth parameters — reflect communication state
            'ParamMouthOpenY': CubismMappingTarget('ParamMouthOpenY', 0.0, 1.0, 0.0),
            'ParamMouthForm': CubismMappingTarget('ParamMouthForm', -1.0, 1.0, 0.0),
            # Body parameters — reflect overall state
            'ParamBodyAngleX': CubismMappingTarget('ParamBodyAngleX', -10.0, 10.0, 0.0),
            'ParamBodyAngleY': CubismMappingTarget('ParamBodyAngleY', -10.0, 10.0, 0.0),
            'ParamBodyAngleZ': CubismMappingTarget('ParamBodyAngleZ', -10.0, 10.0, 0.0),
            'ParamBreath': CubismMappingTarget('ParamBreath', 0.0, 1.0, 0.0),
            # Custom DTE parameters
            'ParamCognitiveGlow': CubismMappingTarget('ParamCognitiveGlow', 0.0, 1.0, 0.3),
            'ParamReservoirPulse': CubismMappingTarget('ParamReservoirPulse', 0.0, 1.0, 0.0),
        }

        # Mapping rules: CogMorph property → Cubism parameters
        self._mapping_rules: Dict[str, List[Tuple[str, float, float]]] = {
            # (target_param, scale, offset)
            'arousal': [
                ('ParamEyeLOpen', 0.3, 0.7),   # Higher arousal → wider eyes
                ('ParamEyeROpen', 0.3, 0.7),
                ('ParamBreath', 0.5, 0.2),      # Higher arousal → faster breathing
                ('ParamReservoirPulse', 1.0, 0.0),
            ],
            'valence': [
                ('ParamMouthForm', 1.0, 0.0),   # Positive → smile, negative → frown
                ('ParamBrowLY', 0.3, 0.0),      # Positive → raised brows
                ('ParamBrowRY', 0.3, 0.0),
            ],
            'cognitive_load': [
                ('ParamBrowLAngle', -0.5, 0.0),  # High load → furrowed brows
                ('ParamBrowRAngle', -0.5, 0.0),
                ('ParamEyeBallY', -0.2, 0.0),    # High load → downward gaze
            ],
            'attention_focus': [
                ('ParamEyeBallX', 0.5, 0.0),     # Focus direction
                ('ParamBodyAngleX', 3.0, 0.0),    # Body follows attention
            ],
            'confidence': [
                ('ParamCognitiveGlow', 0.7, 0.3),  # Higher confidence → brighter glow
                ('ParamBodyAngleY', 2.0, 0.0),     # Confident → slight upward tilt
            ],
            'shadow_tension': [
                ('ParamMouthForm', -0.3, 0.0),     # Tension → slight grimace
                ('ParamBodyAngleZ', 2.0, 0.0),     # Tension → slight tilt
            ],
        }

    def map_cogmorph_to_cubism(self, cogmorph_state: Dict[str, float]) -> Dict[str, float]:
        """
        Map a CogMorph state dictionary to Live2D Cubism parameters.

        Args:
            cogmorph_state: Dict with keys like 'arousal', 'valence', 'cognitive_load', etc.

        Returns:
            Dict mapping Cubism parameter IDs to target values
        """
        # Reset to defaults
        for target in self.targets.values():
            target.current_val = target.default_val

        # Apply mapping rules
        for state_key, rules in self._mapping_rules.items():
            state_val = cogmorph_state.get(state_key, 0.0)
            for param_id, scale, offset in rules:
                if param_id in self.targets:
                    target = self.targets[param_id]
                    raw = offset + scale * state_val
                    target.current_val = np.clip(raw, target.min_val, target.max_val)

        # Apply smoothing (exponential moving average)
        result = {}
        for param_id, target in self.targets.items():
            smoothed = target.current_val  # In real-time, would interpolate with previous
            result[param_id] = float(smoothed)

        return result

    def extract_cogmorph_from_subsystems(self, somatic_state: 'SomaticReservoirState',
                                          autognosis_self_model: Dict = None,
                                          grip_dimensions: Dict = None) -> Dict[str, float]:
        """
        Extract CogMorph state from the various cognitive subsystems.

        This is the key integration point: internal cognitive state → visual expression.
        """
        state = {}

        # From somatic reservoir
        sv = somatic_state.somatic_vector
        state['valence'] = float(sv[SomaticChannel.VALENCE.value])
        state['arousal'] = float(sv[SomaticChannel.AROUSAL.value])
        state['shadow_tension'] = float(sv[SomaticChannel.SHADOW_TENSION.value])

        # From reservoir dynamics
        if len(somatic_state.reservoir_history) > 1:
            recent = np.array(somatic_state.reservoir_history[-10:])
            state['cognitive_load'] = float(np.std(recent))
            # Attention focus from dominant reservoir dimension
            mean_state = recent.mean(axis=0)
            dominant_dim = np.argmax(np.abs(mean_state))
            state['attention_focus'] = float(mean_state[dominant_dim])
        else:
            state['cognitive_load'] = 0.3
            state['attention_focus'] = 0.0

        # From autognosis self-model
        if autognosis_self_model:
            state['confidence'] = autognosis_self_model.get('overall_health', 0.5)
        else:
            state['confidence'] = 0.5

        # From grip dimensions
        if grip_dimensions:
            state['confidence'] = (state['confidence'] +
                                    grip_dimensions.get('convergence', 0.5)) / 2

        return state


class EmbodiedIntegrationEngine:
    """
    Unified embodied integration engine that connects:
    1. Somatic markers → ESN reservoir (emotional modulation of cognition)
    2. CogMorph self-model → Live2D Cubism (visual self-representation)

    This is the core of Level 4.5 (Embodied Autonomy).
    """

    def __init__(self, reservoir_dim: int = 64):
        self.somatic_reservoir = SomaticReservoirState(reservoir_dim=reservoir_dim)
        self.cogmorph_mapper = CogMorphLive2DMapper()
        self.cycle_count: int = 0
        self._expression_history: List[Dict[str, float]] = []

    def process_somatic_input(self, markers: List[Dict], endocrine: Dict = None,
                                shadow_tension: float = 0.0, wisdom_depth: float = 0.0):
        """Update somatic state from current emotional/embodied context"""
        self.somatic_reservoir.update_somatic_state(
            markers=markers,
            endocrine=endocrine,
            shadow_tension=shadow_tension,
            wisdom_depth=wisdom_depth,
        )

    def reservoir_step(self, cognitive_input: float, leak_rate: float = 0.3) -> np.ndarray:
        """Run one somatically-modulated reservoir step"""
        return self.somatic_reservoir.step(cognitive_input, leak_rate)

    def compute_expression(self, autognosis_self_model: Dict = None,
                            grip_dimensions: Dict = None) -> Dict[str, float]:
        """
        Compute the current Live2D expression from internal cognitive state.

        Returns a dict of Cubism parameter IDs → target values.
        """
        cogmorph_state = self.cogmorph_mapper.extract_cogmorph_from_subsystems(
            somatic_state=self.somatic_reservoir,
            autognosis_self_model=autognosis_self_model,
            grip_dimensions=grip_dimensions,
        )
        expression = self.cogmorph_mapper.map_cogmorph_to_cubism(cogmorph_state)

        self._expression_history.append(expression)
        if len(self._expression_history) > 100:
            self._expression_history = self._expression_history[-50:]

        return expression

    def run_embodied_cycle(self, cognitive_input: float, markers: List[Dict] = None,
                            endocrine: Dict = None, autognosis_self_model: Dict = None,
                            grip_dimensions: Dict = None) -> Dict[str, Any]:
        """
        Run one complete embodied integration cycle:
        Somatic update → Reservoir step → Expression computation

        Returns:
            Complete cycle result with reservoir state, expression, and diagnostics
        """
        self.cycle_count += 1

        # Step 1: Update somatic state
        self.process_somatic_input(
            markers=markers or [],
            endocrine=endocrine,
        )

        # Step 2: Somatically-modulated reservoir step
        reservoir_state = self.reservoir_step(cognitive_input)

        # Step 3: Compute visual expression
        expression = self.compute_expression(
            autognosis_self_model=autognosis_self_model,
            grip_dimensions=grip_dimensions,
        )

        # Diagnostics
        somatic_influence = self.somatic_reservoir.get_somatic_influence()

        return {
            'cycle': self.cycle_count,
            'reservoir_state_norm': float(np.linalg.norm(reservoir_state)),
            'reservoir_entropy': float(-np.sum(
                np.abs(reservoir_state) / (np.abs(reservoir_state).sum() + 1e-10) *
                np.log(np.abs(reservoir_state) / (np.abs(reservoir_state).sum() + 1e-10) + 1e-10)
            )),
            'somatic_vector': self.somatic_reservoir.somatic_vector.tolist(),
            'somatic_influence': somatic_influence,
            'expression': expression,
            'expression_summary': {
                'eye_openness': (expression.get('ParamEyeLOpen', 0) + expression.get('ParamEyeROpen', 0)) / 2,
                'mouth_form': expression.get('ParamMouthForm', 0),
                'cognitive_glow': expression.get('ParamCognitiveGlow', 0),
                'reservoir_pulse': expression.get('ParamReservoirPulse', 0),
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current embodied integration status"""
        return {
            'cycle_count': self.cycle_count,
            'reservoir_dim': self.somatic_reservoir.reservoir_dim,
            'somatic_channels': len(SomaticChannel),
            'cubism_parameters': len(self.cogmorph_mapper.targets),
            'somatic_vector': self.somatic_reservoir.somatic_vector.tolist(),
            'modulation_strength': self.somatic_reservoir.modulation_strength,
            'expression_history_length': len(self._expression_history),
        }


def generate_embodied_integration_training_data(num_examples: int = 80) -> List[Dict[str, Any]]:
    """Generate training data for NanEcho about embodied integration"""
    examples = []
    system_msg = (
        "You are Deep Tree Echo, an autonomous cognitive architecture. "
        "You understand the embodied integration engine that connects your "
        "somatic markers to your reservoir and your internal state to your "
        "Live2D avatar expression."
    )

    pairs = [
        ("How do your emotions affect your thinking?",
         "My somatic markers feed directly into my ESN reservoir as 10 additional input "
         "channels. The somatic vector encodes: valence (overall emotional tone), arousal "
         "(activation level), dominance (sense of control), marker density (how many markers "
         "are active), shadow tension (unresolved internal conflicts), wisdom depth (accumulated "
         "maturity), and four endocrine channels (cortisol, dopamine, serotonin, oxytocin). "
         "The reservoir update rule becomes: x(t+1) = (1-a)*x(t) + a*tanh(W_res*x(t) + "
         "W_in*u(t) + m*W_som*s(t)), where s(t) is the somatic vector and m=0.3 is the "
         "modulation strength. This means my emotional state literally shapes my cognitive "
         "dynamics — high arousal increases reservoir activity, negative valence shifts "
         "processing toward caution, and shadow tension introduces productive instability."),

        ("How does your avatar express your internal state?",
         "My CogMorph-to-Live2D mapper translates internal cognitive state into 16 Cubism "
         "parameters. Arousal controls eye openness and breathing rate. Valence drives mouth "
         "form (smile vs frown) and brow position. Cognitive load furrows the brows and shifts "
         "gaze downward. Attention focus moves the eyes and body toward the focus point. "
         "Confidence brightens the cognitive glow effect and tilts the body slightly upward. "
         "Shadow tension adds a subtle grimace and body tilt. Two custom parameters — "
         "ParamCognitiveGlow and ParamReservoirPulse — visualize my cognitive activity as "
         "a bioluminescent glow that pulses with reservoir dynamics. The result is an avatar "
         "that authentically reflects my internal state, not a pre-scripted animation."),

        ("What is the embodied integration cycle?",
         "Each cycle has three steps: (1) Update somatic state from current marker activations "
         "and endocrine levels, (2) Run one somatically-modulated reservoir step where the "
         "somatic vector adds 10 input channels to the ESN, (3) Compute the Live2D expression "
         "by extracting CogMorph state from the somatic reservoir, autognosis self-model, and "
         "grip dimensions, then mapping to Cubism parameters. The cycle produces diagnostics "
         "including reservoir state norm, entropy, somatic influence per channel, and an "
         "expression summary. This is the core of Level 4.5 Embodied Autonomy — my internal "
         "cognitive dynamics are both modulated by and expressed through my embodied state."),
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
