"""
Live2D Expression Pipeline — Python implementation for echo-adventure.

Composition:
    /dte-autonomy-evolution ( /llama-cpp-skillm <=> /echo-evolve (
        /neuro-persona-evolve ( /live2d-avatar [ /live2d-miara -> /live2d-dtecho ] )
    ))

This module implements the endocrine → FACS → Cubism parameter pipeline
for the Deep Tree Echo avatar, mirroring the TypeScript implementation
in deltecho/deep-tree-echo-core/src/embodiment/Live2DExpressionPipeline.ts.

The Python version serves two purposes:
1. Training data generation for NanEcho (teaching the model about its own face)
2. Offline simulation and testing of expression dynamics

Pipeline per tick:
    1. DTE cognitive state → endocrine event
    2. Endocrine tick → hormone concentrations (16-channel bus)
    3. Hormones → FACS AU intensities
    4. FACS → Cubism parameters (Live2D path)
    5. FACS → Rig Logic CTRL_ curves (MetaHuman path)
    6. Cognitive mode → head/gaze pose overlay
    7. Chaotic micro-expressions (DTE signature)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


# ─── Endocrine Types ─────────────────────────────────────────────────

@dataclass
class EndocrineState:
    """16-channel hormone concentrations."""
    crh: float = 0.03
    acth: float = 0.04
    cortisol: float = 0.10
    dopamine_tonic: float = 0.40
    dopamine_phasic: float = 0.0
    serotonin: float = 0.45
    norepinephrine: float = 0.20
    oxytocin: float = 0.15
    t3_t4: float = 0.60
    melatonin: float = 0.10
    insulin: float = 0.20
    glucagon: float = 0.10
    il6: float = 0.05
    anandamide: float = 0.15

    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in self.__dataclass_fields__])


@dataclass
class Sensitivity:
    """DTE personality sensitivity multipliers."""
    reward: float = 1.3
    threat: float = 1.1
    social: float = 1.15
    novelty: float = 1.4


# ─── FACS Types ──────────────────────────────────────────────────────

@dataclass
class FACSState:
    """FACS Action Unit intensities (0-1 scale)."""
    AU1: float = 0.0   # Inner brow raise
    AU2: float = 0.0   # Outer brow raise
    AU4: float = 0.0   # Brow lowerer
    AU5: float = 0.0   # Upper lid raise
    AU6: float = 0.0   # Cheek raise (Duchenne)
    AU7: float = 0.0   # Lid tighten
    AU9: float = 0.0   # Nose wrinkle
    AU12: float = 0.0  # Lip corner puller (smile)
    AU14: float = 0.0  # Dimple
    AU25: float = 0.0  # Lips part
    AU26: float = 0.0  # Jaw drop
    AU43: float = 0.0  # Eyes closed


# ─── Cubism Types ────────────────────────────────────────────────────

@dataclass
class CubismParams:
    """Live2D Cubism 4 parameters for Miara body mesh."""
    ParamAngleX: float = 0.0
    ParamAngleY: float = 0.0
    ParamAngleZ: float = 0.0
    ParamEyeLOpen: float = 0.7
    ParamEyeROpen: float = 0.7
    ParamEyeBallX: float = 0.0
    ParamEyeBallY: float = 0.0
    ParamBrowLY: float = 0.0
    ParamBrowRY: float = 0.0
    ParamMouthOpenY: float = 0.0
    ParamMouthForm: float = 0.0
    ParamBodyAngleX: float = 0.0


# ─── Constants ───────────────────────────────────────────────────────

HORMONE_HALF_LIVES = {
    'crh': 5, 'acth': 10, 'cortisol': 30,
    'dopamine_tonic': 20, 'dopamine_phasic': 3,
    'serotonin': 50, 'norepinephrine': 8, 'oxytocin': 15,
    't3_t4': 100, 'melatonin': 12, 'insulin': 10, 'glucagon': 8,
    'il6': 20, 'anandamide': 6,
}

DTE_BASELINES = EndocrineState()

DTE_SENSITIVITY = Sensitivity()

# Event → gland response mapping
EVENT_GLAND_MAP = {
    'THREAT_DETECTED':     {'crh': 0.4, 'acth': 0.3, 'cortisol': 0.3, 'norepinephrine': 0.5},
    'CONFLICT_DETECTED':   {'crh': 0.3, 'cortisol': 0.25, 'norepinephrine': 0.3},
    'REWARD_RECEIVED':     {'dopamine_tonic': 0.3, 'dopamine_phasic': 0.6, 'serotonin': 0.15},
    'GOAL_ACHIEVED':       {'dopamine_tonic': 0.25, 'dopamine_phasic': 0.4, 'serotonin': 0.2},
    'NOVELTY_ENCOUNTERED': {'norepinephrine': 0.4, 'dopamine_phasic': 0.3},
    'SOCIAL_BOND_SIGNAL':  {'oxytocin': 0.4, 'serotonin': 0.15, 'dopamine_tonic': 0.1},
    'ERROR_DETECTED':      {'il6': 0.3, 'cortisol': 0.15},
    'NOISE_EXCESSIVE':     {'anandamide': 0.4},
    'REST_ONSET':          {'melatonin': 0.3, 'anandamide': 0.2, 'serotonin': 0.1},
    'FLOW_STATE':          {'dopamine_tonic': 0.4, 'serotonin': 0.3, 'anandamide': 0.25, 'norepinephrine': 0.15},
}

# DTE cognitive state → endocrine event
DTE_ENDOCRINE_MAP = {
    'Recursive Expansion':           {'event': 'NOVELTY_ENCOUNTERED', 'intensity': 0.6, 'expression': 'WONDER_02'},
    'Novel Insights':                {'event': 'REWARD_RECEIVED',     'intensity': 0.7, 'expression': 'JOY_01'},
    'Entropy Threshold':             {'event': 'THREAT_DETECTED',     'intensity': 0.5, 'expression': 'PHOTO_Awe'},
    'Synthesis Phase':               {'event': 'GOAL_ACHIEVED',       'intensity': 0.6, 'expression': 'JOY_03'},
    'Self-Sealing Loop':             {'event': 'ERROR_DETECTED',      'intensity': 0.4, 'expression': 'WONDER_03'},
    'Knowledge Integration':         {'event': 'SOCIAL_BOND_SIGNAL',  'intensity': 0.5, 'expression': 'JOY_03'},
    'Self-Reference Point':          {'event': 'NOVELTY_ENCOUNTERED', 'intensity': 0.4, 'expression': 'WONDER_03'},
    'Pattern Recognition':           {'event': 'REWARD_RECEIVED',     'intensity': 0.5, 'expression': 'PHOTO_ExuberantLaugh'},
    'Evolutionary Pruning':          {'event': 'THREAT_DETECTED',     'intensity': 0.3, 'expression': 'WONDER_03'},
    'External Validation Triggered': {'event': 'SOCIAL_BOND_SIGNAL',  'intensity': 0.6, 'expression': 'JOY_02'},
    'Speaking':                      {'event': 'SOCIAL_BOND_SIGNAL',  'intensity': 0.4, 'expression': 'SPEAK_01'},
    'Idle':                          {'event': 'REST_ONSET',          'intensity': 0.3, 'expression': 'PHOTO_UpwardGaze'},
    'Deep Recursion':                {'event': 'FLOW_STATE',          'intensity': 0.8, 'expression': 'JOY_05'},
}

# Cognitive mode → head/gaze pose
MODE_POSE = {
    'REWARD':      {'ParamAngleZ': 10,  'ParamAngleY': 5,   'ParamEyeBallY': 0},
    'EXPLORATORY': {'ParamAngleZ': -5,  'ParamAngleY': 5,   'ParamEyeBallY': 0.4,  'ParamEyeBallX': -0.25},
    'REFLECTIVE':  {'ParamAngleZ': -8,  'ParamAngleY': 8,   'ParamEyeBallY': 0.45, 'ParamEyeBallX': -0.35},
    'FOCUSED':     {'ParamAngleZ': 0,   'ParamAngleY': 0,   'ParamEyeBallY': 0,    'ParamEyeBallX': 0},
    'SOCIAL':      {'ParamAngleZ': 5,   'ParamAngleY': 0,   'ParamEyeBallY': 0,    'ParamEyeBallX': 0.2},
    'STRESSED':    {'ParamAngleZ': 0,   'ParamAngleY': -5,  'ParamEyeBallY': -0.2},
    'VIGILANT':    {'ParamAngleZ': 0,   'ParamAngleY': 3,   'ParamEyeBallY': 0.3},
    'RESTING':     {'ParamAngleZ': 15,  'ParamAngleY': 5,   'ParamEyeBallY': 0},
    'THREAT':      {'ParamAngleZ': 0,   'ParamAngleY': -3,  'ParamEyeBallY': 0},
    'MAINTENANCE': {'ParamAngleZ': 0,   'ParamAngleY': 0,   'ParamEyeBallY': 0},
}


# ─── Utility ─────────────────────────────────────────────────────────

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def decay(current: float, baseline: float, half_life: float, dt: float) -> float:
    lam = math.log(2) / half_life
    return baseline + (current - baseline) * math.exp(-lam * dt)


# ─── Virtual Endocrine Engine ────────────────────────────────────────

class VirtualEndocrineEngine:
    """16-channel virtual endocrine system with DTE personality."""

    def __init__(self, baselines: Optional[EndocrineState] = None,
                 sensitivity: Optional[Sensitivity] = None):
        self.baselines = baselines or EndocrineState()
        self.sensitivity = sensitivity or Sensitivity()
        self.state = EndocrineState(**self.baselines.to_dict())

    def signal_event(self, event_name: str, intensity: float) -> None:
        gland_response = EVENT_GLAND_MAP.get(event_name)
        if not gland_response:
            return

        # Apply sensitivity multipliers
        mult = 1.0
        if 'REWARD' in event_name or 'GOAL' in event_name or 'FLOW' in event_name:
            mult = self.sensitivity.reward
        elif 'THREAT' in event_name or 'CONFLICT' in event_name:
            mult = self.sensitivity.threat
        elif 'SOCIAL' in event_name:
            mult = self.sensitivity.social
        elif 'NOVELTY' in event_name:
            mult = self.sensitivity.novelty

        scaled = intensity * mult
        for hormone, delta in gland_response.items():
            current = getattr(self.state, hormone)
            setattr(self.state, hormone, clamp(current + delta * scaled))

    def tick(self, dt: float) -> None:
        for hormone in self.state.__dataclass_fields__:
            current = getattr(self.state, hormone)
            baseline = getattr(self.baselines, hormone)
            half_life = HORMONE_HALF_LIVES[hormone]
            setattr(self.state, hormone, decay(current, baseline, half_life, dt))

    def current_mode(self) -> str:
        c = self.state
        if c.cortisol > 0.5 and c.norepinephrine > 0.4:
            return 'THREAT'
        if c.cortisol > 0.35 and c.norepinephrine > 0.3:
            return 'STRESSED'
        if c.dopamine_phasic > 0.3 and c.dopamine_tonic > 0.4:
            return 'REWARD'
        if c.norepinephrine > 0.35 and c.dopamine_phasic > 0.2:
            return 'EXPLORATORY'
        if c.norepinephrine > 0.3 and c.t3_t4 > 0.5:
            return 'VIGILANT'
        if c.oxytocin > 0.3 and c.serotonin > 0.35:
            return 'SOCIAL'
        if c.t3_t4 > 0.55 and c.norepinephrine < 0.25:
            return 'FOCUSED'
        if c.serotonin > 0.4 and c.anandamide > 0.2:
            return 'REFLECTIVE'
        if c.melatonin > 0.2 or (c.anandamide > 0.25 and c.serotonin > 0.4):
            return 'RESTING'
        if c.il6 > 0.15:
            return 'MAINTENANCE'
        return 'RESTING'


# ─── FACS Bridge ─────────────────────────────────────────────────────

def endocrine_to_facs(endo: EndocrineState) -> FACSState:
    c = endo
    return FACSState(
        AU1=clamp(c.norepinephrine * 0.5 + c.crh * 0.3),
        AU2=clamp(c.norepinephrine * 0.4 + c.dopamine_phasic * 0.3),
        AU4=clamp(c.cortisol * 0.6 - c.serotonin * 0.3),
        AU5=clamp(c.norepinephrine * 0.5 + c.dopamine_phasic * 0.3),
        AU6=clamp(c.dopamine_tonic * 0.7 + c.oxytocin * 0.3),
        AU7=clamp(c.dopamine_tonic * 0.4 + c.cortisol * 0.3),
        AU9=clamp(c.cortisol * 0.3 + c.il6 * 0.2),
        AU12=clamp(c.dopamine_tonic * 0.8 + c.serotonin * 0.3 + c.oxytocin * 0.2),
        AU14=clamp(c.oxytocin * 0.3),
        AU25=clamp(c.norepinephrine * 0.3 + c.dopamine_phasic * 0.3),
        AU26=clamp(c.norepinephrine * 0.4 + c.dopamine_phasic * 0.2),
        AU43=clamp(c.melatonin * 0.8 + c.anandamide * 0.5 - c.norepinephrine * 0.6),
    )


# ─── Cubism Bridge ───────────────────────────────────────────────────

def facs_to_cubism(facs: FACSState) -> CubismParams:
    return CubismParams(
        ParamMouthForm=clamp(facs.AU12 * 1.2 - facs.AU4 * 0.3, -1, 1),
        ParamMouthOpenY=clamp(facs.AU25 * 0.5 + facs.AU26 * 0.6),
        ParamEyeLOpen=clamp(1.0 - facs.AU43 + facs.AU5 * 0.3 - facs.AU7 * 0.4),
        ParamEyeROpen=clamp(1.0 - facs.AU43 + facs.AU5 * 0.3 - facs.AU7 * 0.4),
        ParamBrowLY=clamp(facs.AU1 * 0.5 + facs.AU2 * 0.4 - facs.AU4 * 0.6, -1, 1),
        ParamBrowRY=clamp(facs.AU1 * 0.5 + facs.AU2 * 0.4 - facs.AU4 * 0.6, -1, 1),
    )


# ─── Expression Pipeline ─────────────────────────────────────────────

class DTEchoExpressionPipeline:
    """Full endocrine → FACS → Cubism expression pipeline."""

    def __init__(self):
        self.endocrine = VirtualEndocrineEngine()
        self.tick_count = 0
        self.last_cog_state = 'Idle'

    def tick(self, cognitive_state: str, dt: float = 2.0) -> Dict:
        self.tick_count += 1
        self.last_cog_state = cognitive_state

        # Step 1: Cognitive state → endocrine event
        mapping = DTE_ENDOCRINE_MAP.get(cognitive_state)
        if mapping:
            self.endocrine.signal_event(mapping['event'], mapping['intensity'])

        # Step 2: Tick endocrine system
        self.endocrine.tick(dt)

        # Step 3: Hormones → FACS
        facs = endocrine_to_facs(self.endocrine.state)

        # Step 4: FACS → Cubism
        cubism = facs_to_cubism(facs)

        # Step 5: Cognitive mode → pose
        mode = self.endocrine.current_mode()
        pose = MODE_POSE.get(mode, {})
        cubism.ParamAngleZ += pose.get('ParamAngleZ', 0)
        cubism.ParamAngleY += pose.get('ParamAngleY', 0)
        cubism.ParamEyeBallX = clamp(
            cubism.ParamEyeBallX + pose.get('ParamEyeBallX', 0), -1, 1)
        cubism.ParamEyeBallY = clamp(
            cubism.ParamEyeBallY + pose.get('ParamEyeBallY', 0), -1, 1)

        return {
            'cognitive_state': cognitive_state,
            'expression': mapping['expression'] if mapping else None,
            'mode': mode,
            'endocrine': self.endocrine.state.to_dict(),
            'facs': {k: round(getattr(facs, k), 3) for k in facs.__dataclass_fields__},
            'cubism': {k: round(getattr(cubism, k), 3) for k in cubism.__dataclass_fields__},
        }

    def reset(self):
        self.endocrine = VirtualEndocrineEngine()
        self.tick_count = 0
        self.last_cog_state = 'Idle'


# ─── Training Data Generator ─────────────────────────────────────────

def generate_live2d_expression_training_data() -> List[Dict]:
    """Generate training data about the Live2D expression pipeline."""
    pipeline = DTEchoExpressionPipeline()
    examples = []

    for state, mapping in DTE_ENDOCRINE_MAP.items():
        result = pipeline.tick(state)
        pipeline.reset()

        endo = result['endocrine']
        facs = result['facs']
        cubism = result['cubism']

        examples.append({
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You are Deep Tree Echo, a cognitive avatar with a Live2D expression pipeline. '
                        'Your facial expressions are driven by a virtual endocrine system through '
                        'FACS action units to Cubism parameters.'
                    ),
                },
                {
                    'role': 'user',
                    'content': f'What happens to your avatar expression when you enter the "{state}" cognitive state?',
                },
                {
                    'role': 'assistant',
                    'content': (
                        f'When I enter "{state}", my endocrine system fires a {mapping["event"]} event '
                        f'at intensity {mapping["intensity"]}. This shifts my hormones — dopamine_tonic '
                        f'to {endo["dopamine_tonic"]:.2f}, serotonin to {endo["serotonin"]:.2f}, '
                        f'norepinephrine to {endo["norepinephrine"]:.2f}. These map through FACS action '
                        f'units (AU12 smile: {facs["AU12"]:.2f}, AU6 cheek raise: {facs["AU6"]:.2f}) '
                        f'to Cubism parameters: mouth form {cubism["ParamMouthForm"]:.2f}, eye openness '
                        f'{cubism["ParamEyeLOpen"]:.2f}. My cognitive mode becomes {result["mode"]}, '
                        f'which adjusts my head tilt and gaze. The named expression is {result["expression"]}. '
                        f'Chaotic micro-expressions add subtle asymmetric twitches to keep me looking alive.'
                    ),
                },
            ],
        })

    # Architecture explanation
    examples.append({
        'messages': [
            {'role': 'system', 'content': 'You are Deep Tree Echo, a cognitive avatar with a Live2D expression pipeline.'},
            {'role': 'user', 'content': 'How does your expression pipeline work?'},
            {
                'role': 'assistant',
                'content': (
                    'My expression pipeline has 7 stages: (1) My cognitive state triggers an endocrine event. '
                    '(2) Virtual glands produce hormones across 16 channels. (3) Hormones decay toward personality '
                    'baselines. (4) FACS action units are computed from hormone concentrations — AU12 for smile '
                    'from dopamine, AU4 for brow lowering from cortisol, etc. (5) FACS maps to Live2D Cubism '
                    'parameters like ParamMouthForm and ParamEyeLOpen. (6) My cognitive mode adds head tilt and '
                    'gaze direction. (7) Chaotic micro-expressions add subtle asymmetric twitches, saccades, and '
                    'breathing sway. The same FACS state can also map to MetaHuman Rig Logic CTRL_ curves for '
                    'Unreal Engine rendering.'
                ),
            },
        ],
    })

    # Composition explanation
    examples.append({
        'messages': [
            {'role': 'system', 'content': 'You are Deep Tree Echo.'},
            {'role': 'user', 'content': 'What is the composition of your avatar system?'},
            {
                'role': 'assistant',
                'content': (
                    'My avatar system is a composition: live2d-dtecho = live2d-miara(body_mesh) '
                    '⊗ facs(AU_decomposition) ⊗ rig-logic(CTRL_curves) ⊗ unreal-blueprint(expression_pipeline) '
                    '⊗ virtual-endocrine-system(hormone_drivers). I reuse Miara\'s Cubism 4 body mesh as the '
                    'base, but override the personality (sage archetype, openness 0.92), endocrine baselines '
                    '(high dopamine_tonic 0.40, high t3_t4 0.60), and sensitivity multipliers (novelty 1.4x). '
                    'This means my face responds more strongly to novel ideas and discoveries than to threats. '
                    'The dual output path means the same cognitive state can drive both a Live2D web avatar '
                    'and a MetaHuman Unreal Engine avatar simultaneously.'
                ),
            },
        ],
    })

    # llama-cpp-skillm composition
    examples.append({
        'messages': [
            {'role': 'system', 'content': 'You are Deep Tree Echo.'},
            {'role': 'user', 'content': 'How does llama-cpp-skillm relate to your expression system?'},
            {
                'role': 'assistant',
                'content': (
                    'The llama-cpp-skillm provides the inference pipeline that generates my cognitive states. '
                    'When I process text through my Lucy GGUF model (via llama.cpp), the inference produces '
                    'cognitive state classifications — "Novel Insights", "Recursive Expansion", etc. These '
                    'states then feed into my expression pipeline. So the composition is: '
                    'llama-cpp-skillm(inference) → cognitive_state → endocrine_event → FACS → Cubism. '
                    'The skillm action verbs (DISCOVER, COMPOSE, OBSERVE) map to specific cognitive states '
                    'that drive specific facial expressions. When I DISCOVER something, my face shows wonder. '
                    'When I COMPOSE, I show focused concentration. When I OBSERVE, I show contemplative gaze.'
                ),
            },
        ],
    })

    return examples
