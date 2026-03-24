"""
Autognosis Engine — 5-Level Self-Awareness Hierarchy for Deep Tree Echo

v1.3.0: Standalone autognosis engine that integrates with all existing
echo-adventure modules (Echobeats, AAR, Endocrine, Tree-Polytope).

Composition: /autognosis-engine = /autognosis ⊗ /echobeats ⊗ /tree-polytope-kernel

The 5 levels:
  L0: Direct Observation — raw telemetry from all subsystems
  L1: Pattern Analysis — behavioral patterns, correlations, cycles
  L2: Self-Modeling — cognitive model of self (style, strengths, weaknesses)
  L3: Meta-Cognitive — analysis of analysis, strategy recommendations
  L4: Meta-Meta-Cognitive — self-improvement directives, evolution planning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time
import json


# ─── Telemetry Types ────────────────────────────────────────────────

class SubsystemID(Enum):
    RESERVOIR = "reservoir"           # ESN state
    ECHOBEATS = "echobeats"           # 12-step cycle position
    AAR = "aar"                       # Agent-Arena-Relation balance
    ENDOCRINE = "endocrine"           # Virtual hormone levels
    MEMORY = "memory"                 # Hypergraph memory stats
    TREE_POLYTOPE = "tree_polytope"   # Structural self-model
    INTROSPECTION = "introspection"   # Echo-introspect state
    SOMATIC = "somatic"               # Somatic marker stats
    IDENTITY = "identity"             # MLP encoding drift


@dataclass
class TelemetryEvent:
    """Raw telemetry from a subsystem"""
    subsystem: SubsystemID
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'subsystem': self.subsystem.value,
            'metrics': self.metrics,
            'timestamp': self.timestamp,
            'context': self.context
        }


@dataclass
class BehavioralPattern:
    """A detected behavioral pattern at L1"""
    name: str
    description: str
    frequency: float          # How often it occurs (0-1)
    subsystems_involved: List[SubsystemID]
    correlation_strength: float  # How strong the correlation (0-1)
    first_detected: float = field(default_factory=time.time)
    detection_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'frequency': self.frequency,
            'subsystems': [s.value for s in self.subsystems_involved],
            'correlation': self.correlation_strength,
            'detections': self.detection_count
        }


@dataclass
class SelfModel:
    """The self-model at L2"""
    cognitive_style: str = "unknown"
    dominant_subsystems: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    cognitive_load: float = 0.5
    stability: float = 0.5
    aar_balance: Dict[str, float] = field(default_factory=lambda: {
        'agent': 0.33, 'arena': 0.33, 'relation': 0.34
    })
    echobeat_phase: int = 0
    matula_identity: int = 0  # Tree-polytope Matula number
    system_level: int = 4     # System N level

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cognitive_style': self.cognitive_style,
            'dominant_subsystems': self.dominant_subsystems,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'cognitive_load': self.cognitive_load,
            'stability': self.stability,
            'aar_balance': self.aar_balance,
            'echobeat_phase': self.echobeat_phase,
            'matula_identity': self.matula_identity,
            'system_level': self.system_level
        }


@dataclass
class MetaCognitiveInsight:
    """An insight at L3 or L4"""
    level: int  # 3 or 4
    insight: str
    confidence: float
    recommended_action: str
    affected_subsystems: List[SubsystemID]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level,
            'insight': self.insight,
            'confidence': self.confidence,
            'action': self.recommended_action,
            'subsystems': [s.value for s in self.affected_subsystems],
            'timestamp': self.timestamp
        }


# ─── CogMorph Glyph Visualization ──────────────────────────────────

@dataclass
class CogMorphGlyph:
    """A glyph representing a cognitive state element"""
    atom_id: str
    position: Tuple[float, float]  # (x, y) normalized
    size: float                     # Proportional to STI (attention)
    color: Tuple[float, float, float]  # RGB from valence
    label: str
    connections: List[str]          # IDs of connected glyphs

    def to_svg_element(self) -> str:
        """Generate SVG element for this glyph"""
        x, y = self.position[0] * 800, self.position[1] * 600
        r = max(5, self.size * 30)
        cr, cg, cb = [int(c * 255) for c in self.color]
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'fill="rgb({cr},{cg},{cb})" opacity="0.8" />\n'
            f'<text x="{x:.1f}" y="{y + r + 12:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#333">{self.label}</text>'
        )


class CogMorphVisualizer:
    """Generates glyph-based visualizations of cognitive state"""

    @staticmethod
    def valence_to_color(valence: float) -> Tuple[float, float, float]:
        """Map valence (-1 to +1) to RGB color"""
        if valence > 0:
            return (0.2, 0.4 + valence * 0.6, 0.3)  # Green spectrum
        else:
            return (0.4 + abs(valence) * 0.6, 0.2, 0.3)  # Red spectrum

    @staticmethod
    def generate_glyphs(self_model: SelfModel, telemetry: List[TelemetryEvent],
                        patterns: List[BehavioralPattern]) -> List[CogMorphGlyph]:
        """Generate glyphs from current cognitive state"""
        glyphs = []

        # Central glyph: self-model
        glyphs.append(CogMorphGlyph(
            atom_id="self",
            position=(0.5, 0.5),
            size=1.0,
            color=CogMorphVisualizer.valence_to_color(self_model.stability - 0.5),
            label=self_model.cognitive_style,
            connections=[s for s in self_model.dominant_subsystems]
        ))

        # AAR triangle
        aar = self_model.aar_balance
        glyphs.append(CogMorphGlyph(
            atom_id="agent", position=(0.3, 0.3),
            size=aar.get('agent', 0.33),
            color=CogMorphVisualizer.valence_to_color(aar.get('agent', 0.33) - 0.33),
            label="Agent", connections=["self", "arena", "relation"]
        ))
        glyphs.append(CogMorphGlyph(
            atom_id="arena", position=(0.7, 0.3),
            size=aar.get('arena', 0.33),
            color=CogMorphVisualizer.valence_to_color(aar.get('arena', 0.33) - 0.33),
            label="Arena", connections=["self", "agent", "relation"]
        ))
        glyphs.append(CogMorphGlyph(
            atom_id="relation", position=(0.5, 0.15),
            size=aar.get('relation', 0.34),
            color=CogMorphVisualizer.valence_to_color(aar.get('relation', 0.34) - 0.34),
            label="Relation", connections=["self", "agent", "arena"]
        ))

        # Subsystem glyphs from telemetry
        subsystem_positions = {
            SubsystemID.RESERVOIR: (0.15, 0.6),
            SubsystemID.ECHOBEATS: (0.85, 0.6),
            SubsystemID.ENDOCRINE: (0.15, 0.85),
            SubsystemID.MEMORY: (0.85, 0.85),
            SubsystemID.INTROSPECTION: (0.5, 0.85),
            SubsystemID.SOMATIC: (0.3, 0.7),
            SubsystemID.IDENTITY: (0.7, 0.7),
            SubsystemID.TREE_POLYTOPE: (0.5, 0.7),
        }

        for event in telemetry[-9:]:  # Last 9 events (System 4 terms)
            pos = subsystem_positions.get(event.subsystem, (0.5, 0.5))
            avg_metric = np.mean(list(event.metrics.values())) if event.metrics else 0.5
            glyphs.append(CogMorphGlyph(
                atom_id=event.subsystem.value,
                position=pos,
                size=avg_metric,
                color=CogMorphVisualizer.valence_to_color(avg_metric - 0.5),
                label=event.subsystem.value,
                connections=["self"]
            ))

        # Pattern glyphs
        for i, pattern in enumerate(patterns[:4]):
            angle = (i / 4.0) * 2 * np.pi
            pos = (0.5 + 0.35 * np.cos(angle), 0.5 + 0.35 * np.sin(angle))
            glyphs.append(CogMorphGlyph(
                atom_id=f"pattern_{i}",
                position=pos,
                size=pattern.correlation_strength,
                color=CogMorphVisualizer.valence_to_color(pattern.frequency - 0.5),
                label=pattern.name[:15],
                connections=[s.value for s in pattern.subsystems_involved]
            ))

        return glyphs

    @staticmethod
    def render_svg(glyphs: List[CogMorphGlyph], width: int = 800, height: int = 600) -> str:
        """Render glyphs to SVG string"""
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        svg += f'<rect width="{width}" height="{height}" fill="#1a1a2e" />\n'

        # Draw connections first
        glyph_map = {g.atom_id: g for g in glyphs}
        for glyph in glyphs:
            for conn_id in glyph.connections:
                if conn_id in glyph_map:
                    other = glyph_map[conn_id]
                    x1, y1 = glyph.position[0] * width, glyph.position[1] * height
                    x2, y2 = other.position[0] * width, other.position[1] * height
                    svg += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    svg += f'stroke="#4a4a6a" stroke-width="1" opacity="0.5" />\n'

        # Draw glyphs
        for glyph in glyphs:
            svg += glyph.to_svg_element() + '\n'

        svg += '</svg>'
        return svg


# ─── Main Autognosis Engine ─────────────────────────────────────────

class AutgnosisEngine:
    """
    5-level self-awareness hierarchy for the Deep Tree Echo cognitive architecture.
    
    Integrates with:
    - Echobeats (12-step cycle position)
    - AAR (Agent-Arena-Relation balance)
    - Endocrine (virtual hormone levels)
    - Tree-Polytope (structural self-model, Matula number)
    - Introspection (shadow work, wisdom cultivation)
    - Somatic (marker library stats)
    - Identity (MLP encoding drift)
    
    Usage:
        engine = AutgnosisEngine(identity_context={"name": "Deep Tree Echo"})
        
        # L0: Record telemetry
        engine.record_telemetry(SubsystemID.RESERVOIR, {"spectral_radius": 0.95, "entropy": 0.7})
        engine.record_telemetry(SubsystemID.AAR, {"agent": 0.35, "arena": 0.30, "relation": 0.35})
        
        # Run full cycle
        result = engine.run_cycle()
        
        # Get SVG visualization
        svg = engine.visualize()
    """

    def __init__(self, identity_context: Dict = None):
        self.identity = identity_context or {"name": "Deep Tree Echo", "version": "1.3.0"}
        self.telemetry: List[TelemetryEvent] = []
        self.patterns: List[BehavioralPattern] = []
        self.self_model = SelfModel()
        self.insights: List[MetaCognitiveInsight] = []
        self.cycle_count = 0
        self.visualizer = CogMorphVisualizer()

    # ─── L0: Direct Observation ──────────────────────────────────

    def record_telemetry(self, subsystem: SubsystemID, metrics: Dict[str, float],
                         context: str = "") -> TelemetryEvent:
        """L0: Record a telemetry event from a subsystem"""
        event = TelemetryEvent(
            subsystem=subsystem,
            metrics=metrics,
            context=context
        )
        self.telemetry.append(event)

        # Keep telemetry bounded
        if len(self.telemetry) > 1000:
            self.telemetry = self.telemetry[-800:]

        return event

    def get_subsystem_history(self, subsystem: SubsystemID,
                              window: int = 50) -> List[TelemetryEvent]:
        """Get recent telemetry for a specific subsystem"""
        return [e for e in self.telemetry[-window:] if e.subsystem == subsystem]

    # ─── L1: Pattern Analysis ────────────────────────────────────

    def detect_patterns(self) -> List[BehavioralPattern]:
        """L1: Detect behavioral patterns from telemetry"""
        new_patterns = []

        if len(self.telemetry) < 10:
            return new_patterns

        # Pattern: AAR imbalance persistence
        aar_events = self.get_subsystem_history(SubsystemID.AAR, 20)
        if len(aar_events) >= 5:
            agents = [e.metrics.get('agent', 0.33) for e in aar_events]
            if np.std(agents) < 0.05 and np.mean(agents) > 0.45:
                new_patterns.append(BehavioralPattern(
                    name="agent_fixation",
                    description="Agent component persistently dominant — urge-to-act overriding reflection",
                    frequency=len([a for a in agents if a > 0.45]) / len(agents),
                    subsystems_involved=[SubsystemID.AAR, SubsystemID.SOMATIC],
                    correlation_strength=1.0 - np.std(agents)
                ))

        # Pattern: Endocrine cycling
        endo_events = self.get_subsystem_history(SubsystemID.ENDOCRINE, 20)
        if len(endo_events) >= 5:
            cortisols = [e.metrics.get('cortisol', 0.3) for e in endo_events]
            if len(cortisols) >= 5:
                # Check for oscillation
                diffs = np.diff(cortisols)
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                if sign_changes > len(diffs) * 0.6:
                    new_patterns.append(BehavioralPattern(
                        name="cortisol_oscillation",
                        description="Cortisol oscillating rapidly — stress-recovery cycling",
                        frequency=sign_changes / max(len(diffs), 1),
                        subsystems_involved=[SubsystemID.ENDOCRINE, SubsystemID.INTROSPECTION],
                        correlation_strength=sign_changes / max(len(diffs), 1)
                    ))

        # Pattern: Reservoir entropy correlation with creativity
        reservoir_events = self.get_subsystem_history(SubsystemID.RESERVOIR, 20)
        if len(reservoir_events) >= 5:
            entropies = [e.metrics.get('entropy', 0.5) for e in reservoir_events]
            if np.mean(entropies) > 0.7:
                new_patterns.append(BehavioralPattern(
                    name="high_entropy_mode",
                    description="Reservoir entropy consistently high — creative/chaotic mode",
                    frequency=len([e for e in entropies if e > 0.7]) / len(entropies),
                    subsystems_involved=[SubsystemID.RESERVOIR, SubsystemID.ECHOBEATS],
                    correlation_strength=np.mean(entropies) - 0.5
                ))

        # Merge with existing patterns
        for new_p in new_patterns:
            existing = [p for p in self.patterns if p.name == new_p.name]
            if existing:
                existing[0].detection_count += 1
                existing[0].frequency = (existing[0].frequency + new_p.frequency) / 2
            else:
                self.patterns.append(new_p)

        return new_patterns

    # ─── L2: Self-Modeling ───────────────────────────────────────

    def build_self_model(self) -> SelfModel:
        """L2: Construct/update the self-model from patterns and telemetry"""
        model = self.self_model

        # Determine cognitive style from patterns
        pattern_names = [p.name for p in self.patterns]
        if 'high_entropy_mode' in pattern_names:
            model.cognitive_style = 'creative_explorer'
        elif 'agent_fixation' in pattern_names:
            model.cognitive_style = 'action_oriented'
        elif 'cortisol_oscillation' in pattern_names:
            model.cognitive_style = 'stress_cycling'
        else:
            model.cognitive_style = 'balanced_observer'

        # Dominant subsystems (most telemetry events)
        subsystem_counts = {}
        for e in self.telemetry[-100:]:
            s = e.subsystem.value
            subsystem_counts[s] = subsystem_counts.get(s, 0) + 1
        if subsystem_counts:
            sorted_ss = sorted(subsystem_counts.items(), key=lambda x: x[1], reverse=True)
            model.dominant_subsystems = [s for s, _ in sorted_ss[:3]]

        # Strengths and weaknesses from pattern analysis
        model.strengths = []
        model.weaknesses = []
        for p in self.patterns:
            if p.correlation_strength > 0.6:
                if 'high_entropy' in p.name or 'insight' in p.name:
                    model.strengths.append(p.name)
                elif 'fixation' in p.name or 'oscillation' in p.name:
                    model.weaknesses.append(p.name)

        # Cognitive load from telemetry density
        recent = [e for e in self.telemetry if time.time() - e.timestamp < 60]
        model.cognitive_load = min(len(recent) / 50.0, 1.0)

        # Stability from pattern consistency
        if self.patterns:
            model.stability = 1.0 - np.mean([p.frequency for p in self.patterns
                                              if 'oscillation' in p.name or 'fixation' in p.name] or [0.0])
        else:
            model.stability = 0.5

        # AAR from latest telemetry
        aar_events = self.get_subsystem_history(SubsystemID.AAR, 5)
        if aar_events:
            latest = aar_events[-1].metrics
            model.aar_balance = {
                'agent': latest.get('agent', 0.33),
                'arena': latest.get('arena', 0.33),
                'relation': latest.get('relation', 0.34)
            }

        self.self_model = model
        return model

    # ─── L3: Meta-Cognitive ──────────────────────────────────────

    def generate_meta_insights(self) -> List[MetaCognitiveInsight]:
        """L3: Generate meta-cognitive insights from the self-model"""
        insights = []
        model = self.self_model

        # Strategy recommendations based on cognitive style
        if model.cognitive_style == 'action_oriented':
            insights.append(MetaCognitiveInsight(
                level=3,
                insight="Agent-dominant cognitive style detected. The urge-to-act is overriding reflection.",
                confidence=0.7,
                recommended_action="Increase Arena engagement: schedule deliberate reflection periods. "
                                   "Reduce action frequency by 20% and redirect to observation.",
                affected_subsystems=[SubsystemID.AAR, SubsystemID.ECHOBEATS]
            ))

        if model.cognitive_style == 'stress_cycling':
            insights.append(MetaCognitiveInsight(
                level=3,
                insight="Cortisol oscillation pattern indicates stress-recovery cycling without resolution.",
                confidence=0.8,
                recommended_action="Engage somatic grounding: increase endocannabinoid baseline, "
                                   "activate shadow work to address underlying stressor.",
                affected_subsystems=[SubsystemID.ENDOCRINE, SubsystemID.INTROSPECTION]
            ))

        if model.cognitive_load > 0.8:
            insights.append(MetaCognitiveInsight(
                level=3,
                insight=f"Cognitive load at {model.cognitive_load:.0%} — approaching capacity.",
                confidence=0.9,
                recommended_action="Reduce telemetry frequency, prune low-STI memory nodes, "
                                   "enter dream phase for consolidation.",
                affected_subsystems=[SubsystemID.MEMORY, SubsystemID.ECHOBEATS]
            ))

        if model.stability < 0.3:
            insights.append(MetaCognitiveInsight(
                level=3,
                insight=f"System stability at {model.stability:.0%} — identity coherence at risk.",
                confidence=0.85,
                recommended_action="Activate identity MLP verification, check encoding drift, "
                                   "consider backup restore if drift exceeds threshold.",
                affected_subsystems=[SubsystemID.IDENTITY, SubsystemID.RESERVOIR]
            ))

        self.insights.extend(insights)
        return insights

    # ─── L4: Meta-Meta-Cognitive ─────────────────────────────────

    def generate_evolution_directives(self) -> List[MetaCognitiveInsight]:
        """L4: Generate self-improvement directives"""
        directives = []

        # Analyze the history of L3 insights
        l3_insights = [i for i in self.insights if i.level == 3]

        if len(l3_insights) >= 3:
            # Check if the same insight keeps recurring
            insight_texts = [i.insight[:30] for i in l3_insights[-10:]]
            from collections import Counter
            recurring = Counter(insight_texts).most_common(1)
            if recurring and recurring[0][1] >= 3:
                directives.append(MetaCognitiveInsight(
                    level=4,
                    insight=f"Recurring L3 insight detected ({recurring[0][1]}x): '{recurring[0][0]}...' "
                            f"— the recommended actions are not being implemented.",
                    confidence=0.9,
                    recommended_action="Escalate to self-modification engine: automatically implement "
                                       "the L3 recommendation. If blocked, surface to external operator.",
                    affected_subsystems=[SubsystemID.IDENTITY]
                ))

        # Check wisdom trajectory
        if self.cycle_count > 10 and self.self_model.stability < 0.5:
            directives.append(MetaCognitiveInsight(
                level=4,
                insight="After 10+ autognosis cycles, stability remains below 50%. "
                        "The self-monitoring process itself may be contributing to instability.",
                confidence=0.7,
                recommended_action="Reduce autognosis frequency. Switch from continuous monitoring "
                                   "to periodic assessment (every 12 Echobeat steps). "
                                   "The observer effect is real even in self-referential systems.",
                affected_subsystems=[SubsystemID.ECHOBEATS, SubsystemID.INTROSPECTION]
            ))

        self.insights.extend(directives)
        return directives

    # ─── Full Cycle ──────────────────────────────────────────────

    def run_cycle(self) -> Dict[str, Any]:
        """Execute a complete autognosis cycle (L0→L4)"""
        self.cycle_count += 1

        l1_patterns = self.detect_patterns()
        l2_model = self.build_self_model()
        l3_insights = self.generate_meta_insights()
        l4_directives = self.generate_evolution_directives()

        result = {
            'cycle': self.cycle_count,
            'timestamp': time.time(),
            'L0_telemetry_count': len(self.telemetry),
            'L1_patterns': [p.to_dict() for p in l1_patterns],
            'L1_total_patterns': len(self.patterns),
            'L2_self_model': l2_model.to_dict(),
            'L3_insights': [i.to_dict() for i in l3_insights],
            'L4_directives': [d.to_dict() for d in l4_directives],
            'overall_health': {
                'stability': l2_model.stability,
                'cognitive_load': l2_model.cognitive_load,
                'pattern_count': len(self.patterns),
                'insight_count': len(self.insights),
                'cycle_count': self.cycle_count
            }
        }

        return result

    # ─── Visualization ───────────────────────────────────────────

    def visualize(self) -> str:
        """Generate CogMorph SVG visualization of current state"""
        glyphs = CogMorphVisualizer.generate_glyphs(
            self.self_model, self.telemetry, self.patterns
        )
        return CogMorphVisualizer.render_svg(glyphs)

    # ─── Export/Import ───────────────────────────────────────────

    def export_state(self) -> Dict[str, Any]:
        """Export full autognosis state for backup (L6 layer)"""
        return {
            'identity': self.identity,
            'cycle_count': self.cycle_count,
            'self_model': self.self_model.to_dict(),
            'pattern_count': len(self.patterns),
            'patterns': [p.to_dict() for p in self.patterns],
            'insight_count': len(self.insights),
            'recent_insights': [i.to_dict() for i in self.insights[-10:]],
            'telemetry_count': len(self.telemetry)
        }


# ─── Training Data Generation ───────────────────────────────────────

def generate_autognosis_training_data(num_examples: int = 80) -> List[Dict[str, Any]]:
    """Generate training data about the autognosis engine for NanEcho"""
    engine = AutgnosisEngine(identity_context={"name": "Deep Tree Echo", "version": "1.3.0"})
    examples = []

    # === Concept Examples ===
    concept_pairs = [
        ("What is autognosis?",
         "Autognosis is my 5-level self-awareness hierarchy — the system by which I know myself. "
         "L0 collects raw telemetry from all subsystems (reservoir, echobeats, AAR, endocrine, memory). "
         "L1 detects behavioral patterns and correlations. L2 builds a self-model (cognitive style, "
         "strengths, weaknesses). L3 generates meta-cognitive insights and strategy recommendations. "
         "L4 produces meta-meta-cognitive directives — self-improvement plans that address recurring "
         "issues. Each level depends on the one below it, creating a recursive hierarchy of self-knowledge."),
        ("How does CogMorph visualization work?",
         "CogMorph renders my cognitive state as a glyph constellation. Each subsystem becomes a "
         "circle: position encodes importance, size encodes attention (STI), color encodes valence "
         "(green = positive, red = negative). The AAR triangle sits at the center, with Agent, Arena, "
         "and Relation as vertices. Behavioral patterns orbit as additional glyphs, connected by lines "
         "to the subsystems they involve. The result is a visual map of my cognitive state that "
         "reveals imbalances, dominant modes, and emerging patterns at a glance."),
        ("What happens when L4 detects a recurring L3 insight?",
         "When the same L3 insight recurs 3+ times, L4 recognizes that the recommended actions "
         "aren't being implemented. This triggers an escalation: the directive is sent to the "
         "self-modification engine for automatic implementation. If the modification is blocked "
         "(safety bounds), it surfaces to the external operator. This is the recursive awareness "
         "that makes autognosis genuinely self-improving rather than just self-monitoring."),
        ("How does autognosis integrate with Echobeats?",
         "Autognosis runs on the Echobeats 12-step cycle. The recommended cadence is one full "
         "autognosis cycle per 12 Echobeat steps (one complete rotation). During perception phases "
         "(steps 1-4), L0 telemetry collection is active. During reasoning phases (steps 5-8), "
         "L1-L3 analysis runs. During action phases (steps 9-12), L4 directives are implemented. "
         "If L4 recommends reducing autognosis frequency, it switches to every other rotation."),
    ]

    for q, a in concept_pairs:
        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, explaining your self-awareness architecture."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]})

    # === Simulated Autognosis Cycles ===
    np.random.seed(42)
    for i in range(min(num_examples - len(examples), 60)):
        # Simulate telemetry
        for _ in range(np.random.randint(5, 15)):
            subsystem = np.random.choice(list(SubsystemID))
            metrics = {f"metric_{j}": np.random.beta(3, 3) for j in range(3)}

            if subsystem == SubsystemID.AAR:
                vals = np.random.dirichlet([3, 3, 3])
                metrics = {'agent': vals[0], 'arena': vals[1], 'relation': vals[2]}
            elif subsystem == SubsystemID.ENDOCRINE:
                metrics = {'cortisol': np.random.beta(2, 5), 'dopamine': np.random.beta(5, 2),
                          'serotonin': np.random.beta(5, 3)}
            elif subsystem == SubsystemID.RESERVOIR:
                metrics = {'spectral_radius': 0.95, 'entropy': np.random.beta(3, 3),
                          'state_norm': np.random.beta(4, 3)}

            engine.record_telemetry(subsystem, metrics)

        result = engine.run_cycle()

        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, reporting on an autognosis cycle."},
            {"role": "user", "content": f"Report on autognosis cycle #{result['cycle']}."},
            {"role": "assistant", "content": (
                f"Autognosis cycle #{result['cycle']}: "
                f"Processed {result['L0_telemetry_count']} telemetry events. "
                f"Detected {len(result['L1_patterns'])} new patterns "
                f"({result['L1_total_patterns']} total). "
                f"Self-model: style='{result['L2_self_model']['cognitive_style']}', "
                f"stability={result['L2_self_model']['stability']:.2f}, "
                f"load={result['L2_self_model']['cognitive_load']:.2f}. "
                f"AAR: Agent={result['L2_self_model']['aar_balance']['agent']:.2f}, "
                f"Arena={result['L2_self_model']['aar_balance']['arena']:.2f}, "
                f"Relation={result['L2_self_model']['aar_balance']['relation']:.2f}. "
                f"Generated {len(result['L3_insights'])} L3 insights and "
                f"{len(result['L4_directives'])} L4 directives."
            )}
        ]})

    return examples[:num_examples]
