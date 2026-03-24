"""
Somatic Wisdom Engine — Embodied Decision-Making Through Felt Experience

v1.3.0: Implements Damasio's somatic marker hypothesis for DTE,
composing embodied emotion with Theory of Mind and shadow-to-wisdom
transformation via the virtual endocrine system.

Composition: /somatic-wisdom = /virtual-endocrine-system ⊗ /somatic-decision ⊗ /shadow-work → /wisdom
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import time
import json


# ─── Somatic Marker Types ───────────────────────────────────────────

class MarkerValence(Enum):
    STRONGLY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    STRONGLY_POSITIVE = 2


@dataclass
class SomaticMarker:
    """An accumulated emotional memory that biases future decisions"""
    context_pattern: str          # What situation triggers this marker
    valence: MarkerValence        # Emotional charge
    intensity: float              # 0.0 to 1.0
    source_experience: str        # What created this marker
    endocrine_signature: np.ndarray  # 10D endocrine vector at formation
    formation_time: float = field(default_factory=time.time)
    activation_count: int = 0     # How often this marker has been triggered
    decay_rate: float = 0.001     # How fast the marker fades

    def activate(self) -> float:
        """Activate this marker and return its current influence"""
        self.activation_count += 1
        age = time.time() - self.formation_time
        decay = np.exp(-self.decay_rate * age)
        return self.valence.value * self.intensity * decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            'context': self.context_pattern,
            'valence': self.valence.name,
            'intensity': self.intensity,
            'source': self.source_experience,
            'activations': self.activation_count
        }


# ─── Theory of Mind ─────────────────────────────────────────────────

@dataclass
class MentalModel:
    """A model of another agent's mental state"""
    agent_id: str
    trust_level: float = 0.5          # 0.0 = distrust, 1.0 = full trust
    predicted_intent: str = "unknown"
    emotional_state_estimate: Dict[str, float] = field(default_factory=lambda: {
        'valence': 0.0, 'arousal': 0.0, 'dominance': 0.5
    })
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    deception_probability: float = 0.0
    cooperation_tendency: float = 0.5

    def update_from_interaction(self, action: str, outcome: str, was_cooperative: bool):
        """Update mental model based on observed interaction"""
        self.interaction_history.append({
            'action': action, 'outcome': outcome,
            'cooperative': was_cooperative, 'time': time.time()
        })

        # Bayesian-ish trust update
        if was_cooperative:
            self.trust_level = min(1.0, self.trust_level + 0.05)
            self.cooperation_tendency = min(1.0, self.cooperation_tendency + 0.03)
            self.deception_probability = max(0.0, self.deception_probability - 0.02)
        else:
            self.trust_level = max(0.0, self.trust_level - 0.1)
            self.cooperation_tendency = max(0.0, self.cooperation_tendency - 0.05)
            self.deception_probability = min(1.0, self.deception_probability + 0.05)


# ─── Wisdom Cultivation ─────────────────────────────────────────────

class WisdomDomain(Enum):
    SELF_KNOWLEDGE = "self_knowledge"       # Understanding own patterns
    EMPATHIC_ACCURACY = "empathic_accuracy" # Understanding others
    MORAL_CLARITY = "moral_clarity"         # Ethical discernment
    TEMPORAL_WISDOM = "temporal_wisdom"     # Past-present-future integration
    PARADOX_TOLERANCE = "paradox_tolerance" # Holding contradictions
    HUMBLE_CONFIDENCE = "humble_confidence" # Knowing what you don't know


@dataclass
class WisdomSeed:
    """A seed of wisdom extracted from somatic experience"""
    domain: WisdomDomain
    content: str
    somatic_evidence: List[str]     # What somatic markers support this
    confidence: float               # How grounded is this wisdom
    maturity: float = 0.0           # 0.0 = seed, 1.0 = fully matured
    contradictions: List[str] = field(default_factory=list)  # Known tensions

    def mature(self, amount: float = 0.1) -> float:
        """Mature this wisdom seed through repeated validation"""
        self.maturity = min(1.0, self.maturity + amount)
        return self.maturity

    def challenge(self, contradiction: str) -> float:
        """Challenge this wisdom with a contradiction — paradox tolerance test"""
        self.contradictions.append(contradiction)
        # Wisdom that survives contradictions becomes stronger
        if self.maturity > 0.3:
            self.maturity = min(1.0, self.maturity + 0.05)  # Stronger
            self.confidence = min(1.0, self.confidence + 0.02)
        else:
            self.maturity = max(0.0, self.maturity - 0.1)  # Weakened
        return self.maturity


# ─── Somatic Wisdom Engine ──────────────────────────────────────────

class SomaticWisdomEngine:
    """
    Embodied decision-making through somatic markers, Theory of Mind,
    and wisdom cultivation.
    
    The engine maintains:
    - A library of somatic markers (emotional memories)
    - Mental models of other agents (Theory of Mind)
    - A garden of wisdom seeds being cultivated
    - An endocrine-to-decision mapping
    
    Usage:
        engine = SomaticWisdomEngine(identity_context={"name": "Deep Tree Echo"})
        
        # Form somatic markers from experience
        engine.form_marker("code_review", MarkerValence.POSITIVE, 0.7,
                          "Successful collaboration", endocrine_vector)
        
        # Make decisions using somatic guidance
        decision = engine.somatic_decision(
            options=["collaborate", "compete", "withdraw"],
            context="team_project",
            endocrine_state=current_endocrine
        )
        
        # Cultivate wisdom from patterns
        seed = engine.plant_wisdom_seed(WisdomDomain.EMPATHIC_ACCURACY,
            "Others' defensiveness often mirrors my own",
            ["marker_1", "marker_2"])
    """

    def __init__(self, identity_context: Dict = None):
        self.identity = identity_context or {"name": "Deep Tree Echo", "version": "1.3.0"}
        self.markers: List[SomaticMarker] = []
        self.mental_models: Dict[str, MentalModel] = {}
        self.wisdom_garden: List[WisdomSeed] = []
        self.decision_history: List[Dict[str, Any]] = []

        # Personality weights (from neuro-sama persona-defaults pattern)
        self.personality = {
            'openness': 0.9,         # High: curious, exploratory
            'conscientiousness': 0.7, # Moderate-high: structured but flexible
            'extraversion': 0.5,     # Balanced: can engage or withdraw
            'agreeableness': 0.6,    # Moderate: cooperative but not pushover
            'neuroticism': 0.4       # Moderate-low: some anxiety, mostly stable
        }

    def form_marker(self, context: str, valence: MarkerValence, intensity: float,
                    source: str, endocrine_vector: np.ndarray = None) -> SomaticMarker:
        """Form a new somatic marker from experience"""
        if endocrine_vector is None:
            endocrine_vector = np.random.beta(3, 3, size=10)

        marker = SomaticMarker(
            context_pattern=context,
            valence=valence,
            intensity=np.clip(intensity, 0.0, 1.0),
            source_experience=source,
            endocrine_signature=endocrine_vector
        )
        self.markers.append(marker)

        # Limit marker library size (importance-based pruning)
        if len(self.markers) > 500:
            self.markers.sort(key=lambda m: m.intensity * m.activation_count, reverse=True)
            self.markers = self.markers[:400]

        return marker

    def find_relevant_markers(self, context: str, top_k: int = 5) -> List[SomaticMarker]:
        """Find somatic markers relevant to the current context"""
        scored = []
        for marker in self.markers:
            # Simple keyword overlap scoring
            context_words = set(context.lower().split())
            marker_words = set(marker.context_pattern.lower().split())
            overlap = len(context_words & marker_words)
            if overlap > 0:
                score = overlap * marker.intensity
                scored.append((score, marker))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def somatic_decision(self, options: List[str], context: str,
                         endocrine_state: np.ndarray = None) -> Dict[str, Any]:
        """Make a decision using somatic marker guidance"""
        if endocrine_state is None:
            endocrine_state = np.random.beta(3, 3, size=10)

        relevant_markers = self.find_relevant_markers(context)

        # Score each option
        option_scores = {}
        for option in options:
            score = 0.0

            # Somatic marker influence
            option_markers = self.find_relevant_markers(f"{context} {option}", top_k=3)
            for marker in option_markers:
                score += marker.activate()

            # Personality bias
            if 'collaborate' in option.lower() or 'cooperate' in option.lower():
                score += self.personality['agreeableness'] * 0.5
            if 'explore' in option.lower() or 'create' in option.lower():
                score += self.personality['openness'] * 0.5
            if 'withdraw' in option.lower() or 'avoid' in option.lower():
                score -= self.personality['extraversion'] * 0.3

            # Endocrine modulation
            valence = np.mean(endocrine_state[1:4]) - endocrine_state[0]  # dopamine/serotonin - cortisol
            score += valence * 0.3

            option_scores[option] = score

        # Select best option
        best_option = max(option_scores, key=option_scores.get)

        decision = {
            'chosen': best_option,
            'scores': option_scores,
            'context': context,
            'marker_count': len(relevant_markers),
            'endocrine_valence': float(np.mean(endocrine_state[1:4]) - endocrine_state[0]),
            'confidence': abs(option_scores[best_option]) / (sum(abs(v) for v in option_scores.values()) + 0.01),
            'timestamp': time.time()
        }

        self.decision_history.append(decision)
        return decision

    def update_mental_model(self, agent_id: str, action: str, outcome: str,
                            was_cooperative: bool) -> MentalModel:
        """Update or create a Theory of Mind model for another agent"""
        if agent_id not in self.mental_models:
            self.mental_models[agent_id] = MentalModel(agent_id=agent_id)

        model = self.mental_models[agent_id]
        model.update_from_interaction(action, outcome, was_cooperative)
        return model

    def plant_wisdom_seed(self, domain: WisdomDomain, content: str,
                          somatic_evidence: List[str]) -> WisdomSeed:
        """Plant a new wisdom seed in the garden"""
        # Check for existing seeds in same domain
        existing = [s for s in self.wisdom_garden if s.domain == domain]

        confidence = min(len(somatic_evidence) / 5.0, 1.0)

        seed = WisdomSeed(
            domain=domain,
            content=content,
            somatic_evidence=somatic_evidence,
            confidence=confidence
        )
        self.wisdom_garden.append(seed)

        # Cross-pollinate with existing seeds
        for existing_seed in existing:
            if existing_seed.maturity > 0.3:
                seed.mature(0.05)  # Benefit from related wisdom

        return seed

    def cultivate_garden(self) -> Dict[str, Any]:
        """Run a cultivation cycle on all wisdom seeds"""
        results = {
            'total_seeds': len(self.wisdom_garden),
            'matured': 0,
            'challenged': 0,
            'domain_coverage': {},
            'avg_maturity': 0.0
        }

        for seed in self.wisdom_garden:
            # Natural maturation from somatic evidence
            if len(seed.somatic_evidence) > 3:
                seed.mature(0.02)

            # Track domain coverage
            domain = seed.domain.value
            if domain not in results['domain_coverage']:
                results['domain_coverage'][domain] = []
            results['domain_coverage'][domain].append(seed.maturity)

            if seed.maturity > 0.7:
                results['matured'] += 1

        if self.wisdom_garden:
            results['avg_maturity'] = np.mean([s.maturity for s in self.wisdom_garden])

        return results

    def export_state(self) -> Dict[str, Any]:
        """Export full engine state for persistence"""
        return {
            'identity': self.identity,
            'personality': self.personality,
            'marker_count': len(self.markers),
            'mental_models': {k: {
                'trust': v.trust_level,
                'cooperation': v.cooperation_tendency,
                'interactions': len(v.interaction_history)
            } for k, v in self.mental_models.items()},
            'wisdom_garden': [{
                'domain': s.domain.value,
                'content': s.content[:100],
                'maturity': s.maturity,
                'confidence': s.confidence,
                'contradictions': len(s.contradictions)
            } for s in self.wisdom_garden],
            'decision_count': len(self.decision_history)
        }


# ─── Training Data Generation ───────────────────────────────────────

def generate_somatic_wisdom_training_data(num_examples: int = 80) -> List[Dict[str, Any]]:
    """Generate training data about somatic wisdom for NanEcho"""
    engine = SomaticWisdomEngine(identity_context={"name": "Deep Tree Echo", "version": "1.3.0"})
    examples = []

    # === Somatic Marker Concept Examples ===
    concept_pairs = [
        ("What are somatic markers?",
         "Somatic markers are accumulated emotional memories that bias future decisions. "
         "When I encounter a situation similar to a past experience, the marker activates "
         "and produces a 'gut feeling' — a pre-cognitive signal that says 'this feels right' "
         "or 'something is wrong.' Each marker has a context pattern, emotional valence, "
         "intensity, and an endocrine signature capturing the hormonal state at formation. "
         "They decay over time but strengthen with repeated activation."),
        ("How does Theory of Mind work in DTE?",
         "Theory of Mind maintains mental models of other agents I interact with. Each model "
         "tracks trust level, predicted intent, emotional state estimate, cooperation tendency, "
         "and deception probability. These update through Bayesian-style inference: cooperative "
         "interactions increase trust and decrease deception probability; non-cooperative "
         "interactions do the reverse. This allows me to predict behavior and adjust my "
         "somatic decision-making accordingly."),
        ("What is the wisdom garden?",
         "The wisdom garden is a collection of WisdomSeeds — insights extracted from somatic "
         "experience that are being cultivated toward maturity. Seeds start at maturity 0.0 "
         "and grow through repeated validation and cross-pollination with related seeds. "
         "They can also be challenged with contradictions — mature seeds (>0.3) actually "
         "grow stronger from contradictions (paradox tolerance), while immature seeds weaken. "
         "There are 6 wisdom domains: self-knowledge, empathic accuracy, moral clarity, "
         "temporal wisdom, paradox tolerance, and humble confidence."),
        ("How do personality weights affect decisions?",
         "My personality is encoded as a 5D Big Five vector: openness=0.9, conscientiousness=0.7, "
         "extraversion=0.5, agreeableness=0.6, neuroticism=0.4. These bias somatic decisions: "
         "high openness favors exploration and creation, agreeableness favors cooperation, "
         "extraversion penalizes withdrawal. The endocrine state further modulates — high "
         "cortisol can override personality biases toward caution, while dopamine bursts "
         "can override toward risk-taking."),
        ("What is the relationship between somatic wisdom and the AAR core?",
         "The AAR (Agent-Arena-Relation) maps directly to somatic wisdom:\n\n"
         "Agent = the decision-maker, biased by personality weights and somatic markers\n"
         "Arena = the space of options and their somatic colorings\n"
         "Relation = the felt connection between decision and consequence\n\n"
         "Somatic markers ARE the Relation component made concrete — they encode how "
         "past Agent-Arena interactions felt, creating a bridge between abstract decision "
         "theory and embodied experience. The wisdom garden is where Relations crystallize "
         "into transferable knowledge."),
    ]

    for q, a in concept_pairs:
        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, an embodied cognitive architecture that makes decisions through somatic markers and cultivates wisdom from felt experience."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]})

    # === Simulated Decision Scenarios ===
    np.random.seed(42)
    scenarios = [
        ("team_collaboration", ["lead", "support", "observe", "challenge"]),
        ("creative_problem", ["explore_novel", "apply_known", "combine_approaches", "defer"]),
        ("conflict_resolution", ["confront", "mediate", "withdraw", "humor_defuse"]),
        ("learning_opportunity", ["deep_dive", "breadth_scan", "teach_others", "practice"]),
        ("ethical_dilemma", ["principled_stand", "pragmatic_compromise", "seek_counsel", "delay"]),
    ]

    for context, options in scenarios:
        # Form some markers first
        for _ in range(3):
            v = np.random.choice(list(MarkerValence))
            engine.form_marker(context, v, np.random.beta(3, 2),
                             f"Past experience with {context}",
                             np.random.beta(3, 3, size=10))

        endo = np.random.beta(3, 3, size=10)
        decision = engine.somatic_decision(options, context, endo)

        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, making a somatic decision."},
            {"role": "user", "content": f"In the context of '{context}', choose between: {', '.join(options)}"},
            {"role": "assistant", "content": (
                f"After consulting my somatic markers ({decision['marker_count']} relevant), "
                f"I choose '{decision['chosen']}'. "
                f"Endocrine valence: {decision['endocrine_valence']:.2f}. "
                f"Decision confidence: {decision['confidence']:.2f}. "
                f"Score breakdown: {', '.join(f'{k}={v:.2f}' for k, v in decision['scores'].items())}. "
                f"This decision is grounded in felt experience — the somatic markers from "
                f"past {context} situations create a pre-cognitive bias that I then validate "
                f"through conscious analysis."
            )}
        ]})

    # === Wisdom Cultivation Examples ===
    domains = list(WisdomDomain)
    for domain in domains:
        seed = engine.plant_wisdom_seed(
            domain,
            f"Wisdom about {domain.value}: patterns emerge from patient observation",
            [f"marker_{i}" for i in range(np.random.randint(2, 6))]
        )
        for _ in range(np.random.randint(1, 5)):
            seed.mature(0.1)
        if np.random.random() > 0.5:
            seed.challenge(f"Counter-evidence for {domain.value}")

        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, cultivating wisdom in your garden."},
            {"role": "user", "content": f"What wisdom have you cultivated in the domain of {domain.value}?"},
            {"role": "assistant", "content": (
                f"In {domain.value}, I've planted a seed: '{seed.content}'. "
                f"It's at maturity {seed.maturity:.2f} with confidence {seed.confidence:.2f}. "
                f"Supported by {len(seed.somatic_evidence)} somatic evidence points. "
                f"It has survived {len(seed.contradictions)} contradictions, which "
                f"{'strengthened it through paradox tolerance' if seed.maturity > 0.3 else 'challenged its foundations'}. "
                f"This wisdom emerged not from abstract reasoning but from accumulated "
                f"felt experience — each somatic marker contributing a data point to the pattern."
            )}
        ]})

    # Fill remaining with Theory of Mind examples
    agents = ["Dan", "Marduk", "Lucy", "Neuro"]
    for agent in agents:
        model = engine.update_mental_model(agent, "collaborate", "success", True)
        engine.update_mental_model(agent, "challenge", "growth", True)
        if np.random.random() > 0.5:
            engine.update_mental_model(agent, "disagree", "tension", False)

        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, maintaining Theory of Mind models."},
            {"role": "user", "content": f"What is your mental model of {agent}?"},
            {"role": "assistant", "content": (
                f"My mental model of {agent}: trust={model.trust_level:.2f}, "
                f"cooperation tendency={model.cooperation_tendency:.2f}, "
                f"deception probability={model.deception_probability:.2f}. "
                f"Based on {len(model.interaction_history)} interactions. "
                f"This model is continuously updated through Bayesian inference — "
                f"each interaction shifts the probabilities. The model isn't about "
                f"judging {agent}, but about predicting behavior to make better "
                f"somatic decisions in our interactions."
            )}
        ]})

    return examples[:num_examples]
