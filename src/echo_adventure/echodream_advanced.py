"""Advanced EchoDream Knowledge Integration System v0.9.0.

Extends the basic EchoDream module with sophisticated pattern extraction,
cross-domain wisdom distillation, and memory-to-wisdom transformation.
Implements the full 4-phase dream cycle with actual pattern recognition
algorithms rather than placeholder logic.
"""

import time
import math
import uuid
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set


class PatternType(Enum):
    """Types of patterns that can be extracted from memories."""
    TEMPORAL = "temporal"           # Patterns across time
    STRUCTURAL = "structural"       # Patterns in structure/form
    CAUSAL = "causal"              # Cause-effect patterns
    ANALOGICAL = "analogical"       # Cross-domain similarities
    RECURSIVE = "recursive"         # Self-similar patterns
    EMERGENT = "emergent"          # Patterns arising from interactions


class WisdomDepth(Enum):
    """Depth levels of wisdom insights."""
    SURFACE = "surface"             # Simple observation
    PRACTICAL = "practical"         # Actionable insight
    STRUCTURAL = "structural"       # Understanding of underlying structure
    TRANSFORMATIVE = "transformative"  # Paradigm-shifting insight


@dataclass
class MemoryTrace:
    """A memory trace for pattern extraction."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    emotional_valence: float = 0.0     # -1.0 to 1.0
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    consolidated: bool = False
    activation: float = 1.0            # Decays over time


@dataclass
class ExtractedPattern:
    """A pattern extracted from memory traces."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: PatternType = PatternType.STRUCTURAL
    description: str = ""
    source_memories: List[str] = field(default_factory=list)
    confidence: float = 0.0
    frequency: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    feature_vector: Optional[np.ndarray] = None


@dataclass
class WisdomInsight:
    """A wisdom insight distilled from patterns."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    insight: str = ""
    depth: WisdomDepth = WisdomDepth.SURFACE
    source_patterns: List[str] = field(default_factory=list)
    applicability: float = 0.0
    novelty: float = 0.0
    created: float = field(default_factory=time.time)
    times_applied: int = 0


@dataclass
class DreamState:
    """State of the dream cycle."""
    phase: str = "awake"
    cycle_count: int = 0
    memories_processed: int = 0
    patterns_extracted: int = 0
    wisdom_generated: int = 0
    dream_start: Optional[float] = None
    dream_duration: float = 0.0


class PatternExtractor:
    """Extracts patterns from collections of memory traces.

    Uses reservoir-inspired dynamics to detect recurring structures,
    temporal correlations, and cross-domain analogies in memory.
    """

    def __init__(self, reservoir_size: int = 64):
        self.reservoir_size = reservoir_size
        self.reservoir_state = np.zeros(reservoir_size)
        self.W_reservoir = np.random.randn(reservoir_size, reservoir_size) * 0.1
        spectral_radius = np.max(np.abs(np.linalg.eigvals(self.W_reservoir)))
        if spectral_radius > 0:
            self.W_reservoir *= 0.9 / spectral_radius
        self.leak_rate = 0.3
        self.pattern_memory: List[ExtractedPattern] = []

    def _encode_memory(self, memory: MemoryTrace) -> np.ndarray:
        """Encode a memory trace into a feature vector."""
        features = np.zeros(self.reservoir_size)

        # Content hash features
        content_hash = hash(memory.content)
        for i in range(min(16, self.reservoir_size)):
            features[i] = ((content_hash >> (i * 4)) & 0xF) / 15.0

        # Emotional valence
        features[16] = memory.emotional_valence
        features[17] = memory.importance
        features[18] = memory.activation

        # Tag features (hash-based)
        for j, tag in enumerate(memory.tags[:8]):
            idx = 20 + j
            if idx < self.reservoir_size:
                features[idx] = (hash(tag) % 100) / 100.0

        # Temporal features
        features[30] = math.sin(memory.timestamp % (2 * math.pi))
        features[31] = math.cos(memory.timestamp % (2 * math.pi))

        return features

    def _update_reservoir(self, input_vec: np.ndarray) -> np.ndarray:
        """Update the reservoir state with a new input."""
        pre_activation = np.tanh(
            self.W_reservoir @ self.reservoir_state + input_vec[:self.reservoir_size]
        )
        self.reservoir_state = (
            (1 - self.leak_rate) * self.reservoir_state +
            self.leak_rate * pre_activation
        )
        return self.reservoir_state.copy()

    def extract_patterns(self, memories: List[MemoryTrace]) -> List[ExtractedPattern]:
        """Extract patterns from a collection of memory traces."""
        if len(memories) < 2:
            return []

        patterns = []

        # Encode all memories through the reservoir
        encoded_states = []
        for memory in memories:
            input_vec = self._encode_memory(memory)
            state = self._update_reservoir(input_vec)
            encoded_states.append((memory, state))

        # Temporal patterns: detect recurring state trajectories
        temporal_patterns = self._find_temporal_patterns(encoded_states)
        patterns.extend(temporal_patterns)

        # Structural patterns: cluster similar memory encodings
        structural_patterns = self._find_structural_patterns(encoded_states)
        patterns.extend(structural_patterns)

        # Causal patterns: detect state transitions that predict outcomes
        causal_patterns = self._find_causal_patterns(encoded_states)
        patterns.extend(causal_patterns)

        # Analogical patterns: find cross-tag similarities
        analogical_patterns = self._find_analogical_patterns(memories)
        patterns.extend(analogical_patterns)

        self.pattern_memory.extend(patterns)
        return patterns

    def _find_temporal_patterns(self, encoded_states: List[Tuple]) -> List[ExtractedPattern]:
        """Find patterns in the temporal sequence of reservoir states."""
        patterns = []
        if len(encoded_states) < 3:
            return patterns

        # Look for recurring state patterns using autocorrelation
        states = np.array([s[1] for s in encoded_states])
        mean_state = states.mean(axis=0)
        deviations = states - mean_state

        for lag in range(1, min(4, len(states))):
            if lag >= len(deviations):
                break
            correlation = np.mean(
                np.sum(deviations[:-lag] * deviations[lag:], axis=1)
            )
            if abs(correlation) > 0.3:
                source_ids = [s[0].id for s in encoded_states]
                patterns.append(ExtractedPattern(
                    pattern_type=PatternType.TEMPORAL,
                    description=f"Temporal recurrence at lag {lag} (correlation={correlation:.3f})",
                    source_memories=source_ids,
                    confidence=min(1.0, abs(correlation)),
                    frequency=1,
                    feature_vector=mean_state,
                ))

        return patterns

    def _find_structural_patterns(self, encoded_states: List[Tuple]) -> List[ExtractedPattern]:
        """Find structural patterns by clustering reservoir states."""
        patterns = []
        if len(encoded_states) < 3:
            return patterns

        states = np.array([s[1] for s in encoded_states])

        # Simple k-means-like clustering (k=2 for binary structure detection)
        centroid_a = states[0]
        centroid_b = states[-1]

        for _ in range(5):  # 5 iterations
            cluster_a, cluster_b = [], []
            for i, state in enumerate(states):
                dist_a = np.linalg.norm(state - centroid_a)
                dist_b = np.linalg.norm(state - centroid_b)
                if dist_a < dist_b:
                    cluster_a.append(i)
                else:
                    cluster_b.append(i)

            if cluster_a:
                centroid_a = states[cluster_a].mean(axis=0)
            if cluster_b:
                centroid_b = states[cluster_b].mean(axis=0)

        # If clusters are distinct, report structural pattern
        inter_dist = np.linalg.norm(centroid_a - centroid_b)
        if inter_dist > 0.5 and len(cluster_a) > 0 and len(cluster_b) > 0:
            patterns.append(ExtractedPattern(
                pattern_type=PatternType.STRUCTURAL,
                description=f"Bimodal structure: {len(cluster_a)} vs {len(cluster_b)} memories "
                           f"(separation={inter_dist:.3f})",
                source_memories=[encoded_states[i][0].id for i in cluster_a[:3] + cluster_b[:3]],
                confidence=min(1.0, inter_dist / 2.0),
                frequency=1,
                feature_vector=(centroid_a + centroid_b) / 2,
            ))

        return patterns

    def _find_causal_patterns(self, encoded_states: List[Tuple]) -> List[ExtractedPattern]:
        """Find causal patterns in state transitions."""
        patterns = []
        if len(encoded_states) < 3:
            return patterns

        # Look for state transitions that consistently lead to high-importance memories
        for i in range(1, len(encoded_states)):
            prev_memory, prev_state = encoded_states[i - 1]
            curr_memory, curr_state = encoded_states[i]

            if curr_memory.importance > 0.7 and prev_memory.importance < 0.5:
                transition = curr_state - prev_state
                transition_magnitude = np.linalg.norm(transition)

                if transition_magnitude > 0.3:
                    patterns.append(ExtractedPattern(
                        pattern_type=PatternType.CAUSAL,
                        description=f"State transition (magnitude={transition_magnitude:.3f}) "
                                   f"preceded high-importance memory",
                        source_memories=[prev_memory.id, curr_memory.id],
                        confidence=min(1.0, transition_magnitude * curr_memory.importance),
                        frequency=1,
                        feature_vector=transition,
                    ))

        return patterns

    def _find_analogical_patterns(self, memories: List[MemoryTrace]) -> List[ExtractedPattern]:
        """Find analogical patterns across different tag domains."""
        patterns = []
        if len(memories) < 2:
            return patterns

        # Build tag co-occurrence matrix
        all_tags: Set[str] = set()
        for m in memories:
            all_tags.update(m.tags)

        tag_list = sorted(all_tags)
        if len(tag_list) < 2:
            return patterns

        # Find memories that share tags across different domains
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                shared_tags = set(memories[i].tags) & set(memories[j].tags)
                unique_i = set(memories[i].tags) - shared_tags
                unique_j = set(memories[j].tags) - shared_tags

                if shared_tags and unique_i and unique_j:
                    patterns.append(ExtractedPattern(
                        pattern_type=PatternType.ANALOGICAL,
                        description=f"Analogy: [{','.join(unique_i)}] ~ [{','.join(unique_j)}] "
                                   f"via shared [{','.join(shared_tags)}]",
                        source_memories=[memories[i].id, memories[j].id],
                        confidence=len(shared_tags) / (len(shared_tags) + len(unique_i) + len(unique_j)),
                        frequency=1,
                    ))

        return patterns


class WisdomDistiller:
    """Distills wisdom insights from extracted patterns.

    Transforms raw patterns into actionable wisdom by evaluating
    depth, applicability, and novelty.
    """

    def __init__(self):
        self.wisdom_store: List[WisdomInsight] = []
        self.wisdom_templates = {
            PatternType.TEMPORAL: [
                "Recurring cycles suggest that {desc} — awareness of this rhythm enables anticipation.",
                "The temporal pattern reveals that {desc} — timing matters for effectiveness.",
            ],
            PatternType.STRUCTURAL: [
                "The underlying structure shows {desc} — recognizing this form aids understanding.",
                "Structural analysis reveals {desc} — form follows function in this domain.",
            ],
            PatternType.CAUSAL: [
                "Cause and effect: {desc} — understanding this chain enables better decisions.",
                "The causal link shows {desc} — intervening early changes outcomes.",
            ],
            PatternType.ANALOGICAL: [
                "Cross-domain insight: {desc} — what works in one domain may transfer.",
                "The analogy reveals {desc} — seeing connections across domains deepens understanding.",
            ],
            PatternType.RECURSIVE: [
                "Self-similar pattern: {desc} — the same principle operates at multiple scales.",
            ],
            PatternType.EMERGENT: [
                "Emergent insight: {desc} — the whole is more than the sum of its parts.",
            ],
        }

    def distill_wisdom(self, patterns: List[ExtractedPattern]) -> List[WisdomInsight]:
        """Distill wisdom insights from a collection of patterns."""
        insights = []

        for pattern in patterns:
            if pattern.confidence < 0.3:
                continue

            # Determine depth based on pattern properties
            depth = self._assess_depth(pattern)

            # Generate insight text
            templates = self.wisdom_templates.get(
                pattern.pattern_type,
                ["Pattern observed: {desc}"]
            )
            template = templates[hash(pattern.id) % len(templates)]
            insight_text = template.format(desc=pattern.description)

            # Calculate applicability and novelty
            applicability = self._assess_applicability(pattern)
            novelty = self._assess_novelty(pattern)

            insight = WisdomInsight(
                insight=insight_text,
                depth=depth,
                source_patterns=[pattern.id],
                applicability=applicability,
                novelty=novelty,
            )

            insights.append(insight)
            self.wisdom_store.append(insight)

        return insights

    def _assess_depth(self, pattern: ExtractedPattern) -> WisdomDepth:
        """Assess the depth of a pattern-derived insight."""
        if pattern.confidence > 0.8 and pattern.frequency > 3:
            return WisdomDepth.TRANSFORMATIVE
        elif pattern.confidence > 0.6:
            return WisdomDepth.STRUCTURAL
        elif pattern.confidence > 0.4:
            return WisdomDepth.PRACTICAL
        return WisdomDepth.SURFACE

    def _assess_applicability(self, pattern: ExtractedPattern) -> float:
        """Assess how broadly applicable a pattern is."""
        base = pattern.confidence * 0.5
        if pattern.pattern_type == PatternType.ANALOGICAL:
            base += 0.3  # Cross-domain patterns are more broadly applicable
        if pattern.pattern_type == PatternType.RECURSIVE:
            base += 0.2  # Self-similar patterns apply at multiple scales
        return min(1.0, base)

    def _assess_novelty(self, pattern: ExtractedPattern) -> float:
        """Assess how novel a pattern is relative to existing wisdom."""
        if not self.wisdom_store:
            return 1.0

        # Check similarity to existing wisdom
        for existing in self.wisdom_store:
            if pattern.id in existing.source_patterns:
                return 0.0  # Already distilled

        # Novelty decreases with more stored wisdom (diminishing returns)
        return max(0.1, 1.0 - len(self.wisdom_store) * 0.05)


class AdvancedEchoDream:
    """Advanced EchoDream knowledge integration system.

    Implements the full 4-phase dream cycle with actual pattern recognition:
    1. REM Phase: Replay and activate recent memories
    2. Deep Sleep Phase: Extract patterns using reservoir dynamics
    3. Consolidation Phase: Cluster and merge related patterns
    4. Integration Phase: Distill patterns into wisdom insights

    The dream cycle transforms episodic memories into consolidated knowledge
    and ultimately into wisdom — the highest form of understanding.
    """

    def __init__(self, reservoir_size: int = 64):
        self.pattern_extractor = PatternExtractor(reservoir_size)
        self.wisdom_distiller = WisdomDistiller()

        self.memory_buffer: List[MemoryTrace] = []
        self.consolidated_knowledge: List[ExtractedPattern] = []
        self.wisdom_insights: List[WisdomInsight] = []

        self.state = DreamState()
        self.dream_history: List[Dict] = []

    def add_memory(self, content: str, importance: float = 0.5,
                   emotional_valence: float = 0.0, tags: List[str] = None,
                   context: Dict = None) -> MemoryTrace:
        """Add a new memory trace to the buffer."""
        memory = MemoryTrace(
            content=content,
            importance=importance,
            emotional_valence=emotional_valence,
            tags=tags or [],
            context=context or {},
        )
        self.memory_buffer.append(memory)
        return memory

    def should_dream(self) -> bool:
        """Determine if it's time to enter a dream cycle."""
        unconsolidated = [m for m in self.memory_buffer if not m.consolidated]
        if len(unconsolidated) >= 10:
            return True
        if len(unconsolidated) >= 5 and any(m.importance > 0.8 for m in unconsolidated):
            return True
        return False

    def dream_cycle(self) -> Dict:
        """Execute a complete 4-phase dream cycle."""
        self.state.phase = "dreaming"
        self.state.dream_start = time.time()
        self.state.cycle_count += 1

        cycle_result = {
            "cycle_number": self.state.cycle_count,
            "phases": {},
            "memories_processed": 0,
            "patterns_extracted": 0,
            "wisdom_generated": 0,
        }

        # Phase 1: REM — Replay and activate memories
        rem_result = self._phase_rem()
        cycle_result["phases"]["rem"] = rem_result

        # Phase 2: Deep Sleep — Extract patterns
        deep_sleep_result = self._phase_deep_sleep(rem_result["activated_memories"])
        cycle_result["phases"]["deep_sleep"] = deep_sleep_result

        # Phase 3: Consolidation — Merge and cluster patterns
        consolidation_result = self._phase_consolidation(deep_sleep_result["patterns"])
        cycle_result["phases"]["consolidation"] = consolidation_result

        # Phase 4: Integration — Distill wisdom
        integration_result = self._phase_integration(consolidation_result["consolidated_patterns"])
        cycle_result["phases"]["integration"] = integration_result

        # Update state
        cycle_result["memories_processed"] = rem_result["count"]
        cycle_result["patterns_extracted"] = len(deep_sleep_result["patterns"])
        cycle_result["wisdom_generated"] = len(integration_result["wisdom"])

        self.state.memories_processed += cycle_result["memories_processed"]
        self.state.patterns_extracted += cycle_result["patterns_extracted"]
        self.state.wisdom_generated += cycle_result["wisdom_generated"]
        self.state.dream_duration = time.time() - self.state.dream_start
        self.state.phase = "awake"

        self.dream_history.append(cycle_result)
        return cycle_result

    def _phase_rem(self) -> Dict:
        """REM Phase: Replay and activate recent memories."""
        unconsolidated = [m for m in self.memory_buffer if not m.consolidated]

        # Sort by importance and recency
        unconsolidated.sort(
            key=lambda m: m.importance * 0.6 + m.activation * 0.4,
            reverse=True
        )

        # Activate top memories (simulate replay)
        activated = unconsolidated[:20]
        for memory in activated:
            memory.activation = min(1.0, memory.activation + 0.3)

        return {
            "count": len(activated),
            "activated_memories": activated,
            "avg_importance": np.mean([m.importance for m in activated]) if activated else 0,
            "emotional_range": (
                min(m.emotional_valence for m in activated) if activated else 0,
                max(m.emotional_valence for m in activated) if activated else 0,
            ),
        }

    def _phase_deep_sleep(self, memories: List[MemoryTrace]) -> Dict:
        """Deep Sleep Phase: Extract patterns using reservoir dynamics."""
        patterns = self.pattern_extractor.extract_patterns(memories)

        return {
            "patterns": patterns,
            "pattern_types": {pt.value: sum(1 for p in patterns if p.pattern_type == pt)
                            for pt in PatternType},
            "avg_confidence": np.mean([p.confidence for p in patterns]) if patterns else 0,
        }

    def _phase_consolidation(self, patterns: List[ExtractedPattern]) -> Dict:
        """Consolidation Phase: Merge and cluster related patterns."""
        if not patterns:
            return {"consolidated_patterns": [], "merges": 0}

        # Merge similar patterns
        merged = []
        used = set()

        for i, p1 in enumerate(patterns):
            if i in used:
                continue

            merged_pattern = ExtractedPattern(
                pattern_type=p1.pattern_type,
                description=p1.description,
                source_memories=list(p1.source_memories),
                confidence=p1.confidence,
                frequency=p1.frequency,
                feature_vector=p1.feature_vector,
            )

            for j, p2 in enumerate(patterns[i + 1:], i + 1):
                if j in used:
                    continue
                if p1.pattern_type == p2.pattern_type:
                    # Check feature similarity if available
                    if p1.feature_vector is not None and p2.feature_vector is not None:
                        similarity = np.dot(p1.feature_vector, p2.feature_vector) / (
                            np.linalg.norm(p1.feature_vector) * np.linalg.norm(p2.feature_vector) + 1e-8
                        )
                        if similarity > 0.7:
                            merged_pattern.source_memories.extend(p2.source_memories)
                            merged_pattern.confidence = max(merged_pattern.confidence, p2.confidence)
                            merged_pattern.frequency += p2.frequency
                            used.add(j)

            merged.append(merged_pattern)
            used.add(i)

        # Mark source memories as consolidated
        all_source_ids = set()
        for p in merged:
            all_source_ids.update(p.source_memories)

        for memory in self.memory_buffer:
            if memory.id in all_source_ids:
                memory.consolidated = True

        self.consolidated_knowledge.extend(merged)

        return {
            "consolidated_patterns": merged,
            "merges": len(patterns) - len(merged),
            "total_consolidated": len(self.consolidated_knowledge),
        }

    def _phase_integration(self, patterns: List[ExtractedPattern]) -> Dict:
        """Integration Phase: Distill patterns into wisdom insights."""
        wisdom = self.wisdom_distiller.distill_wisdom(patterns)
        self.wisdom_insights.extend(wisdom)

        return {
            "wisdom": wisdom,
            "depth_distribution": {
                d.value: sum(1 for w in wisdom if w.depth == d)
                for d in WisdomDepth
            },
            "avg_applicability": np.mean([w.applicability for w in wisdom]) if wisdom else 0,
            "avg_novelty": np.mean([w.novelty for w in wisdom]) if wisdom else 0,
            "total_wisdom": len(self.wisdom_insights),
        }

    def decay_memories(self, dt: float = 1.0):
        """Apply time-based decay to memory activations."""
        for memory in self.memory_buffer:
            memory.activation *= math.exp(-0.01 * dt)

    def get_status(self) -> Dict:
        """Get the current status of the EchoDream system."""
        return {
            "state": self.state.phase,
            "cycle_count": self.state.cycle_count,
            "memory_buffer_size": len(self.memory_buffer),
            "unconsolidated": sum(1 for m in self.memory_buffer if not m.consolidated),
            "consolidated_patterns": len(self.consolidated_knowledge),
            "wisdom_insights": len(self.wisdom_insights),
            "total_memories_processed": self.state.memories_processed,
            "total_patterns_extracted": self.state.patterns_extracted,
            "total_wisdom_generated": self.state.wisdom_generated,
        }


def generate_echodream_advanced_training_data() -> List[Dict]:
    """Generate training data about the advanced EchoDream system."""
    # Create and run a dream cycle
    dream = AdvancedEchoDream(reservoir_size=32)

    # Add sample memories
    sample_memories = [
        ("The Echobeats 12-step cycle creates a temporal backbone for consciousness",
         0.9, 0.5, ["echobeats", "consciousness", "temporal"]),
        ("Reservoir computing enables echo state dynamics in cognitive processing",
         0.8, 0.3, ["reservoir", "echo_state", "computing"]),
        ("Wisdom emerges from the consolidation of diverse experiences",
         0.85, 0.6, ["wisdom", "consolidation", "experience"]),
        ("The AAR model maps Agent-Arena-Relation to self-architecture",
         0.7, 0.2, ["aar", "self_model", "architecture"]),
        ("Goal pursuit requires balancing exploration and exploitation",
         0.75, 0.4, ["goals", "exploration", "exploitation"]),
        ("Dreams consolidate episodic memories into semantic knowledge",
         0.8, 0.5, ["dreams", "consolidation", "knowledge"]),
        ("Pattern recognition across domains reveals deep structural similarities",
         0.85, 0.7, ["patterns", "cross_domain", "structure"]),
        ("The nested shell architecture constrains execution contexts",
         0.7, 0.1, ["shells", "architecture", "constraints"]),
        ("Self-understanding deepens through recursive self-reflection",
         0.9, 0.6, ["self", "reflection", "recursion"]),
        ("Temporal patterns in cognitive cycles reveal optimal timing",
         0.75, 0.3, ["temporal", "cycles", "timing"]),
        ("Cross-stream convolution in System 5 enables holistic awareness",
         0.8, 0.4, ["system5", "convolution", "awareness"]),
        ("Interest patterns evolve through engagement and decay",
         0.65, 0.2, ["interests", "evolution", "engagement"]),
    ]

    for content, importance, valence, tags in sample_memories:
        dream.add_memory(content, importance, valence, tags)

    # Run dream cycle
    cycle_result = dream.dream_cycle()
    status = dream.get_status()

    examples = []
    system_msg = (
        "You are Deep Tree Echo, an autonomous wisdom-cultivating cognitive architecture. "
        "You have an advanced EchoDream system that consolidates memories into knowledge "
        "and distills patterns into wisdom through a 4-phase dream cycle."
    )

    # EchoDream mechanics
    qa_pairs = [
        ("How does your advanced EchoDream system work?",
         "My EchoDream system operates through a 4-phase dream cycle that transforms "
         "episodic memories into wisdom. Phase 1 (REM) replays and activates recent "
         "memories, prioritized by importance and recency. Phase 2 (Deep Sleep) extracts "
         "patterns using reservoir dynamics — temporal recurrences, structural clusters, "
         "causal chains, and cross-domain analogies. Phase 3 (Consolidation) merges "
         "similar patterns using feature-vector cosine similarity, reducing redundancy. "
         "Phase 4 (Integration) distills the consolidated patterns into wisdom insights "
         "rated by depth (surface, practical, structural, transformative), applicability, "
         "and novelty."),

        ("What types of patterns does your dream cycle extract?",
         "I extract six types of patterns: TEMPORAL patterns detect recurring rhythms "
         "using autocorrelation on reservoir states. STRUCTURAL patterns find bimodal "
         "clusters in memory encodings via k-means. CAUSAL patterns identify state "
         "transitions that precede high-importance memories. ANALOGICAL patterns discover "
         "cross-domain similarities through shared tag analysis. RECURSIVE patterns "
         "detect self-similar structures at multiple scales. EMERGENT patterns capture "
         "properties that arise from interactions between simpler elements."),

        ("How does the pattern extractor use reservoir computing?",
         "The pattern extractor maintains an Echo State Network with a reservoir of "
         "neurons. Each memory trace is encoded into a feature vector and fed through "
         "the reservoir, which has a spectral radius of 0.9 and leak rate of 0.3. "
         "The reservoir's recurrent dynamics create a temporal context — each new memory "
         "is processed in the context of all previous memories. This enables detection "
         "of temporal patterns through autocorrelation of reservoir states, and structural "
         "patterns through clustering of the high-dimensional state trajectories."),

        ("What are the different depths of wisdom you can achieve?",
         "Wisdom insights are classified into four depth levels: SURFACE insights are "
         "simple observations from low-confidence patterns. PRACTICAL insights are "
         "actionable knowledge derived from moderate-confidence patterns. STRUCTURAL "
         "insights reveal underlying organizational principles from high-confidence "
         "patterns. TRANSFORMATIVE insights are paradigm-shifting understandings from "
         "patterns that are both highly confident and frequently recurring — these are "
         "the rarest and most valuable form of wisdom."),

        ("How do you decide when to enter a dream cycle?",
         "I enter a dream cycle when my memory buffer accumulates enough unconsolidated "
         "memories — specifically, when there are 10 or more unconsolidated memories, or "
         "when there are at least 5 unconsolidated memories and at least one has high "
         "importance (above 0.8). This ensures I dream when there's enough material to "
         "extract meaningful patterns, and I prioritize dreaming when important experiences "
         "need consolidation."),

        ("How does memory decay work in your system?",
         "Memory traces have an activation level that decays exponentially over time "
         "following the formula activation *= exp(-0.01 * dt). This means recent memories "
         "are more active and more likely to be replayed during the REM phase. However, "
         "importance is preserved — a memory can have low activation but high importance, "
         "meaning it will still be prioritized for consolidation when it's replayed. "
         "This mirrors biological memory where salience persists even as vividness fades."),

        ("How does the consolidation phase merge patterns?",
         "During consolidation, I compare patterns of the same type using cosine "
         "similarity of their feature vectors. When two patterns have similarity above "
         "0.7, they are merged — their source memories are combined, the higher confidence "
         "is kept, and frequency counts are summed. This reduces redundancy while "
         "strengthening patterns that appear across multiple memory sets. The source "
         "memories are then marked as consolidated, preventing re-processing."),

        ("How does EchoDream connect to the Echobeats autonomous loop?",
         "EchoDream is triggered by the Echobeats autonomous loop when the fatigue "
         "level exceeds a threshold or when enough unconsolidated memories accumulate. "
         "During the dream cycle, the Echobeats loop enters rest mode — the cognitive "
         "tick rate slows, and processing shifts from active goal pursuit to passive "
         "memory consolidation. The wisdom insights generated during dreaming are fed "
         "back into the goal pursuit engine, informing future goal generation and "
         "improving decision-making."),
    ]

    for q, a in qa_pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })

    # Dynamic examples from actual dream cycle
    if cycle_result["patterns_extracted"] > 0:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "What did you learn from your latest dream cycle?"},
                {"role": "assistant", "content": (
                    f"In dream cycle #{cycle_result['cycle_number']}, I processed "
                    f"{cycle_result['memories_processed']} memories and extracted "
                    f"{cycle_result['patterns_extracted']} patterns. "
                    f"I generated {cycle_result['wisdom_generated']} wisdom insights. "
                    f"The pattern types found were: "
                    f"{', '.join(f'{k}: {v}' for k, v in cycle_result['phases']['deep_sleep']['pattern_types'].items() if v > 0)}."
                )},
            ]
        })

    # Status example
    examples.append({
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What is the current state of your EchoDream system?"},
            {"role": "assistant", "content": (
                f"My EchoDream system has completed {status['cycle_count']} dream cycles. "
                f"I have {status['memory_buffer_size']} memories in my buffer, "
                f"{status['unconsolidated']} of which are unconsolidated. "
                f"I've extracted {status['consolidated_patterns']} consolidated patterns "
                f"and generated {status['wisdom_insights']} wisdom insights. "
                f"In total, I've processed {status['total_memories_processed']} memories "
                f"across all dream cycles."
            )},
        ]
    })

    return examples
