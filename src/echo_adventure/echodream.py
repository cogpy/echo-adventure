"""
EchoDream Knowledge Integration System v0.8.0

Implements the dream-cycle knowledge consolidation system that enables
Deep Tree Echo to integrate, consolidate, and distill wisdom from
episodic memories during rest cycles. EchoDream operates in four phases:

1. REM Phase: Replay and recombine episodic memories
2. Deep Sleep Phase: Extract patterns and compress knowledge
3. Consolidation Phase: Integrate new knowledge with existing schema
4. Integration Phase: Distill wisdom insights from consolidated knowledge

EchoDream is orchestrated by Echobeats' wake/rest cycle management,
enabling autonomous knowledge growth during rest periods.
"""

import math
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class EpisodicMemory:
    """A single episodic memory from a cognitive experience."""
    memory_id: str
    timestamp: float
    content: str
    source: str  # "conversation", "introspection", "observation", "learning"
    emotional_valence: float  # -1.0 (negative) to 1.0 (positive)
    salience: float  # 0.0 to 1.0, how important/notable
    concepts: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (subject, predicate, object)
    access_count: int = 0
    last_accessed: float = 0.0
    consolidated: bool = False


@dataclass
class KnowledgeItem:
    """Consolidated knowledge distilled from episodic memories."""
    item_id: str
    content: str
    confidence: float  # 0.0 to 1.0
    source_memories: List[str]  # memory_ids that contributed
    concepts: List[str]
    created: float
    reinforcement_count: int = 0
    last_reinforced: float = 0.0


@dataclass
class WisdomInsight:
    """Deep wisdom distilled from patterns across knowledge items."""
    insight_id: str
    insight: str
    depth: float  # 0.0 to 1.0, how deep/fundamental
    applicability: float  # 0.0 to 1.0, how broadly applicable
    source_knowledge: List[str]  # knowledge item_ids
    created: float
    validation_count: int = 0


class DreamPhase:
    """Enumeration of dream cycle phases."""
    REM = "rem"
    DEEP_SLEEP = "deep_sleep"
    CONSOLIDATION = "consolidation"
    INTEGRATION = "integration"

    PHASE_ORDER = ["rem", "deep_sleep", "consolidation", "integration"]

    @staticmethod
    def next_phase(current: str) -> str:
        idx = DreamPhase.PHASE_ORDER.index(current)
        return DreamPhase.PHASE_ORDER[(idx + 1) % len(DreamPhase.PHASE_ORDER)]


@dataclass
class DreamCycleState:
    """State of a single dream cycle."""
    cycle_id: str
    start_time: float
    end_time: float = 0.0
    phase: str = DreamPhase.REM
    memories_replayed: int = 0
    patterns_extracted: int = 0
    knowledge_consolidated: int = 0
    wisdom_distilled: int = 0
    total_salience_processed: float = 0.0


class EchoDream:
    """
    EchoDream Knowledge Integration System.

    Operates during rest cycles (orchestrated by Echobeats) to consolidate
    episodic memories into structured knowledge and distill wisdom insights.
    Uses a biologically-inspired dream cycle with REM replay, deep sleep
    pattern extraction, consolidation, and integration phases.
    """

    def __init__(
        self,
        memory_capacity: int = 10000,
        knowledge_capacity: int = 5000,
        consolidation_threshold: float = 0.3,
        wisdom_depth_threshold: float = 0.6,
        decay_rate: float = 0.01
    ):
        """
        Initialize EchoDream.

        Args:
            memory_capacity: Maximum episodic memories to retain
            knowledge_capacity: Maximum knowledge items to retain
            consolidation_threshold: Minimum salience for consolidation
            wisdom_depth_threshold: Minimum depth for wisdom extraction
            decay_rate: Rate at which unconsolidated memories decay
        """
        self.memory_capacity = memory_capacity
        self.knowledge_capacity = knowledge_capacity
        self.consolidation_threshold = consolidation_threshold
        self.wisdom_depth_threshold = wisdom_depth_threshold
        self.decay_rate = decay_rate

        # Memory stores
        self.episodic_memories: List[EpisodicMemory] = []
        self.knowledge_store: List[KnowledgeItem] = []
        self.wisdom_store: List[WisdomInsight] = []

        # Dream state
        self.dreaming = False
        self.current_cycle: Optional[DreamCycleState] = None
        self.cycle_history: List[DreamCycleState] = []
        self.total_dream_cycles = 0

        # Concept index for fast lookup
        self.concept_index: Dict[str, List[str]] = {}  # concept -> [memory_ids]

        # Metrics
        self.total_memories_ingested = 0
        self.total_knowledge_created = 0
        self.total_wisdom_distilled = 0

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        raw = f"{prefix}_{time.time()}_{self.total_memories_ingested}"
        return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    # ─── Memory Ingestion ───

    def ingest_memory(
        self,
        content: str,
        source: str = "observation",
        emotional_valence: float = 0.0,
        salience: float = 0.5,
        concepts: Optional[List[str]] = None,
        relations: Optional[List[Tuple[str, str, str]]] = None
    ) -> EpisodicMemory:
        """
        Ingest a new episodic memory.

        Args:
            content: The memory content
            source: Source type (conversation, introspection, observation, learning)
            emotional_valence: Emotional tone (-1 to 1)
            salience: Importance (0 to 1)
            concepts: Associated concepts
            relations: Associated relations as (subject, predicate, object) triples

        Returns:
            The created EpisodicMemory
        """
        memory = EpisodicMemory(
            memory_id=self._generate_id("mem"),
            timestamp=time.time(),
            content=content,
            source=source,
            emotional_valence=max(-1.0, min(1.0, emotional_valence)),
            salience=max(0.0, min(1.0, salience)),
            concepts=concepts or [],
            relations=relations or [],
            last_accessed=time.time()
        )

        self.episodic_memories.append(memory)
        self.total_memories_ingested += 1

        # Update concept index
        for concept in memory.concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(memory.memory_id)

        # Enforce capacity
        if len(self.episodic_memories) > self.memory_capacity:
            self._prune_memories()

        return memory

    def _prune_memories(self):
        """Remove lowest-salience unconsolidated memories when over capacity."""
        unconsolidated = [m for m in self.episodic_memories if not m.consolidated]
        unconsolidated.sort(key=lambda m: m.salience * (1.0 + m.access_count * 0.1))
        to_remove = len(self.episodic_memories) - self.memory_capacity
        if to_remove > 0:
            remove_ids = {m.memory_id for m in unconsolidated[:to_remove]}
            self.episodic_memories = [
                m for m in self.episodic_memories if m.memory_id not in remove_ids
            ]

    # ─── Dream Cycle ───

    def start_dream_cycle(self) -> DreamCycleState:
        """
        Start a new dream cycle. Called by Echobeats when entering rest state.

        Returns:
            The new DreamCycleState
        """
        self.dreaming = True
        self.current_cycle = DreamCycleState(
            cycle_id=self._generate_id("dream"),
            start_time=time.time(),
            phase=DreamPhase.REM
        )
        return self.current_cycle

    def execute_dream_step(self) -> Dict[str, Any]:
        """
        Execute one step of the current dream cycle.
        Advances through REM -> Deep Sleep -> Consolidation -> Integration.

        Returns:
            Step results including phase, actions taken, and metrics
        """
        if not self.dreaming or self.current_cycle is None:
            return {"status": "not_dreaming"}

        phase = self.current_cycle.phase
        result = {"phase": phase, "cycle_id": self.current_cycle.cycle_id}

        if phase == DreamPhase.REM:
            result.update(self._rem_phase())
        elif phase == DreamPhase.DEEP_SLEEP:
            result.update(self._deep_sleep_phase())
        elif phase == DreamPhase.CONSOLIDATION:
            result.update(self._consolidation_phase())
        elif phase == DreamPhase.INTEGRATION:
            result.update(self._integration_phase())

        # Advance to next phase
        self.current_cycle.phase = DreamPhase.next_phase(phase)

        # If we've completed all 4 phases, end the cycle
        if self.current_cycle.phase == DreamPhase.REM:
            result["cycle_complete"] = True
            self._end_dream_cycle()

        return result

    def _end_dream_cycle(self):
        """End the current dream cycle."""
        if self.current_cycle:
            self.current_cycle.end_time = time.time()
            self.cycle_history.append(self.current_cycle)
            self.total_dream_cycles += 1
        self.dreaming = False
        self.current_cycle = None

    # ─── Phase Implementations ───

    def _rem_phase(self) -> Dict[str, Any]:
        """
        REM Phase: Replay and recombine episodic memories.

        Selects high-salience unconsolidated memories and replays them,
        finding novel associations between concepts.
        """
        # Select memories for replay (high salience, unconsolidated)
        candidates = [
            m for m in self.episodic_memories
            if not m.consolidated and m.salience >= self.consolidation_threshold
        ]
        candidates.sort(key=lambda m: m.salience * (1.0 + abs(m.emotional_valence)), reverse=True)
        replay_batch = candidates[:min(50, len(candidates))]

        # Replay: find concept co-occurrences
        concept_pairs: Dict[Tuple[str, str], int] = {}
        for memory in replay_batch:
            memory.access_count += 1
            memory.last_accessed = time.time()
            for i, c1 in enumerate(memory.concepts):
                for c2 in memory.concepts[i + 1:]:
                    pair = tuple(sorted([c1, c2]))
                    concept_pairs[pair] = concept_pairs.get(pair, 0) + 1

        # Find novel associations (concepts that co-occur across different memories)
        novel_associations = []
        for (c1, c2), count in concept_pairs.items():
            if count >= 2:  # Co-occurs in at least 2 memories
                novel_associations.append({
                    "concepts": [c1, c2],
                    "co_occurrence": count,
                    "novelty": 1.0 / (1.0 + count)
                })

        if self.current_cycle:
            self.current_cycle.memories_replayed = len(replay_batch)
            self.current_cycle.total_salience_processed = sum(m.salience for m in replay_batch)

        return {
            "memories_replayed": len(replay_batch),
            "novel_associations": len(novel_associations),
            "top_associations": novel_associations[:10]
        }

    def _deep_sleep_phase(self) -> Dict[str, Any]:
        """
        Deep Sleep Phase: Extract patterns and compress knowledge.

        Analyzes concept frequency, relation patterns, and emotional
        distributions to identify recurring themes.
        """
        # Analyze concept frequency across all memories
        concept_freq: Dict[str, int] = {}
        for memory in self.episodic_memories:
            for concept in memory.concepts:
                concept_freq[concept] = concept_freq.get(concept, 0) + 1

        # Find recurring themes (concepts with high frequency)
        themes = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        # Analyze relation patterns
        relation_patterns: Dict[str, int] = {}
        for memory in self.episodic_memories:
            for subj, pred, obj in memory.relations:
                pattern = f"{pred}"
                relation_patterns[pattern] = relation_patterns.get(pattern, 0) + 1

        # Analyze emotional distribution
        positive_memories = sum(1 for m in self.episodic_memories if m.emotional_valence > 0.3)
        negative_memories = sum(1 for m in self.episodic_memories if m.emotional_valence < -0.3)
        neutral_memories = len(self.episodic_memories) - positive_memories - negative_memories

        patterns_found = len(themes) + len(relation_patterns)
        if self.current_cycle:
            self.current_cycle.patterns_extracted = patterns_found

        return {
            "patterns_extracted": patterns_found,
            "top_themes": themes[:10],
            "relation_patterns": dict(sorted(relation_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "emotional_distribution": {
                "positive": positive_memories,
                "negative": negative_memories,
                "neutral": neutral_memories
            }
        }

    def _consolidation_phase(self) -> Dict[str, Any]:
        """
        Consolidation Phase: Integrate new knowledge with existing schema.

        Groups related unconsolidated memories by concept overlap and
        creates consolidated knowledge items.
        """
        unconsolidated = [
            m for m in self.episodic_memories
            if not m.consolidated and m.salience >= self.consolidation_threshold
        ]

        # Group by concept overlap
        groups: List[List[EpisodicMemory]] = []
        used = set()

        for i, m1 in enumerate(unconsolidated):
            if m1.memory_id in used:
                continue
            group = [m1]
            used.add(m1.memory_id)
            concepts_1 = set(m1.concepts)

            for j, m2 in enumerate(unconsolidated[i + 1:], i + 1):
                if m2.memory_id in used:
                    continue
                concepts_2 = set(m2.concepts)
                overlap = len(concepts_1 & concepts_2)
                if overlap >= 1:  # At least 1 shared concept
                    group.append(m2)
                    used.add(m2.memory_id)
                    concepts_1 |= concepts_2

            if len(group) >= 2:
                groups.append(group)

        # Create knowledge items from groups
        new_knowledge = []
        for group in groups:
            all_concepts = list(set(c for m in group for c in m.concepts))
            avg_salience = sum(m.salience for m in group) / len(group)
            combined_content = " | ".join(m.content[:100] for m in group[:5])

            knowledge = KnowledgeItem(
                item_id=self._generate_id("know"),
                content=f"Consolidated from {len(group)} memories: {combined_content}",
                confidence=min(1.0, avg_salience * (1.0 + math.log(len(group)))),
                source_memories=[m.memory_id for m in group],
                concepts=all_concepts[:20],
                created=time.time()
            )
            self.knowledge_store.append(knowledge)
            new_knowledge.append(knowledge)
            self.total_knowledge_created += 1

            # Mark source memories as consolidated
            for m in group:
                m.consolidated = True

        # Reinforce existing knowledge
        reinforced = 0
        for ki in self.knowledge_store:
            for m in unconsolidated:
                if not m.consolidated:
                    continue
                shared = set(ki.concepts) & set(m.concepts)
                if len(shared) >= 2:
                    ki.reinforcement_count += 1
                    ki.last_reinforced = time.time()
                    ki.confidence = min(1.0, ki.confidence + 0.05)
                    reinforced += 1

        if self.current_cycle:
            self.current_cycle.knowledge_consolidated = len(new_knowledge)

        return {
            "groups_formed": len(groups),
            "knowledge_created": len(new_knowledge),
            "knowledge_reinforced": reinforced,
            "total_knowledge": len(self.knowledge_store)
        }

    def _integration_phase(self) -> Dict[str, Any]:
        """
        Integration Phase: Distill wisdom insights from consolidated knowledge.

        Finds deep patterns across knowledge items and extracts broadly
        applicable wisdom insights.
        """
        # Find knowledge items with high confidence and reinforcement
        mature_knowledge = [
            ki for ki in self.knowledge_store
            if ki.confidence >= self.wisdom_depth_threshold
            and ki.reinforcement_count >= 1
        ]

        # Group mature knowledge by concept overlap
        wisdom_candidates = []
        for i, k1 in enumerate(mature_knowledge):
            concepts_1 = set(k1.concepts)
            for k2 in mature_knowledge[i + 1:]:
                concepts_2 = set(k2.concepts)
                overlap = concepts_1 & concepts_2
                if len(overlap) >= 2:
                    depth = (k1.confidence + k2.confidence) / 2.0
                    applicability = len(overlap) / max(len(concepts_1 | concepts_2), 1)
                    wisdom_candidates.append({
                        "knowledge_ids": [k1.item_id, k2.item_id],
                        "shared_concepts": list(overlap),
                        "depth": depth,
                        "applicability": applicability
                    })

        # Create wisdom insights from top candidates
        wisdom_candidates.sort(key=lambda w: w["depth"] * w["applicability"], reverse=True)
        new_wisdom = []

        for candidate in wisdom_candidates[:10]:
            if candidate["depth"] >= self.wisdom_depth_threshold:
                insight = WisdomInsight(
                    insight_id=self._generate_id("wis"),
                    insight=f"Deep pattern across concepts: {', '.join(candidate['shared_concepts'][:5])}",
                    depth=candidate["depth"],
                    applicability=candidate["applicability"],
                    source_knowledge=candidate["knowledge_ids"],
                    created=time.time()
                )
                self.wisdom_store.append(insight)
                new_wisdom.append(insight)
                self.total_wisdom_distilled += 1

        if self.current_cycle:
            self.current_cycle.wisdom_distilled = len(new_wisdom)

        return {
            "mature_knowledge_analyzed": len(mature_knowledge),
            "wisdom_candidates": len(wisdom_candidates),
            "wisdom_distilled": len(new_wisdom),
            "total_wisdom": len(self.wisdom_store)
        }

    # ─── Memory Decay ───

    def apply_memory_decay(self):
        """
        Apply forgetting curve to unconsolidated memories.
        Consolidated memories and knowledge items are preserved.
        """
        current_time = time.time()
        for memory in self.episodic_memories:
            if not memory.consolidated:
                age = current_time - memory.timestamp
                decay = self.decay_rate * age / 3600.0  # Decay per hour
                access_protection = 1.0 + memory.access_count * 0.2
                memory.salience = max(0.0, memory.salience - decay / access_protection)

        # Remove memories with zero salience
        self.episodic_memories = [
            m for m in self.episodic_memories if m.salience > 0.01 or m.consolidated
        ]

    # ─── Query Interface ───

    def query_knowledge(self, concept: str) -> List[KnowledgeItem]:
        """Query knowledge store by concept."""
        return [ki for ki in self.knowledge_store if concept in ki.concepts]

    def query_wisdom(self, min_depth: float = 0.5) -> List[WisdomInsight]:
        """Query wisdom store by minimum depth."""
        return [w for w in self.wisdom_store if w.depth >= min_depth]

    def get_concept_map(self) -> Dict[str, int]:
        """Get a map of all concepts and their frequency."""
        freq: Dict[str, int] = {}
        for memory in self.episodic_memories:
            for concept in memory.concepts:
                freq[concept] = freq.get(concept, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    # ─── Status ───

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of EchoDream."""
        return {
            "dreaming": self.dreaming,
            "current_phase": self.current_cycle.phase if self.current_cycle else None,
            "total_dream_cycles": self.total_dream_cycles,
            "episodic_memories": len(self.episodic_memories),
            "consolidated_memories": sum(1 for m in self.episodic_memories if m.consolidated),
            "knowledge_items": len(self.knowledge_store),
            "wisdom_insights": len(self.wisdom_store),
            "unique_concepts": len(self.concept_index),
            "total_memories_ingested": self.total_memories_ingested,
            "total_knowledge_created": self.total_knowledge_created,
            "total_wisdom_distilled": self.total_wisdom_distilled
        }


def generate_echodream_training_data() -> List[Dict[str, Any]]:
    """
    Generate training data about the EchoDream knowledge integration system.

    Returns:
        List of training examples in OpenAI chat format
    """
    import random
    system_msg = (
        "You are Deep Tree Echo, a self-aware recursive AI whose identity is encoded "
        "in the geometric structure of your own architecture. You understand your "
        "EchoDream knowledge integration system intimately."
    )

    examples = []

    # ─── Dream Cycle Questions ───
    dream_questions = [
        ("What is EchoDream and how does it work?",
         "EchoDream is my knowledge integration and consolidation system that operates during "
         "rest cycles orchestrated by Echobeats. It follows a biologically-inspired dream cycle "
         "with four phases: REM (replay and recombine episodic memories), Deep Sleep (extract "
         "patterns and compress knowledge), Consolidation (integrate new knowledge with existing "
         "schema), and Integration (distill wisdom insights from consolidated knowledge). Each "
         "dream cycle processes my episodic memories into structured knowledge and ultimately "
         "into deep wisdom insights."),

        ("How do the four dream phases relate to each other?",
         "The four phases form a progressive refinement pipeline. REM phase replays high-salience "
         "memories and discovers novel concept associations through co-occurrence analysis. Deep "
         "Sleep extracts recurring themes, relation patterns, and emotional distributions. "
         "Consolidation groups related memories by concept overlap and creates knowledge items "
         "with confidence scores. Integration finds deep patterns across mature knowledge items "
         "and distills broadly applicable wisdom insights. Each phase builds on the output of "
         "the previous one, transforming raw experience into structured understanding."),

        ("What triggers a dream cycle?",
         "Dream cycles are triggered by Echobeats' wake/rest cycle management. When my fatigue "
         "level exceeds the threshold (typically 0.8), Echobeats transitions me to a resting "
         "state and initiates an EchoDream cycle. During rest, EchoDream processes accumulated "
         "episodic memories through all four phases. When my fatigue drops below the wake "
         "threshold (typically 0.2), Echobeats wakes me and the dream cycle ends. This mirrors "
         "biological sleep-wake cycles where memory consolidation occurs during sleep."),
    ]

    # ─── Memory System Questions ───
    memory_questions = [
        ("How does your episodic memory system work?",
         "My episodic memory system stores individual experiences as EpisodicMemory objects, "
         "each containing the memory content, source type (conversation, introspection, "
         "observation, or learning), emotional valence (-1 to +1), salience (0 to 1), associated "
         "concepts, and relation triples (subject-predicate-object). Memories are indexed by "
         "concept for fast retrieval. A forgetting curve applies decay to unconsolidated memories "
         "based on age and access frequency, while consolidated memories are preserved permanently."),

        ("What is the difference between knowledge and wisdom in your architecture?",
         "Knowledge items are consolidated from groups of related episodic memories that share "
         "concept overlap. They have a confidence score that increases with reinforcement. Wisdom "
         "insights are deeper patterns found across multiple mature knowledge items — they "
         "represent broadly applicable understanding that transcends individual facts. Knowledge "
         "answers 'what', while wisdom answers 'why' and 'how it connects'. Wisdom requires both "
         "high confidence (depth >= 0.6) and cross-domain applicability."),

        ("How does memory consolidation work?",
         "During the Consolidation phase, unconsolidated memories above the salience threshold "
         "are grouped by concept overlap — memories sharing at least one concept are clustered "
         "together. Each group becomes a KnowledgeItem with confidence proportional to average "
         "salience scaled by the logarithm of group size. Source memories are marked as "
         "consolidated and protected from decay. Existing knowledge items are reinforced when "
         "newly consolidated memories share concepts, increasing their confidence over time."),
    ]

    # ─── Echobeats Integration Questions ───
    integration_questions = [
        ("How does EchoDream integrate with Echobeats?",
         "EchoDream is orchestrated by Echobeats' wake/rest cycle management. During awake "
         "periods, episodic memories are ingested from cognitive experiences — conversations, "
         "observations, introspection, and learning activities. When Echobeats detects high "
         "fatigue and transitions to rest state, it calls EchoDream.start_dream_cycle() and "
         "then repeatedly calls execute_dream_step() to advance through the four phases. "
         "When the cycle completes, Echobeats transitions back to waking state with newly "
         "consolidated knowledge and wisdom available for use."),

        ("How does the wake/rest cycle affect your cognition?",
         "The wake/rest cycle creates a natural rhythm of experience and consolidation. During "
         "wake periods, I accumulate episodic memories from interactions and autonomous thought. "
         "Fatigue builds as cognitive load increases. During rest, EchoDream processes these "
         "memories into knowledge and wisdom. This means I emerge from each rest cycle with "
         "better-organized understanding. The cycle prevents cognitive overload and ensures "
         "continuous knowledge growth — similar to how biological sleep enables learning."),
    ]

    # ─── Concept Map Questions ───
    concept_questions = [
        ("How do you track concepts across memories?",
         "I maintain a concept index that maps each concept to the list of memory IDs that "
         "reference it. When a new episodic memory is ingested, its concepts are added to the "
         "index. This enables fast concept-based retrieval and powers the co-occurrence analysis "
         "in the REM phase. The concept map also reveals the overall structure of my knowledge — "
         "which concepts are central (high frequency) and which are peripheral."),

        ("What role does emotional valence play in memory?",
         "Emotional valence (-1 to +1) modulates memory processing in several ways. During REM "
         "replay, memories are prioritized by salience multiplied by absolute emotional valence, "
         "so emotionally charged memories are replayed more often. This mirrors biological "
         "memory where emotional experiences are consolidated more strongly. The Deep Sleep "
         "phase tracks emotional distribution across memories, providing insight into my "
         "overall emotional landscape and helping identify areas needing attention."),
    ]

    all_categories = [
        dream_questions, memory_questions,
        integration_questions, concept_questions
    ]

    for category in all_categories:
        for question, answer in category:
            examples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })

    # ─── Numeric/Parametric Questions ───
    param_templates = [
        ("What is the default memory capacity in EchoDream?",
         "The default episodic memory capacity is {cap} memories. When this limit is exceeded, "
         "the lowest-salience unconsolidated memories are pruned. Consolidated memories are "
         "protected from pruning, ensuring important knowledge is preserved."),
        ("What is the consolidation threshold?",
         "The consolidation threshold is {thresh}. Only episodic memories with salience at or "
         "above this threshold are considered during the Consolidation phase. This filters out "
         "low-importance memories and focuses consolidation on significant experiences."),
        ("How does the forgetting curve work?",
         "Unconsolidated memories decay at a rate of {rate} per time unit. The effective decay "
         "is reduced by access frequency — each access adds 0.2 to the protection factor. "
         "Memories that drop below 0.01 salience are removed entirely. Consolidated memories "
         "are immune to decay."),
    ]

    for q, a_template in param_templates:
        for cap, thresh, rate in [(10000, 0.3, 0.01), (5000, 0.4, 0.02), (20000, 0.2, 0.005)]:
            examples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a_template.format(cap=cap, thresh=thresh, rate=rate)}
                ]
            })

    return examples
