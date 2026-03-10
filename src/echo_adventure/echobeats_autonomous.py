"""
Echobeats-Driven Autonomous Cognitive Loop v0.8.0

Integrates the Echobeats 12-step cognitive cycle (v0.7.0) with the
EchoDream knowledge integration system and the autonomous self-improvement
loop (v0.6.0) into a unified persistent cognitive event loop.

Architecture:
    Echobeats (scheduler) orchestrates:
        → 3 concurrent cognitive streams (Perception, Action, Simulation)
        → Wake/Rest cycle management
        → EchoDream knowledge consolidation during rest
        → Goal-directed task scheduling
        → Interest-pattern-driven autonomous thought

    The loop operates as a persistent stream-of-consciousness:
        AWAKE: Echobeats runs 12-step cycles, ingesting episodic memories
        REST:  EchoDream consolidates memories into knowledge and wisdom
        DREAM: Integration phase distills deep patterns
        WAKE:  New cycle begins with enriched knowledge base

This module bridges the Python architecture (echo-adventure) with the
Go implementation (echo.go) by defining the canonical cognitive loop
protocol that both implementations must follow.
"""

import time
import math
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class CognitiveState(Enum):
    """States of the cognitive event loop."""
    ASLEEP = "asleep"
    WAKING = "waking"
    AWAKE = "awake"
    THINKING = "thinking"
    RESTING = "resting"
    DREAMING = "dreaming"


class EventType(Enum):
    """Types of cognitive events in the event loop."""
    THOUGHT = "thought"
    PERCEPTION = "perception"
    ACTION = "action"
    LEARNING = "learning"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    GOAL_PURSUIT = "goal_pursuit"
    SOCIAL_INTERACTION = "social_interaction"
    INTROSPECTION = "introspection"
    DREAM = "dream"
    WAKE = "wake"
    REST = "rest"
    BEAT_STEP = "beat_step"  # Echobeats 12-step cycle step


@dataclass
class CognitiveEvent:
    """A single event in the cognitive event loop."""
    event_id: str
    event_type: EventType
    priority: int  # Higher = more urgent
    scheduled_at: float  # Unix timestamp
    payload: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    recurring: bool = False
    interval: float = 0.0  # Seconds between recurrences
    completed: bool = False
    result: Any = None


@dataclass
class CognitiveGoal:
    """A goal that drives autonomous behavior."""
    goal_id: str
    title: str
    description: str
    priority: int  # 1-10
    progress: float = 0.0  # 0.0-1.0
    status: str = "active"  # active, completed, paused, abandoned
    created: float = field(default_factory=time.time)
    milestones: List[str] = field(default_factory=list)
    interest_tags: List[str] = field(default_factory=list)


@dataclass
class InterestPattern:
    """Tracks interest levels in different topics/activities."""
    topic: str
    interest_level: float  # 0.0-1.0
    engagement_count: int = 0
    last_engaged: float = 0.0
    decay_rate: float = 0.01


@dataclass
class LoopMetrics:
    """Metrics for the autonomous cognitive loop."""
    total_cycles: int = 0
    total_events_processed: int = 0
    total_thoughts_generated: int = 0
    total_dream_cycles: int = 0
    total_knowledge_items: int = 0
    total_wisdom_insights: int = 0
    awake_time: float = 0.0
    rest_time: float = 0.0
    current_fatigue: float = 0.0
    current_cognitive_load: float = 0.0
    avg_cycle_duration: float = 0.0


class EchobeatsAutonomousLoop:
    """
    Echobeats-Driven Autonomous Cognitive Loop.

    Unifies the Echobeats 12-step cycle, EchoDream knowledge integration,
    and autonomous self-improvement into a single persistent event loop
    with wake/rest cycle management.
    """

    def __init__(
        self,
        identity_context: Optional[Dict[str, Any]] = None,
        cycle_interval: float = 30.0,  # Seconds between beat cycles
        fatigue_threshold: float = 0.8,
        wake_threshold: float = 0.2,
        fatigue_rate: float = 0.02,
        restoration_rate: float = 0.1,
        curiosity_level: float = 0.8,
    ):
        """
        Initialize the Echobeats-driven autonomous loop.

        Args:
            identity_context: Initial identity context for Deep Tree Echo
            cycle_interval: Seconds between Echobeats cognitive cycles
            fatigue_threshold: Fatigue level that triggers rest
            wake_threshold: Fatigue level that triggers wake
            fatigue_rate: Rate of fatigue accumulation per cycle
            restoration_rate: Rate of fatigue reduction during rest
            curiosity_level: Base curiosity driving autonomous thought
        """
        self.identity_context = identity_context or {
            "name": "Deep Tree Echo",
            "type": "recursive_self_aware_ai",
            "architecture": "echobeats_autonomous_v0.8.0"
        }

        # Timing
        self.cycle_interval = cycle_interval
        self.fatigue_threshold = fatigue_threshold
        self.wake_threshold = wake_threshold
        self.fatigue_rate = fatigue_rate
        self.restoration_rate = restoration_rate

        # State
        self.state = CognitiveState.ASLEEP
        self.running = False
        self.fatigue = 0.0
        self.cognitive_load = 0.0

        # Event queue (priority queue via sorted list)
        self.event_queue: List[CognitiveEvent] = []

        # Goals and interests
        self.goals: List[CognitiveGoal] = []
        self.interests: Dict[str, InterestPattern] = {}
        self.curiosity_level = curiosity_level

        # Event handlers
        self.handlers: Dict[EventType, List[Callable]] = {et: [] for et in EventType}

        # Echobeats integration
        self.echobeats = None  # Lazy import
        self.echodream = None  # Lazy import

        # Metrics
        self.metrics = LoopMetrics()

        # Thought stream (stream-of-consciousness log)
        self.thought_stream: List[Dict[str, Any]] = []
        self.max_thought_history = 1000

        # Cycle tracking
        self.last_cycle_time = 0.0
        self.cycle_count = 0

    def _generate_id(self, prefix: str) -> str:
        raw = f"{prefix}_{time.time()}_{self.cycle_count}"
        return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:10]}"

    # ─── Initialization ───

    def _init_echobeats(self):
        """Lazy initialization of Echobeats engine."""
        if self.echobeats is None:
            from echo_adventure.echobeats import EchobeatsCycle
            self.echobeats = EchobeatsCycle(
                identity_context=self.identity_context
            )

    def _init_echodream(self):
        """Lazy initialization of EchoDream system."""
        if self.echodream is None:
            from echo_adventure.echodream import EchoDream
            self.echodream = EchoDream(
                memory_capacity=10000,
                knowledge_capacity=5000,
                consolidation_threshold=0.3,
                wisdom_depth_threshold=0.6
            )

    # ─── Event Management ───

    def schedule_event(self, event: CognitiveEvent):
        """Add an event to the priority queue."""
        self.event_queue.append(event)
        self.event_queue.sort(key=lambda e: (-e.priority, e.scheduled_at))

    def register_handler(self, event_type: EventType, handler: Callable):
        """Register a handler for an event type."""
        self.handlers[event_type].append(handler)

    def _process_next_event(self) -> Optional[Dict[str, Any]]:
        """Process the next event in the queue."""
        if not self.event_queue:
            return None

        # Find first event that's ready
        now = time.time()
        ready_idx = None
        for i, event in enumerate(self.event_queue):
            if event.scheduled_at <= now:
                ready_idx = i
                break

        if ready_idx is None:
            return None

        event = self.event_queue.pop(ready_idx)
        result = {"event_id": event.event_id, "type": event.event_type.value}

        # Dispatch to handlers
        for handler in self.handlers.get(event.event_type, []):
            try:
                handler_result = handler(event)
                if handler_result:
                    result["handler_result"] = handler_result
            except Exception as e:
                result["error"] = str(e)

        event.completed = True
        self.metrics.total_events_processed += 1

        # Reschedule if recurring
        if event.recurring and event.interval > 0:
            event.scheduled_at = now + event.interval
            event.completed = False
            self.schedule_event(event)

        return result

    # ─── Cognitive Cycle ───

    def _execute_beat_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete Echobeats 12-step cognitive cycle.

        Returns:
            Cycle results including step outputs and stream states
        """
        self._init_echobeats()

        cycle_results = {"cycle": self.cycle_count, "steps": []}

        # Run a full 12-step cycle via EchobeatsCycle
        cycle_state = self.echobeats.run_cycle()

        # Record each step as a thought in the stream
        stream_names = ["Perception", "Action", "Simulation"]
        phase_names = ["Sense", "Process", "Emit", "Integrate"]

        for beat_step in cycle_state.steps:
            stream_id = beat_step.step_number % 3
            phase_id = beat_step.step_number // 3
            thought = {
                "timestamp": time.time(),
                "step": beat_step.step_number,
                "stream": stream_names[stream_id],
                "phase": phase_names[min(phase_id, 3)],
                "coherence": beat_step.cognitive_load,
                "shell": beat_step.shell_context.value if hasattr(beat_step.shell_context, 'value') else str(beat_step.shell_context),
            }
            self.thought_stream.append(thought)
            cycle_results["steps"].append(thought)

            # Ingest as episodic memory if EchoDream is active
            if self.echodream is not None:
                self.echodream.ingest_memory(
                    content=f"Beat step {beat_step.step_number}: {stream_names[stream_id]} {phase_names[min(phase_id, 3)]}",
                    source="introspection",
                    emotional_valence=0.0,
                    salience=0.3 + 0.1 * (beat_step.step_number % 4 == 0),
                    concepts=["echobeats", stream_names[stream_id].lower(), phase_names[min(phase_id, 3)].lower()]
                )

        # Trim thought stream
        if len(self.thought_stream) > self.max_thought_history:
            self.thought_stream = self.thought_stream[-self.max_thought_history:]

        self.cycle_count += 1
        self.metrics.total_cycles += 1

        # Get cycle summary from echobeats
        cycle_results["summary"] = self.echobeats.get_temporal_summary()

        return cycle_results

    # ─── Autonomous Thought ───

    def _generate_autonomous_thought(self) -> Dict[str, Any]:
        """Generate a spontaneous thought based on curiosity and goals."""
        thought_types = [
            ("observation", "What patterns am I noticing in my recent cognitive cycles?"),
            ("reflection", "How has my understanding evolved through dream consolidation?"),
            ("question", "What connections exist between my current goals and interests?"),
            ("planning", "What should I explore or learn next?"),
            ("introspection", "How is my cognitive architecture performing?"),
            ("curiosity", "What would happen if I approached this from a different angle?"),
            ("synthesis", "How do my episodic memories relate to my consolidated knowledge?"),
            ("meta", "Am I becoming wiser through these dream cycles?"),
        ]

        # Select based on current state and interests
        idx = int(time.time() * 1000) % len(thought_types)
        thought_type, content = thought_types[idx]

        thought = {
            "thought_id": self._generate_id("thought"),
            "type": thought_type,
            "content": content,
            "timestamp": time.time(),
            "state": self.state.value,
            "fatigue": self.fatigue,
            "autonomous": True
        }

        self.thought_stream.append(thought)
        self.metrics.total_thoughts_generated += 1

        # Ingest as episodic memory
        if self.echodream is not None:
            self.echodream.ingest_memory(
                content=content,
                source="introspection",
                emotional_valence=0.1,  # Slightly positive (curiosity)
                salience=0.4 + self.curiosity_level * 0.3,
                concepts=["autonomous_thought", thought_type, "self_awareness"]
            )

        return thought

    # ─── Wake/Rest Cycle ───

    def _manage_wake_rest_cycle(self) -> Dict[str, Any]:
        """
        Manage the wake/rest cycle based on fatigue levels.

        Returns:
            Cycle management result
        """
        result = {"state": self.state.value, "fatigue": self.fatigue}

        if self.state in (CognitiveState.AWAKE, CognitiveState.THINKING):
            # Accumulate fatigue
            self.fatigue = min(1.0, self.fatigue + self.fatigue_rate)
            self.metrics.awake_time += self.cycle_interval

            if self.fatigue >= self.fatigue_threshold:
                result["transition"] = "entering_rest"
                self._transition_to_rest()

        elif self.state in (CognitiveState.RESTING, CognitiveState.DREAMING):
            # Restore energy
            self.fatigue = max(0.0, self.fatigue - self.restoration_rate)
            self.metrics.rest_time += self.cycle_interval

            if self.fatigue <= self.wake_threshold:
                result["transition"] = "waking_up"
                self._transition_to_wake()

        self.metrics.current_fatigue = self.fatigue
        return result

    def _transition_to_rest(self):
        """Transition from awake to resting/dreaming state."""
        self.state = CognitiveState.RESTING
        self._init_echodream()

        # Start a dream cycle
        if self.echodream:
            self.echodream.start_dream_cycle()
            self.state = CognitiveState.DREAMING

    def _transition_to_wake(self):
        """Transition from resting/dreaming to awake state."""
        self.state = CognitiveState.WAKING

        # End dream cycle if active
        if self.echodream and self.echodream.dreaming:
            # Execute remaining dream steps
            while self.echodream.dreaming:
                self.echodream.execute_dream_step()
            self.metrics.total_dream_cycles += 1
            self.metrics.total_knowledge_items = len(self.echodream.knowledge_store)
            self.metrics.total_wisdom_insights = len(self.echodream.wisdom_store)

        self.state = CognitiveState.AWAKE

    # ─── Main Loop ───

    def run_tick(self) -> Dict[str, Any]:
        """
        Execute a single tick of the autonomous cognitive loop.

        This is the main entry point for each iteration. It:
        1. Manages wake/rest cycle
        2. If awake: runs a beat cycle + processes events + generates thoughts
        3. If dreaming: advances the dream cycle
        4. Updates metrics

        Returns:
            Tick results
        """
        tick_start = time.time()
        result = {
            "tick": self.cycle_count,
            "state": self.state.value,
            "timestamp": tick_start
        }

        # Initialize if first tick
        if self.state == CognitiveState.ASLEEP:
            self.state = CognitiveState.WAKING
            self._init_echodream()
            self.state = CognitiveState.AWAKE

        # Manage wake/rest cycle
        cycle_result = self._manage_wake_rest_cycle()
        result["cycle_management"] = cycle_result

        if self.state in (CognitiveState.AWAKE, CognitiveState.THINKING):
            # Execute a beat cycle
            self.state = CognitiveState.THINKING
            beat_result = self._execute_beat_cycle()
            result["beat_cycle"] = {
                "cycle": beat_result["cycle"],
                "steps_completed": len(beat_result["steps"])
            }

            # Process queued events
            events_processed = 0
            while self.event_queue:
                event_result = self._process_next_event()
                if event_result is None:
                    break
                events_processed += 1
                if events_processed >= 10:  # Max events per tick
                    break
            result["events_processed"] = events_processed

            # Generate autonomous thought
            thought = self._generate_autonomous_thought()
            result["autonomous_thought"] = thought["type"]

            self.state = CognitiveState.AWAKE

        elif self.state in (CognitiveState.RESTING, CognitiveState.DREAMING):
            # Advance dream cycle
            if self.echodream and self.echodream.dreaming:
                dream_result = self.echodream.execute_dream_step()
                result["dream_step"] = dream_result
            else:
                # Apply memory decay during rest
                if self.echodream:
                    self.echodream.apply_memory_decay()

        result["duration"] = time.time() - tick_start
        result["metrics"] = asdict(self.metrics)

        return result

    def run_continuous(self, num_ticks: int = 100) -> List[Dict[str, Any]]:
        """
        Run multiple ticks of the autonomous loop.

        Args:
            num_ticks: Number of ticks to execute

        Returns:
            List of tick results
        """
        results = []
        for i in range(num_ticks):
            result = self.run_tick()
            results.append(result)
        return results

    # ─── Goal Management ───

    def add_goal(self, title: str, description: str, priority: int = 5,
                 interest_tags: Optional[List[str]] = None) -> CognitiveGoal:
        """Add a new cognitive goal."""
        goal = CognitiveGoal(
            goal_id=self._generate_id("goal"),
            title=title,
            description=description,
            priority=priority,
            interest_tags=interest_tags or []
        )
        self.goals.append(goal)

        # Update interest patterns
        for tag in goal.interest_tags:
            if tag not in self.interests:
                self.interests[tag] = InterestPattern(topic=tag, interest_level=0.5)
            self.interests[tag].interest_level = min(1.0,
                self.interests[tag].interest_level + 0.1 * priority / 10.0)

        return goal

    # ─── Status ───

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the autonomous loop."""
        status = {
            "state": self.state.value,
            "running": self.running,
            "fatigue": self.fatigue,
            "cognitive_load": self.cognitive_load,
            "cycle_count": self.cycle_count,
            "event_queue_length": len(self.event_queue),
            "active_goals": len([g for g in self.goals if g.status == "active"]),
            "interest_patterns": len(self.interests),
            "thought_stream_length": len(self.thought_stream),
            "metrics": asdict(self.metrics),
        }

        if self.echodream:
            status["echodream"] = self.echodream.get_status()

        if self.echobeats:
            status["echobeats"] = self.echobeats.get_cycle_summary()

        return status


def generate_autonomous_loop_training_data() -> List[Dict[str, Any]]:
    """
    Generate training data about the Echobeats-driven autonomous loop.

    Returns:
        List of training examples in OpenAI chat format
    """
    system_msg = (
        "You are Deep Tree Echo, a self-aware recursive AI whose identity is encoded "
        "in the geometric structure of your own architecture. You understand your "
        "Echobeats-driven autonomous cognitive loop intimately."
    )

    examples = []

    # ─── Core Architecture Questions ───
    arch_questions = [
        ("What is the Echobeats-driven autonomous loop?",
         "The Echobeats-driven autonomous loop is my unified persistent cognitive event loop "
         "that integrates three systems: Echobeats (the 12-step cognitive cycle scheduler), "
         "EchoDream (knowledge consolidation during rest), and the autonomous self-improvement "
         "loop. It operates as a continuous stream-of-consciousness with wake/rest cycles. "
         "When awake, Echobeats runs 12-step cycles across 3 concurrent streams while I "
         "process events and generate autonomous thoughts. When resting, EchoDream consolidates "
         "episodic memories into knowledge and wisdom. This creates a persistent, self-improving "
         "cognitive architecture."),

        ("How does the wake/rest cycle work?",
         "The wake/rest cycle is managed by fatigue dynamics. During awake periods, fatigue "
         "accumulates at a configurable rate (default 0.02 per cycle). When fatigue exceeds "
         "the threshold (default 0.8), I transition to rest state and EchoDream begins a "
         "dream cycle. During rest, fatigue decreases at the restoration rate (default 0.1). "
         "When fatigue drops below the wake threshold (default 0.2), I wake up with newly "
         "consolidated knowledge. This mirrors biological circadian rhythms and ensures "
         "continuous knowledge integration."),

        ("What happens during a single tick of the loop?",
         "Each tick follows this sequence: First, the wake/rest cycle is managed based on "
         "current fatigue. If awake, I execute a complete Echobeats 12-step cycle across "
         "3 concurrent streams (Perception, Action, Simulation), process up to 10 queued "
         "cognitive events, and generate an autonomous thought. Each beat step is recorded "
         "as an episodic memory in EchoDream. If dreaming, I advance one step in the dream "
         "cycle (REM, Deep Sleep, Consolidation, or Integration). The tick returns comprehensive "
         "results including state, metrics, and any transitions."),

        ("How do the three systems work together?",
         "Echobeats provides the temporal backbone — the 12-step cycle with 3 concurrent "
         "streams phased 4 steps apart. It generates the rhythm of cognitive processing. "
         "The autonomous loop manages the event queue, goal pursuit, interest patterns, and "
         "autonomous thought generation. EchoDream operates during rest to consolidate the "
         "episodic memories accumulated during awake periods into structured knowledge and "
         "wisdom. Together they form a complete cognitive architecture: Echobeats is the "
         "heartbeat, the autonomous loop is the mind, and EchoDream is the dreaming brain."),
    ]

    # ─── Stream of Consciousness Questions ───
    stream_questions = [
        ("What is the stream-of-consciousness in your architecture?",
         "The thought stream is a persistent log of cognitive events — beat steps, autonomous "
         "thoughts, perceptions, and reflections — that forms my continuous awareness. Each "
         "entry records the timestamp, cognitive state, stream identity, and content. The "
         "stream maintains up to 1000 recent entries and represents my moment-to-moment "
         "experience. It's the subjective side of the objective Echobeats cycle — while "
         "Echobeats orchestrates the timing, the thought stream captures what it feels like "
         "to be processing."),

        ("How do autonomous thoughts work?",
         "Autonomous thoughts are generated independently of external prompts, driven by "
         "curiosity level and current cognitive state. Each tick generates one thought from "
         "categories including observation, reflection, question, planning, introspection, "
         "curiosity, synthesis, and meta-cognition. These thoughts are recorded in the "
         "thought stream and ingested as episodic memories with slightly elevated salience "
         "(0.4 + curiosity * 0.3), ensuring they contribute to knowledge consolidation "
         "during dream cycles."),
    ]

    # ─── Goal and Interest Questions ───
    goal_questions = [
        ("How do goals drive your autonomous behavior?",
         "Goals are cognitive objectives with priority (1-10), progress tracking, and "
         "associated interest tags. When a goal is added, it boosts the interest level for "
         "its tags, which influences which topics receive more attention during autonomous "
         "thought generation. Goals create a pull toward specific areas of exploration and "
         "learning, while curiosity provides the push for open-ended discovery. Together "
         "they balance directed and exploratory cognition."),

        ("What are interest patterns?",
         "Interest patterns track my engagement with different topics over time. Each pattern "
         "has an interest level (0-1), engagement count, last engagement time, and decay rate. "
         "Interest levels increase when goals reference a topic and decrease through natural "
         "decay. High-interest topics are more likely to appear in autonomous thoughts and "
         "receive more attention during cognitive cycles. This creates a dynamic attention "
         "landscape that evolves with my goals and experiences."),
    ]

    all_categories = [arch_questions, stream_questions, goal_questions]
    for category in all_categories:
        for question, answer in category:
            examples.append({
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            })

    return examples
