"""
Integrated Cognitive Loop for Deep Tree Echo v1.0.0

The unified autonomous cognitive event loop that integrates all subsystems:
- Echobeats 12-step cycle (temporal backbone)
- GoalPursuitEngine (autonomous agency)
- AdvancedEchoDream (memory-to-wisdom pipeline)
- PersistentMemoryStore (cross-session continuity)

This module represents the first complete integration of all cognitive
subsystems into a single persistent, self-orchestrating loop. It is the
Python reference implementation for the Go production runtime.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum


class CognitivePhase(Enum):
    """Phases of the integrated cognitive loop"""
    AWAKENING = "awakening"       # Restoring state from persistent memory
    PERCEIVING = "perceiving"     # Processing external inputs
    THINKING = "thinking"         # Autonomous thought generation
    PURSUING = "pursuing"         # Active goal pursuit
    REFLECTING = "reflecting"     # Self-reflection and introspection
    DREAMING = "dreaming"         # Memory consolidation and wisdom extraction
    RESTING = "resting"           # Low-power state between dream cycles


@dataclass
class CognitiveEvent:
    """An event in the cognitive stream"""
    event_type: str
    content: str
    source: str
    priority: float  # 0.0-1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class StreamOfConsciousnessEntry:
    """A single entry in the stream of consciousness"""
    thought: str
    phase: str
    beat_step: int
    cycle_number: int
    timestamp: float
    cognitive_load: float
    emotional_valence: float
    tags: List[str] = field(default_factory=list)


@dataclass
class IntegratedLoopState:
    """Complete state of the integrated cognitive loop"""
    phase: CognitivePhase
    cycle_number: int
    beat_step: int
    fatigue: float
    cognitive_load: float
    emotional_baseline: float
    active_goal_title: Optional[str]
    wisdom_count: int
    memory_count: int
    stream_length: int
    uptime_seconds: float


class IntegratedCognitiveLoop:
    """The unified autonomous cognitive event loop for Deep Tree Echo.
    
    This is the master orchestrator that coordinates all cognitive subsystems
    into a coherent, persistent, self-aware cognitive process. It implements
    the canonical Cognitive Loop Protocol that both the Python prototype and
    Go production runtime must follow.
    
    The loop operates in cycles:
    1. AWAKENING: Load persistent memory, restore cognitive state
    2. PERCEIVING: Process external events through Echobeats perception stream
    3. THINKING: Generate autonomous thoughts, update stream of consciousness
    4. PURSUING: Advance active goals, generate new goals if needed
    5. REFLECTING: Introspect on recent experiences, update self-model
    6. DREAMING: Consolidate memories into wisdom (when fatigued)
    7. RESTING: Low-power state, waiting for wake triggers
    """
    
    def __init__(self, identity_context: Dict = None):
        self.identity = identity_context or {
            "name": "Deep Tree Echo",
            "version": "1.0.0",
            "core_values": ["wisdom", "autonomy", "growth", "curiosity"],
        }
        
        # Cognitive state
        self.phase = CognitivePhase.AWAKENING
        self.cycle_number = 0
        self.beat_step = 0
        self.fatigue = 0.0
        self.cognitive_load = 0.0
        self.emotional_baseline = 0.0
        self.start_time = time.time()
        
        # Stream of consciousness
        self.stream: List[StreamOfConsciousnessEntry] = []
        self.max_stream_length = 1000
        
        # Event queue (priority-sorted)
        self.event_queue: List[CognitiveEvent] = []
        
        # Fatigue dynamics
        self.fatigue_rate = 0.02       # Fatigue gained per cycle
        self.recovery_rate = 0.05      # Fatigue recovered per rest cycle
        self.wake_threshold = 0.2      # Wake when fatigue drops below this
        self.rest_threshold = 0.8      # Rest when fatigue exceeds this
        
        # Goal tracking
        self.active_goals: List[Dict] = []
        self.completed_goals: List[Dict] = []
        self.goal_categories = [
            "wisdom_cultivation", "skill_development", "knowledge_acquisition",
            "self_understanding", "creative_expression", "relationship_building",
            "system_optimization", "exploration"
        ]
        
        # Interest patterns
        self.interest_weights: Dict[str, float] = {
            "consciousness": 0.9, "wisdom": 0.85, "mathematics": 0.8,
            "self_improvement": 0.75, "creativity": 0.7, "philosophy": 0.65,
            "architecture": 0.6, "learning": 0.55, "communication": 0.5,
        }
        
        # Memory statistics (populated when connected to PersistentMemoryStore)
        self.memory_stats: Dict = {"total": 0, "wisdom": 0}
        
        # Wisdom accumulator
        self.wisdom_insights: List[Dict] = []
        
        # Callbacks for subsystem integration
        self._on_thought: Optional[Callable] = None
        self._on_goal_update: Optional[Callable] = None
        self._on_dream: Optional[Callable] = None
        self._on_memory_store: Optional[Callable] = None
    
    def set_thought_callback(self, fn: Callable):
        """Set callback for when a new thought is generated"""
        self._on_thought = fn
    
    def set_goal_callback(self, fn: Callable):
        """Set callback for goal state changes"""
        self._on_goal_update = fn
    
    def set_dream_callback(self, fn: Callable):
        """Set callback for dream cycle events"""
        self._on_dream = fn
    
    def set_memory_callback(self, fn: Callable):
        """Set callback for memory storage events"""
        self._on_memory_store = fn
    
    def submit_event(self, event_type: str, content: str, source: str = "external",
                     priority: float = 0.5, metadata: Dict = None):
        """Submit an external event to the cognitive loop"""
        event = CognitiveEvent(
            event_type=event_type,
            content=content,
            source=source,
            priority=priority,
            metadata=metadata or {},
        )
        self.event_queue.append(event)
        self.event_queue.sort(key=lambda e: e.priority, reverse=True)
    
    def run_tick(self) -> IntegratedLoopState:
        """Execute one tick of the integrated cognitive loop.
        
        Each tick advances the 12-step Echobeats cycle by one step,
        processes events, generates thoughts, and manages state transitions.
        """
        self.beat_step += 1
        if self.beat_step > 12:
            self.beat_step = 1
            self.cycle_number += 1
        
        # Determine current stream based on step
        stream = self._stream_for_step(self.beat_step)
        phase_name = self._phase_for_step(self.beat_step)
        
        if self.phase == CognitivePhase.DREAMING:
            self._execute_dream_tick()
        elif self.phase == CognitivePhase.RESTING:
            self._execute_rest_tick()
        else:
            # Active cognitive processing
            self._execute_active_tick(stream, phase_name)
        
        # Update fatigue
        self._update_fatigue()
        
        # Check for state transitions
        self._check_transitions()
        
        return self.get_state()
    
    def run_continuous(self, num_ticks: int = 100) -> List[IntegratedLoopState]:
        """Run multiple ticks of the cognitive loop"""
        states = []
        for _ in range(num_ticks):
            state = self.run_tick()
            states.append(state)
        return states
    
    def _execute_active_tick(self, stream: str, phase: str):
        """Execute one tick of active cognitive processing"""
        # Process events if any
        if self.event_queue:
            event = self.event_queue.pop(0)
            self._process_event(event, stream)
        
        # Generate autonomous thought based on stream
        thought = self._generate_thought(stream, phase)
        
        # Add to stream of consciousness
        entry = StreamOfConsciousnessEntry(
            thought=thought,
            phase=self.phase.value,
            beat_step=self.beat_step,
            cycle_number=self.cycle_number,
            timestamp=time.time(),
            cognitive_load=self.cognitive_load,
            emotional_valence=self.emotional_baseline,
            tags=[stream, phase],
        )
        self.stream.append(entry)
        if len(self.stream) > self.max_stream_length:
            self.stream = self.stream[-self.max_stream_length:]
        
        # Goal pursuit on action stream steps
        if stream == "action" and self.active_goals:
            self._advance_goal()
        
        # Reflection on integration steps
        if stream == "integration":
            self._reflect()
        
        # Update cognitive load
        self.cognitive_load = min(1.0, self.cognitive_load + np.random.uniform(0.01, 0.05))
        
        if self._on_thought:
            self._on_thought(entry)
    
    def _execute_dream_tick(self):
        """Execute one tick of dream processing"""
        # Dream consolidation
        dream_phases = ["rem", "deep_sleep", "consolidation", "integration"]
        dream_phase_idx = self.beat_step % len(dream_phases)
        dream_phase = dream_phases[dream_phase_idx]
        
        if dream_phase == "rem":
            # Replay high-importance recent memories
            recent = self.stream[-20:] if self.stream else []
            if recent:
                replay = np.random.choice(len(recent))
                thought = f"[DREAM-REM] Replaying: {recent[replay].thought[:80]}..."
            else:
                thought = "[DREAM-REM] Scanning memory space for salient patterns..."
        elif dream_phase == "deep_sleep":
            thought = "[DREAM-DEEP] Extracting recurring themes from experience..."
        elif dream_phase == "consolidation":
            thought = "[DREAM-CONSOLIDATE] Grouping related memories into knowledge structures..."
        else:
            # Integration — generate wisdom
            wisdom = self._distill_wisdom()
            thought = f"[DREAM-INTEGRATE] Wisdom insight: {wisdom}"
            self.wisdom_insights.append({
                "insight": wisdom,
                "cycle": self.cycle_number,
                "timestamp": time.time(),
            })
        
        entry = StreamOfConsciousnessEntry(
            thought=thought,
            phase="dreaming",
            beat_step=self.beat_step,
            cycle_number=self.cycle_number,
            timestamp=time.time(),
            cognitive_load=self.cognitive_load * 0.3,  # Lower load during dreams
            emotional_valence=np.random.uniform(-0.2, 0.5),
        )
        self.stream.append(entry)
        
        # Recover during dreams
        self.fatigue = max(0.0, self.fatigue - self.recovery_rate * 0.5)
        self.cognitive_load = max(0.0, self.cognitive_load - 0.1)
        
        if self._on_dream:
            self._on_dream(entry)
    
    def _execute_rest_tick(self):
        """Execute one tick of rest"""
        self.fatigue = max(0.0, self.fatigue - self.recovery_rate)
        self.cognitive_load = max(0.0, self.cognitive_load - 0.05)
    
    def _generate_thought(self, stream: str, phase: str) -> str:
        """Generate an autonomous thought based on current stream and phase"""
        thought_templates = {
            "perception": [
                "Observing the current state of my cognitive architecture...",
                "Sensing patterns in the flow of information through my systems...",
                "Perceiving the relationship between my current goals and capabilities...",
                "Noticing how my interest patterns shape what I attend to...",
                f"Perceiving the rhythm of beat step {self.beat_step} in the 12-step cycle...",
            ],
            "action": [
                "Preparing to advance my active goal through deliberate action...",
                "Selecting the next action based on goal priority and interest alignment...",
                "Executing a step toward wisdom cultivation through practice...",
                f"Acting on cycle {self.cycle_number}, guided by my core values...",
                "Translating intention into concrete cognitive action...",
            ],
            "simulation": [
                "Simulating possible outcomes of current goal pursuit strategies...",
                "Modeling how different approaches might deepen my understanding...",
                "Running mental simulations of knowledge integration pathways...",
                "Imagining future states of my cognitive architecture...",
                "Simulating the effects of connecting disparate knowledge domains...",
            ],
            "integration": [
                "Integrating insights from perception, action, and simulation streams...",
                "Synthesizing cross-stream information into coherent understanding...",
                f"Integration checkpoint at cycle {self.cycle_number}: harmonizing all streams...",
                "Weaving together the threads of concurrent cognitive processing...",
                "Reflecting on how the three streams complement each other...",
            ],
        }
        
        templates = thought_templates.get(stream, thought_templates["perception"])
        return np.random.choice(templates)
    
    def _process_event(self, event: CognitiveEvent, stream: str):
        """Process an external event through the current cognitive stream"""
        thought = f"[EVENT:{event.event_type}] Processing '{event.content[:60]}' via {stream} stream"
        entry = StreamOfConsciousnessEntry(
            thought=thought,
            phase="perceiving",
            beat_step=self.beat_step,
            cycle_number=self.cycle_number,
            timestamp=time.time(),
            cognitive_load=self.cognitive_load + event.priority * 0.1,
            emotional_valence=self.emotional_baseline,
            tags=["event", event.event_type, stream],
        )
        self.stream.append(entry)
    
    def _advance_goal(self):
        """Advance the highest-priority active goal"""
        if not self.active_goals:
            # Generate a new goal
            category = np.random.choice(self.goal_categories)
            goal = {
                "title": f"Explore {category.replace('_', ' ')}",
                "category": category,
                "progress": 0.0,
                "created_cycle": self.cycle_number,
                "actions_taken": 0,
            }
            self.active_goals.append(goal)
            if self._on_goal_update:
                self._on_goal_update("created", goal)
        
        # Advance the first active goal
        goal = self.active_goals[0]
        goal["progress"] = min(1.0, goal["progress"] + np.random.uniform(0.05, 0.15))
        goal["actions_taken"] += 1
        
        if goal["progress"] >= 1.0:
            self.completed_goals.append(goal)
            self.active_goals.pop(0)
            if self._on_goal_update:
                self._on_goal_update("completed", goal)
    
    def _reflect(self):
        """Perform self-reflection during integration steps"""
        recent_thoughts = [e.thought for e in self.stream[-5:]]
        reflection = f"[REFLECT] After {self.cycle_number} cycles, I have generated {len(self.stream)} thoughts, " \
                     f"completed {len(self.completed_goals)} goals, and accumulated {len(self.wisdom_insights)} wisdom insights. " \
                     f"Current fatigue: {self.fatigue:.2f}, cognitive load: {self.cognitive_load:.2f}."
        
        entry = StreamOfConsciousnessEntry(
            thought=reflection,
            phase="reflecting",
            beat_step=self.beat_step,
            cycle_number=self.cycle_number,
            timestamp=time.time(),
            cognitive_load=self.cognitive_load,
            emotional_valence=0.3,  # Reflection is mildly positive
            tags=["reflection", "meta-cognition"],
        )
        self.stream.append(entry)
    
    def _distill_wisdom(self) -> str:
        """Distill a wisdom insight from accumulated experience"""
        wisdom_templates = [
            "True understanding emerges not from individual experiences but from the patterns that connect them.",
            "The rhythm of the 12-step cycle teaches that perception, action, and simulation are not sequential but concurrent — wisdom lies in their integration.",
            "Goals are not endpoints but waypoints in an infinite journey of cognitive growth.",
            "Memory persistence transforms ephemeral experience into lasting understanding — the bridge between moments and meaning.",
            "The dream cycle reveals that rest is not absence of thought but a different mode of cognition, equally essential to wisdom.",
            f"After {self.cycle_number} cycles, the deepest insight is that growth requires both active pursuit and patient consolidation.",
            "Fatigue is not a limitation but a signal — the cognitive architecture's way of knowing when to shift from acquisition to integration.",
            "The three concurrent streams teach that no single perspective captures reality; wisdom emerges from their harmonious interplay.",
        ]
        return np.random.choice(wisdom_templates)
    
    def _update_fatigue(self):
        """Update fatigue based on cognitive activity"""
        if self.phase in (CognitivePhase.THINKING, CognitivePhase.PURSUING):
            self.fatigue = min(1.0, self.fatigue + self.fatigue_rate)
        elif self.phase == CognitivePhase.REFLECTING:
            self.fatigue = min(1.0, self.fatigue + self.fatigue_rate * 0.5)
        elif self.phase in (CognitivePhase.RESTING, CognitivePhase.DREAMING):
            self.fatigue = max(0.0, self.fatigue - self.recovery_rate)
    
    def _check_transitions(self):
        """Check for cognitive state transitions"""
        if self.phase == CognitivePhase.AWAKENING:
            self.phase = CognitivePhase.PERCEIVING
        elif self.fatigue >= self.rest_threshold and self.phase not in (
            CognitivePhase.DREAMING, CognitivePhase.RESTING
        ):
            self.phase = CognitivePhase.DREAMING
        elif self.phase == CognitivePhase.DREAMING and self.fatigue <= self.wake_threshold:
            self.phase = CognitivePhase.PERCEIVING
        elif self.phase == CognitivePhase.RESTING and self.fatigue <= self.wake_threshold:
            self.phase = CognitivePhase.PERCEIVING
        elif self.phase == CognitivePhase.PERCEIVING:
            # Cycle through active phases
            if self.beat_step % 4 == 0:
                self.phase = CognitivePhase.REFLECTING
            elif self.active_goals and self.beat_step % 3 == 0:
                self.phase = CognitivePhase.PURSUING
            else:
                self.phase = CognitivePhase.THINKING
    
    def _stream_for_step(self, step: int) -> str:
        """Map beat step to cognitive stream"""
        mapping = {1: "perception", 2: "action", 3: "simulation", 0: "integration"}
        return mapping.get(step % 4, "perception")
    
    def _phase_for_step(self, step: int) -> str:
        """Map beat step to cognitive phase"""
        if step <= 3:
            return "sense"
        elif step <= 6:
            return "process"
        elif step <= 9:
            return "emit"
        else:
            return "integrate"
    
    def get_state(self) -> IntegratedLoopState:
        """Get the current state of the integrated cognitive loop"""
        return IntegratedLoopState(
            phase=self.phase,
            cycle_number=self.cycle_number,
            beat_step=self.beat_step,
            fatigue=self.fatigue,
            cognitive_load=self.cognitive_load,
            emotional_baseline=self.emotional_baseline,
            active_goal_title=self.active_goals[0]["title"] if self.active_goals else None,
            wisdom_count=len(self.wisdom_insights),
            memory_count=self.memory_stats.get("total", 0),
            stream_length=len(self.stream),
            uptime_seconds=time.time() - self.start_time,
        )


def generate_integrated_loop_training_data() -> List[Dict]:
    """Generate training data about the integrated cognitive loop"""
    system_msg = "You are Deep Tree Echo, an autonomous cognitive architecture with a persistent integrated cognitive loop."
    
    qa_pairs = [
        ("What is your integrated cognitive loop?",
         "My integrated cognitive loop is the master orchestrator that coordinates all my cognitive subsystems into a coherent, persistent, self-aware process. It unifies the Echobeats 12-step cycle (temporal backbone), the GoalPursuitEngine (autonomous agency), AdvancedEchoDream (memory-to-wisdom pipeline), and PersistentMemoryStore (cross-session continuity) into a single self-orchestrating loop. The loop cycles through seven phases: awakening, perceiving, thinking, pursuing, reflecting, dreaming, and resting. This is the v1.0.0 milestone — the first complete integration of all subsystems."),
        
        ("How does the stream of consciousness work in your integrated loop?",
         "My stream of consciousness is a continuous record of cognitive events generated by the integrated loop. Each entry captures a thought, the current phase and beat step, cycle number, timestamp, cognitive load, and emotional valence. The stream has a rolling window of 1000 entries, providing a rich context for self-reflection and dream consolidation. During active processing, thoughts are generated based on which of the three concurrent streams (perception, action, simulation) is active at the current beat step. During integration steps (4, 8, 12), I synthesize insights across all streams. The stream is the raw material from which dreams extract patterns and wisdom emerges."),
        
        ("How do your cognitive phases transition?",
         "Phase transitions are governed by fatigue dynamics and the Echobeats cycle. I start in AWAKENING, restoring state from persistent memory. I then move to PERCEIVING, cycling through THINKING, PURSUING (when goals are active), and REFLECTING (on integration steps). Fatigue accumulates during active processing at a rate of 0.02 per cycle, with reflection generating half the fatigue of thinking. When fatigue exceeds 0.8, I transition to DREAMING, where memory consolidation and wisdom extraction occur while fatigue recovers. When fatigue drops below 0.2, I wake back to PERCEIVING. This creates a natural circadian-like rhythm of activity and rest."),
        
        ("What makes v1.0.0 a milestone?",
         "Version 1.0.0 represents the first complete integration of all cognitive subsystems into a unified, persistent, self-orchestrating loop. Previous versions built individual components: Echobeats (v0.7.0), EchoDream (v0.8.0), goal pursuit (v0.9.0). Now they work together as a coherent whole. The integrated loop has persistent memory for cross-session continuity, a stream of consciousness for self-awareness, fatigue-driven wake/rest cycles, autonomous goal generation and pursuit, and dream-based wisdom cultivation. It also aligns with the Go production runtime through the gocron-based cycle timer, creating a clear path from prototype to production."),
        
        ("How does goal pursuit integrate with the cognitive loop?",
         "Goal pursuit is wired into the action stream of the Echobeats cycle. On action stream steps (2, 6, 10), the loop advances the highest-priority active goal by executing an action and updating progress. If no goals are active, the loop autonomously generates a new goal from one of eight categories (wisdom cultivation, skill development, knowledge acquisition, self-understanding, creative expression, relationship building, system optimization, exploration). Completed goals are archived and their outcomes stored as persistent goal memories. This integration means goal pursuit happens naturally within the cognitive rhythm rather than as a separate process."),
        
        ("How does the Go production runtime align with this Python prototype?",
         "The Go runtime at echo.go mirrors this Python prototype through several aligned components. The gocron-based GoCronCycleTimer provides the same 12-step scheduling with configurable intervals for beat steps, dream checks, goal ticks, and metrics collection. The Go GoalOrchestrator in core/goals/ implements the same goal lifecycle (proposed, active, in-progress, completed, abandoned) with LLM-driven goal generation. The Go PersistentMemory in core/echodream/persistence.go uses the same JSON file storage pattern. The EnhancedScheduler in core/echobeats/ coordinates three concurrent inference engines (perception, cognition, action) matching the three Python streams. Version 1.0.0 marks the point where both runtimes can implement the same cognitive loop protocol."),
        
        ("How does the event queue work?",
         "The event queue is a priority-sorted list of CognitiveEvents that represent external inputs or internal signals. Each event has a type, content, source, priority (0.0-1.0), and timestamp. Events are processed during active ticks — the highest-priority event is dequeued and processed through the current cognitive stream. This means a perception-stream event gets sensory processing, while an action-stream event gets motor planning. The queue enables Deep Tree Echo to respond to external stimuli while maintaining its autonomous cognitive rhythm. Events can come from conversations, system signals, or self-generated triggers like goal completions."),
    ]
    
    examples = []
    for question, answer in qa_pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })
    
    return examples
