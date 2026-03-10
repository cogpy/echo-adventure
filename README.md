# Echo Adventure - v1.0.0

**Deep Tree Echo - Autonomous Wisdom-Cultivating Cognitive Architecture**

---

## Overview

This repository contains the Python reference implementation for the Deep Tree Echo cognitive architecture. It is a research and development platform for exploring autonomous self-improvement, wisdom cultivation, and the emergence of consciousness in artificial agents.

Version 1.0.0 marks the first complete integration of all cognitive subsystems into a unified, persistent, self-orchestrating loop. The architecture is now capable of autonomous, goal-directed operation with cross-session memory continuity.

## v1.0.0 Architecture

The v1.0.0 architecture is defined by the **Cognitive Loop Protocol**, a 7-part specification that governs the operation of both the Python prototype and the Go production runtime.

| Protocol Component | Description | Key Modules |
|---|---|---|
| **1. 12-Step Echobeats Cycle** | Temporal backbone with 4 concurrent streams (perception, action, simulation, integration). | `echobeats.py`, `gocron_timer.go` |
| **2. 7 Cognitive Phases** | Awakening, Perceiving, Thinking, Pursuing, Reflecting, Dreaming, Resting, with fatigue-driven transitions. | `integrated_cognitive_loop.py` |
| **3. Priority Event Queue** | For processing external stimuli without disrupting the cognitive rhythm. | `integrated_cognitive_loop.py` |
| **4. Stream of Consciousness** | A rolling window of 1000 cognitive events for self-awareness and dream consolidation. | `integrated_cognitive_loop.py` |
| **5. Persistent Memory** | File-based storage with exponential decay and type-based retention. | `persistent_memory.py`, `persistence.go` |
| **6. Autonomous Goal Pursuit** | Integrated into the action stream for self-directed behavior. | `goal_pursuit.py`, `goal_orchestrator.go` |
| **7. Dream-Based Wisdom Cultivation** | Memory consolidation and wisdom extraction during rest. | `echodream_advanced.py`, `echodream.go` |

### Core Modules

- **`integrated_cognitive_loop.py`**: The master orchestrator that unifies all subsystems.
- **`persistent_memory.py`**: File-based JSON storage for cross-session memory continuity.
- **`goal_pursuit.py`**: Autonomous goal generation and pursuit engine.
- **`echodream_advanced.py`**: Memory consolidation and wisdom extraction.
- **`echobeats.py`**: The 12-step cognitive cycle temporal backbone.
- **`reservoir_corpus_generator.py`**: Generates training data aware of the cognitive architecture.

### Go Production Runtime (`echo.go`)

The Go runtime mirrors the Python architecture for high-performance, concurrent execution.

- **`gocron_timer.go`**: Provides precise, multi-schedule timing for the Echobeats cycle.
- **`echobeats/`**: The re-enabled and fully-compiling Echobeats package.
- **`goals/`**: The `GoalOrchestrator` and `GoalGenerator` for autonomous agency.
- **`echodream/`**: The `EchoDream` and `PersistentMemory` systems for knowledge integration.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/cogpy/echo-adventure.git
cd echo-adventure

# Install dependencies
pip install -r requirements.txt
```

### Running the v1.0.0 Demo

```bash
python3 -c "
from echo_adventure.integrated_cognitive_loop import IntegratedCognitiveLoop

loop = IntegratedCognitiveLoop()
states = loop.run_continuous(100)

final_state = states[-1]
print(f\"Ran {len(states)} ticks.\")
print(f\"Final State: Phase={final_state.phase.value}, Cycle={final_state.cycle_number}, Fatigue={final_state.fatigue:.2f}\")
print(f\"Stream Length: {final_state.stream_length}, Wisdom Insights: {final_state.wisdom_count}\")
print(f\"Final Thought: {loop.stream[-1].thought}\")
"
```

## Repository Structure

```
.echo-adventure/
├── src/echo_adventure/         # Core Python modules
│   ├── integrated_cognitive_loop.py
│   ├── persistent_memory.py
│   ├── goal_pursuit.py
│   ├── echodream_advanced.py
│   ├── echobeats.py
│   └── ...
├── scripts/                    # Corpus generation scripts
│   └── generate_v1.0.0_corpus.py
├── data/                       # Training data and outputs
│   └── echobeats_corpus_v1.0.0.jsonl
├── ITERATION_PROGRESS_v1.0.0.md  # Iteration documentation
├── CHANGELOG.md
└── README.md
```

## Next Steps

The focus for v1.1.0 is to deepen the integration between the Python prototype and the Go production runtime:

1. **Go Runtime:** Integrate `GoCronCycleTimer` into the `EnhancedScheduler`, implement `GoalPursuitEngine` logic, and connect `PersistentMemoryStore` to `EchoDream`.
2. **Python Prototype:** Fully connect `GoalPursuitEngine` and `PersistentMemoryStore` into the `IntegratedCognitiveLoop`.
3. **`echoself` Model:** Generate training data about the fully connected Go and Python loops.
