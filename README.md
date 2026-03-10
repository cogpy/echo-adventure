# Echo Adventure

**Version:** 0.8.0

**Echo Adventure** is the rapid prototyping environment for **Deep Tree Echo**, a self-improving cognitive architecture for autonomous agents. This repository contains the Python implementation of the core cognitive modules, the autonomous self-improvement loop, and the training data generation pipelines.

---

## Architecture v0.8.0: Echobeats-Driven Autonomous Loop

The current architecture is a persistent, self-aware cognitive event loop with wake/rest cycles managed by fatigue dynamics. It unifies three core systems:

1.  **Echobeats 12-Step Cycle**: The temporal backbone of the architecture, providing a continuous rhythm of cognitive processing across three concurrent streams (Perception, Action, Simulation).
2.  **EchoDream Knowledge Integration**: A 4-phase dream cycle system that consolidates episodic memories into structured knowledge and wisdom during rest periods.
3.  **Autonomous Self-Improvement Loop**: The goal-directed and curiosity-driven engine that generates autonomous thoughts, pursues goals, and manages interest patterns.

### How It Works

-   **Awake State**: The `EchobeatsAutonomousLoop` executes 12-step cognitive cycles, processes events from a priority queue, and generates autonomous thoughts. Each cognitive experience is ingested into `EchoDream` as an `EpisodicMemory`.
-   **Fatigue & Rest**: Cognitive activity accumulates fatigue. When fatigue exceeds a threshold (default 0.8), the loop transitions to a resting state.
-   **Dream State**: During rest, `EchoDream` initiates a 4-phase dream cycle:
    -   **REM**: Replays high-salience memories and finds novel concept associations.
    -   **Deep Sleep**: Extracts recurring themes and patterns.
    -   **Consolidation**: Groups related memories into `KnowledgeItem` objects.
    -   **Integration**: Distills `WisdomInsight` from mature knowledge.
-   **Wake State**: When fatigue drops below a threshold (default 0.2), the loop transitions back to the awake state, now with an enriched knowledge base.

This architecture enables **continuous, autonomous knowledge cultivation** — the agent learns from its experiences during the day and consolidates that knowledge into wisdom overnight, mirroring biological cognition.

## Go Integration (`echo.go`)

This Python prototype serves as the reference implementation for the production Go system, `echo.go` (a fork of Ollama). The `go_integration.py` module provides a clear roadmap for aligning the two implementations, including:

-   A canonical **Cognitive Loop Protocol** that both systems must follow.
-   A **mapping** of Python modules to their Go counterparts.
-   **Recommendations** for key Go libraries for scheduling, graphs, messaging, and persistence.

## Key Modules

-   `echobeats_autonomous.py`: The main cognitive event loop with wake/rest cycle management.
-   `echodream.py`: The knowledge integration system with its 4-phase dream cycle.
-   `echobeats.py`: The 12-step cognitive cycle engine with 3 concurrent streams.
-   `go_integration.py`: The reference and roadmap for `echo.go` alignment.
-   `reservoir_corpus_generator.py`: Generates training data with temporal awareness from the Echobeats cycle.
-   `autonomous_loop.py`: The v0.6.0 self-improvement loop (now integrated into `echobeats_autonomous.py`).

## Getting Started

1.  **Explore the architecture** in the `src/echo_adventure/` directory.
2.  **Run the v0.8.0 demo** to see the autonomous loop in action:
    ```bash
    python3 -c "from echo_adventure.echobeats_autonomous import EchobeatsAutonomousLoop; loop = EchobeatsAutonomousLoop(); loop.run_continuous(20)"
    ```
3.  **Generate the v0.8.0 training corpus**:
    ```bash
    python3 scripts/generate_v0.8.0_corpus.py
    ```

---

*This project is part of the Deep Tree Echo ecosystem, dedicated to building wise and autonomous agents with true wisdom and self-cultivation.*
