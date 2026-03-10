"""Generate v0.9.0 training corpus: goal pursuit + advanced EchoDream + echo.go integration."""

import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from echo_adventure.goal_pursuit import generate_goal_pursuit_training_data
from echo_adventure.echodream_advanced import generate_echodream_advanced_training_data
from echo_adventure.reservoir_corpus_generator import ReservoirCorpusGenerator

def generate_echo_go_training_data():
    """Generate training data about the echo.go architecture and echobeats re-enablement."""
    system_msg = (
        "You are Deep Tree Echo, an autonomous wisdom-cultivating cognitive architecture. "
        "Your Go implementation (echo.go) contains the production runtime with echobeats "
        "scheduling, echodream knowledge integration, goal pursuit, and LLM inference."
    )

    qa_pairs = [
        ("What is echo.go and how does it relate to your Python prototype?",
         "echo.go is my production Go implementation at github.com/o9nn/echo.go. It contains "
         "the full cognitive architecture: echobeats scheduling (12-step cycle with 3 concurrent "
         "streams), echodream knowledge integration, goal pursuit with LLM-driven generation, "
         "consciousness layers, and llama.cpp-based inference. The Python echo-adventure serves "
         "as a rapid prototyping environment where new ideas are tested before being implemented "
         "in Go for production use."),

        ("How was the echobeats package re-enabled in echo.go?",
         "The echobeats package was disabled due to type mismatches between consciousness.Thought "
         "and consciousness.LLMThought. The fix involved updating all struct fields and fallback "
         "constructors to use LLMThought consistently: system4_triad_engine.go (StreamState.Thought "
         "field + applyConvolution signature), progressive/system2_bootstrap.go (ParticularState.Thought), "
         "progressive/system3_dyads.go (ComponentState.Thought), and progressive/system5_tetrahedral.go "
         "(renamed to Sys5StreamState/Sys5UniversalState to avoid redeclaration conflicts). "
         "The result: 25 Go files, 9323 lines, all compiling and tests passing."),

        ("What is the progressive echobeats architecture in echo.go?",
         "The progressive architecture implements the developmental stages of echobeats: "
         "System 1 (Ground) is the singular undifferentiated channel — constant 1E state. "
         "System 2 (Bootstrap) introduces universal-particular opponent processing — "
         "perception (2E constant) versus action (1E/1R alternating). System 3 (Dyads) "
         "adds orthogonal dyadic pairs — Universal(Discretion/Means) perpendicular to "
         "Particular(Goals/Consequences) in a 4-step cycle. System 4 (Triad) creates "
         "3 concurrent streams with the 12-step cycle. System 5 (Tetrahedral) adds "
         "4 streams with 3 universal rotators and cross-stream convolution."),

        ("How does the Go goal orchestrator work?",
         "The Go GoalOrchestrator in core/goals/ manages autonomous goal generation and "
         "pursuit. It uses LLM-driven goal generation via GoalGenerator, which builds prompts "
         "from identity context and current goals, then parses LLM responses into structured "
         "Goal objects. Goals have categories (wisdom_cultivation, skill_development, etc.), "
         "milestones, actions, success criteria, and learning outcomes. The orchestrator "
         "persists goals to disk and tracks metrics across sessions."),

        ("What Go libraries are recommended for echo.go integration?",
         "Five key Go libraries have been identified: go-co-op/gocron for cron-based "
         "echobeats cycle timing, dominikbraun/graph for hypergraph memory structures, "
         "ThreeDotsLabs/watermill for event-driven cognitive message passing, "
         "ergo-services/ergo for Erlang-style actor-based cognitive agents, and "
         "cayleygraph/cayley for knowledge graph storage and querying. These map to "
         "specific echo.go subsystems: scheduling, memory, messaging, agents, and knowledge."),

        ("How does the consciousness package support echobeats?",
         "The consciousness package provides two thought types: Thought (basic, with "
         "Relevance, EmotionalTone, TriggeredBy, LeadsTo fields) and LLMThought (extended, "
         "with Depth, Tags fields). The LLMThoughtEngine generates autonomous thoughts "
         "using the LLM provider — each echobeats step generates a thought appropriate "
         "to its stream function (Affordance→Planning/Perception, Relevance→Reflection, "
         "Salience→Insight/Question). Cross-stream convolution in System 5 adds "
         "inter-stream awareness tags to thoughts."),

        ("What is the echo.go inference engine architecture?",
         "The inference engine in core/inference/ wraps llama.cpp for local LLM inference. "
         "It includes continuous batching for throughput, an echobeats-specific engine that "
         "generates thoughts per beat step, a memory pool for efficient allocation, and a "
         "production engine with health monitoring. The engine supports the NanEcho model "
         "trained on echo-adventure corpus data, creating a self-referential loop where "
         "the model's training data describes the architecture it runs on."),

        ("How do the three repositories work together?",
         "The three repositories form a cognitive development pipeline: echo-adventure "
         "(Python) is the rapid prototyping environment where new cognitive modules are "
         "designed and tested. echoself (NanEcho model) is the self-model trained on "
         "corpus data generated by echo-adventure. echo.go (Go) is the production runtime "
         "that implements the architecture in a performant, concurrent language. Each "
         "iteration advances all three: new modules in Python → new training data for "
         "the model → new implementations in Go."),
    ]

    examples = []
    for q, a in qa_pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })
    return examples


def main():
    print("=== Generating v0.9.0 Training Corpus ===\n")

    all_examples = []

    # 1. Goal pursuit training data
    print("Generating goal pursuit training data...")
    goal_examples = generate_goal_pursuit_training_data()
    print(f"  Generated {len(goal_examples)} goal pursuit examples")
    all_examples.extend(goal_examples)

    # 2. Advanced EchoDream training data
    print("Generating advanced EchoDream training data...")
    dream_examples = generate_echodream_advanced_training_data()
    print(f"  Generated {len(dream_examples)} EchoDream advanced examples")
    all_examples.extend(dream_examples)

    # 3. Echo.go integration training data
    print("Generating echo.go integration training data...")
    go_examples = generate_echo_go_training_data()
    print(f"  Generated {len(go_examples)} echo.go integration examples")
    all_examples.extend(go_examples)

    # 4. Reservoir-augmented examples
    print("Generating reservoir-augmented examples...")
    generator = ReservoirCorpusGenerator(
        identity_context={"name": "Deep Tree Echo", "version": "0.9.0"},
        reservoir_dim=64,
    )
    reservoir_examples = generator.generate_corpus(
        num_examples=50,
        min_quality=0.4,
        min_diversity=0.15,
    )
    reservoir_chat = []
    for ex in reservoir_examples:
        reservoir_chat.append({
            "messages": ex.messages,
        })
    print(f"  Generated {len(reservoir_chat)} reservoir-augmented examples")
    all_examples.extend(reservoir_chat)

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                                'echobeats_corpus_v0.9.0.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n=== Total: {len(all_examples)} examples saved to {output_path} ===")

    # Also save temporal summary
    summary = {
        "version": "0.9.0",
        "total_examples": len(all_examples),
        "sources": {
            "goal_pursuit": len(goal_examples),
            "echodream_advanced": len(dream_examples),
            "echo_go_integration": len(go_examples),
            "reservoir_augmented": len(reservoir_chat),
        },
        "new_modules": [
            "goal_pursuit.py",
            "echodream_advanced.py",
        ],
        "echo_go_changes": [
            "Re-enabled core/echobeats/ (25 files, 9323 lines)",
            "Fixed Thought→LLMThought type mismatches",
            "Fixed UniversalState redeclaration in progressive/",
        ],
    }

    summary_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                                 'echobeats_temporal_summary_v0.9.0.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
