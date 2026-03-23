#!/usr/bin/env python3
"""
Generate v1.2.0 training corpus for Deep Tree Echo.

This corpus teaches NanEcho about:
1. Live2D expression pipeline (endocrine → FACS → Cubism)
2. llama-cpp-skillm bridge (verb → cognitive state → expression)
3. Persona expression bridge (emotional state → cognitive state)
4. Reservoir-augmented examples about the new modules

Composition:
    /dte-autonomy-evolution ( /llama-cpp-skillm <=> /echo-evolve (
        /neuro-persona-evolve ( /live2d-avatar [ /live2d-miara -> /live2d-dtecho ] )
    ))
"""

import json
import os
import sys

# Add echo-adventure source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'echo_adventure'))

VERSION = "1.2.0"


def main():
    # Import new modules
    from live2d_expression import generate_live2d_expression_training_data
    from llama_cpp_skillm_bridge import generate_skillm_bridge_training_data

    print(f"Generating v{VERSION} corpus...")

    # Generate training data from each module
    live2d_data = generate_live2d_expression_training_data()
    print(f"  Live2D expression: {len(live2d_data)} examples")

    skillm_data = generate_skillm_bridge_training_data()
    print(f"  llama-cpp-skillm bridge: {len(skillm_data)} examples")

    # Try to import reservoir corpus generator for augmented examples
    try:
        from reservoir_corpus_generator import ReservoirCorpusGenerator
        identity_context = {"name": "Deep Tree Echo", "version": VERSION}
        generator = ReservoirCorpusGenerator(
            identity_context=identity_context,
            reservoir_dim=32,
        )
        reservoir_examples = generator.generate_corpus(
            num_examples=60,
            min_quality=0.4,
            min_diversity=0.15,
            include_echobeats_data=True,
        )
        reservoir_data = [{"messages": ex.messages} for ex in reservoir_examples]
        print(f"  Reservoir-augmented: {len(reservoir_data)} examples")
    except Exception as e:
        print(f"  Reservoir-augmented: skipped ({e})")
        reservoir_data = []

    # Combine all data
    all_data = live2d_data + skillm_data + reservoir_data
    print(f"  Total: {len(all_data)} examples")

    # Write JSONL corpus
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    corpus_path = os.path.join(data_dir, f'live2d_skillm_corpus_v{VERSION}.jsonl')
    with open(corpus_path, 'w') as f:
        for example in all_data:
            f.write(json.dumps(example) + '\n')
    print(f"  Written: {corpus_path} ({len(all_data)} lines)")

    # Write summary JSON
    summary = {
        "version": VERSION,
        "composition": "/dte-autonomy-evolution ( /llama-cpp-skillm <=> /echo-evolve ( /neuro-persona-evolve ( /live2d-avatar [ /live2d-miara -> /live2d-dtecho ] ) ) )",
        "modules": ["live2d_expression", "llama_cpp_skillm_bridge"],
        "examples": {
            "live2d_expression": len(live2d_data),
            "llama_cpp_skillm_bridge": len(skillm_data),
            "reservoir_augmented": len(reservoir_data),
            "total": len(all_data),
        },
        "new_concepts": [
            "Virtual endocrine system (16-channel hormone bus)",
            "FACS action unit decomposition (12 AUs)",
            "Cubism parameter mapping (12 params)",
            "MetaHuman Rig Logic CTRL_ curves (24 controls)",
            "Chaotic micro-expressions (saccades, breathing, asymmetric twitches)",
            "Cognitive mode → head/gaze pose (10 modes)",
            "Skillm verb → cognitive state mapping (10 verbs)",
            "llama.cpp 6-layer architecture → cogpy stack mapping",
            "DTE CoreSelfEngine inference pipeline (9 steps)",
            "Persona expression bridge (emotional state → cognitive state)",
        ],
    }
    summary_path = os.path.join(data_dir, f'live2d_skillm_summary_v{VERSION}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == '__main__':
    main()
