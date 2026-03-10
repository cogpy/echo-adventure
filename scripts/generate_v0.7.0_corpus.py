#!/usr/bin/env python3
"""
Echo Adventure v0.7.0 - Corpus Generation Script

Generates the expanded training corpus for echoself model growth by combining:
1. Echobeats 12-step cognitive cycle training data
2. Reservoir-augmented Q&A pairs
3. System 5 tetradic architecture examples
4. Cross-stream integration dynamics
5. Nested shell execution context encoding

This script produces training data in both OpenAI fine-tuning format and
NanEcho plain text format for the echoself repository.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from echo_adventure.echobeats import (
    EchobeatsCycle, generate_echobeats_training_data
)
from echo_adventure.reservoir_corpus_generator import (
    ReservoirCorpusGenerator, ReservoirCorpusExample
)


def main():
    print("=" * 80)
    print("Echo Adventure v0.7.0 - Corpus Generation")
    print("=" * 80)
    print()

    # Identity context
    identity_context = {
        "name": "Deep Tree Echo",
        "version": "0.7.0",
        "architecture": "Echobeats 12-step cognitive cycle with System 5 tetradic structure",
        "features": [
            "3 concurrent consciousness streams",
            "Reservoir echo state dynamics",
            "Agent-Arena-Relation geometry",
            "Nested shell execution contexts",
            "System 5 tensor bundles",
        ],
        "training_history": {
            "v0.2.0": "EchoSelf introspection module",
            "v0.3.0": "Introspection metrics and fine-tuning integration",
            "v0.4.0": "AAR geometric architecture and identity visualization",
            "v0.5.0": "Real-time AAR monitoring and autonomous corpus generation",
            "v0.6.0": "LLM-based corpus generation and autonomous self-improvement loop",
            "v0.7.0": "Echobeats cognitive cycle and reservoir-augmented generation",
        },
    }

    # Output directory
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Phase 1: Generate Echobeats-native training data
    print("[Phase 1] Generating Echobeats-native training data...")
    echobeats_examples = generate_echobeats_training_data(
        num_cycles=15,
        identity_context=identity_context,
    )
    print(f"  Generated {len(echobeats_examples)} Echobeats-native examples")

    # Phase 2: Generate reservoir-augmented corpus
    print("\n[Phase 2] Generating reservoir-augmented corpus...")
    generator = ReservoirCorpusGenerator(
        identity_context=identity_context,
        num_warmup_cycles=5,
        reservoir_dim=64,
    )

    corpus = generator.generate_corpus(
        num_examples=300,
        min_quality=0.50,
        min_diversity=0.15,
        include_echobeats_data=True,
    )
    print(f"  Generated {len(corpus)} reservoir-augmented examples")

    # Phase 3: Get corpus statistics
    stats = generator.get_corpus_stats(corpus)
    print(f"\n[Phase 3] Corpus Statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Average quality: {stats['avg_quality']:.4f}")
    print(f"  Average diversity: {stats['avg_diversity']:.4f}")
    print(f"  Categories: {json.dumps(stats['categories'], indent=4)}")
    print(f"  Sources: {json.dumps(stats['sources'], indent=4)}")

    # Phase 4: Export in multiple formats
    print("\n[Phase 4] Exporting corpus...")

    # OpenAI fine-tuning format
    openai_path = output_dir / "echoself_corpus_v0.7.0.jsonl"
    generator.export_openai_format(corpus, str(openai_path))
    print(f"  OpenAI format: {openai_path} ({len(corpus)} examples)")

    # NanEcho plain text format
    nanecho_path = output_dir / "echoself_nanecho_v0.7.0.jsonl"
    generator.export_nanecho_format(corpus, str(nanecho_path))
    print(f"  NanEcho format: {nanecho_path}")

    # Full metadata format
    metadata_path = output_dir / "echoself_corpus_metadata_v0.7.0.jsonl"
    generator.export_with_metadata(corpus, str(metadata_path))
    print(f"  Metadata format: {metadata_path}")

    # Phase 5: Generate temporal summary
    temporal_summary = generator.echobeats.get_temporal_summary()
    summary_path = output_dir / "echobeats_temporal_summary_v0.7.0.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "generation_timestamp": datetime.now().isoformat(),
            "version": "0.7.0",
            "corpus_stats": stats,
            "temporal_summary": temporal_summary,
            "identity_context": identity_context,
        }, f, indent=2)
    print(f"  Temporal summary: {summary_path}")

    # Phase 6: Summary
    print("\n" + "=" * 80)
    print("CORPUS GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total examples generated: {len(corpus)}")
    print(f"Echobeats cycles run: {generator.echobeats.current_cycle_number}")
    print(f"Total cognitive beats: {generator.echobeats.total_beats}")
    print(f"Output directory: {output_dir}")
    print()

    return corpus, stats


if __name__ == "__main__":
    main()
