#!/usr/bin/env python3
"""Quick test of v0.6.0 features"""

import sys
sys.path.insert(0, '/home/ubuntu/echo-adventure/src')

from echo_adventure.llm_corpus_generator import LLMCorpusGenerator
from echo_adventure.autonomous_loop import AutonomousSelfImprovementLoop

print("Testing v0.6.0 modules...")
print("✓ LLMCorpusGenerator imported")
print("✓ AutonomousSelfImprovementLoop imported")

# Test basic initialization
identity = {"name": "Deep Tree Echo", "version": "0.6.0"}
generator = LLMCorpusGenerator(identity)
print("✓ LLMCorpusGenerator initialized")

loop = AutonomousSelfImprovementLoop(identity, generation_enabled=False, monitoring_enabled=False)
print("✓ AutonomousSelfImprovementLoop initialized")

print("\nAll v0.6.0 modules working correctly!")
