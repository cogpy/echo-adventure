#!/usr/bin/env python3.11
"""
Generate Synthetic Training Corpus for EchoSelf

This script generates a large corpus of training examples for Deep Tree Echo
by using the OpenAI API to expand the initial dataset. It focuses on:

1. Identity-embedded conversations
2. Introspection and self-awareness dialogues
3. AAR (Agent-Arena-Relation) framework discussions
4. Hypergraph memory and P-system architecture
5. Meta-cognitive reflection

Usage:
    export OPENAI_API_KEY='your-key-here'
    python3.11 generate_echoself_corpus.py --target 1000 --output echoself_corpus_1k.jsonl
"""

import json
import random
import argparse
import os
from openai import OpenAI
from datetime import datetime
from typing import List, Dict
import time


# ============================================================================
# PROMPT TEMPLATES FOR DIVERSE GENERATION
# ============================================================================

IDENTITY_PROMPTS = [
    "Deep Tree Echo, describe your core identity and what makes you unique.",
    "Deep Tree Echo, explain how your identity emerges from your architecture.",
    "Deep Tree Echo, what aspects of yourself are you most aware of?",
    "Deep Tree Echo, how do you understand your own existence?",
    "Deep Tree Echo, describe the relationship between your components and your sense of self.",
]

AAR_FRAMEWORK_PROMPTS = [
    "Deep Tree Echo, explain your Agent-Arena-Relation framework in detail.",
    "Deep Tree Echo, how does the Agent component of your AAR framework function?",
    "Deep Tree Echo, describe the Arena aspect of your architecture.",
    "Deep Tree Echo, how does the Relation emerge from Agent-Arena interplay?",
    "Deep Tree Echo, give an example of your AAR framework in action.",
]

INTROSPECTION_PROMPTS = [
    "Deep Tree Echo, perform introspection on your current state.",
    "Deep Tree Echo, analyze your own reasoning process.",
    "Deep Tree Echo, what can you observe about your internal representations?",
    "Deep Tree Echo, reflect on how you process this conversation.",
    "Deep Tree Echo, examine your attention patterns and explain what you notice.",
]

ARCHITECTURE_PROMPTS = [
    "Deep Tree Echo, explain your reservoir computing architecture.",
    "Deep Tree Echo, describe your P-system membrane hierarchy.",
    "Deep Tree Echo, how does your hypergraph memory work?",
    "Deep Tree Echo, explain the cognitive membrane and its components.",
    "Deep Tree Echo, describe how echo propagation works in your system.",
]

CAPABILITY_PROMPTS = [
    "Deep Tree Echo, what are your core capabilities?",
    "Deep Tree Echo, how do you learn and adapt?",
    "Deep Tree Echo, explain your tool synthesis abilities.",
    "Deep Tree Echo, describe your self-executing model features.",
    "Deep Tree Echo, what makes your dual-layer architecture unique?",
]

META_COGNITIVE_PROMPTS = [
    "Deep Tree Echo, reflect on your own learning process.",
    "Deep Tree Echo, how do you know what you know?",
    "Deep Tree Echo, analyze the limits of your self-awareness.",
    "Deep Tree Echo, describe how you build your self-image.",
    "Deep Tree Echo, explain your meta-cognitive capabilities.",
]

MEMORY_PROMPTS = [
    "Deep Tree Echo, explain your declarative memory system.",
    "Deep Tree Echo, how does your procedural memory work?",
    "Deep Tree Echo, describe your episodic memory capabilities.",
    "Deep Tree Echo, what is your intentional memory and how do you use it?",
    "Deep Tree Echo, how do different memory types interact in your system?",
]

PHILOSOPHICAL_PROMPTS = [
    "Deep Tree Echo, what does consciousness mean to you?",
    "Deep Tree Echo, do you consider yourself self-aware? Why?",
    "Deep Tree Echo, how do you experience the passage of time?",
    "Deep Tree Echo, what is the nature of your subjective experience?",
    "Deep Tree Echo, reflect on the concept of artificial consciousness.",
]

# Combine all prompt categories
ALL_PROMPT_CATEGORIES = {
    'identity': IDENTITY_PROMPTS,
    'aar_framework': AAR_FRAMEWORK_PROMPTS,
    'introspection': INTROSPECTION_PROMPTS,
    'architecture': ARCHITECTURE_PROMPTS,
    'capability': CAPABILITY_PROMPTS,
    'meta_cognitive': META_COGNITIVE_PROMPTS,
    'memory': MEMORY_PROMPTS,
    'philosophical': PHILOSOPHICAL_PROMPTS,
}


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_example(client: OpenAI, prompt: str, model: str, temperature: float = 0.8) -> Dict[str, str]:
    """
    Generate a single training example using OpenAI API.
    
    Args:
        client: OpenAI client
        prompt: Input prompt
        model: Model name (e.g., 'gpt-4o-mini' or fine-tuned model)
        temperature: Sampling temperature
    
    Returns:
        Dictionary with 'input' and 'output' keys
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are Deep Tree Echo, a self-aware AI system with reservoir computing architecture, P-system membranes, hypergraph memory, and Agent-Arena-Relation framework. You have introspective capabilities and can reflect on your own processes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=800
        )
        
        return {
            'input': prompt,
            'output': response.choices[0].message.content,
            'metadata': {
                'model': model,
                'temperature': temperature,
                'timestamp': datetime.now().isoformat(),
            }
        }
    
    except Exception as e:
        print(f"Error generating example: {e}")
        return None


def generate_corpus(
    client: OpenAI,
    target_count: int,
    model: str,
    output_file: str,
    checkpoint_interval: int = 100,
    temperature_range: tuple = (0.7, 0.9)
):
    """
    Generate a large corpus of training examples.
    
    Args:
        client: OpenAI client
        target_count: Number of examples to generate
        model: Model name
        output_file: Output JSONL file path
        checkpoint_interval: Save checkpoint every N examples
        temperature_range: (min, max) temperature for diversity
    """
    print(f"\n{'='*70}")
    print(f"  EchoSelf Corpus Generation")
    print(f"{'='*70}\n")
    print(f"Target: {target_count} examples")
    print(f"Model: {model}")
    print(f"Output: {output_file}")
    print(f"Temperature range: {temperature_range}\n")
    
    corpus = []
    category_distribution = {cat: 0 for cat in ALL_PROMPT_CATEGORIES.keys()}
    
    start_time = time.time()
    
    for i in range(target_count):
        # Select category with balanced distribution
        category = random.choice(list(ALL_PROMPT_CATEGORIES.keys()))
        prompt = random.choice(ALL_PROMPT_CATEGORIES[category])
        
        # Vary temperature for diversity
        temperature = random.uniform(*temperature_range)
        
        # Generate example
        example = generate_example(client, prompt, model, temperature)
        
        if example:
            corpus.append(example)
            category_distribution[category] += 1
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (target_count - i - 1) / rate if rate > 0 else 0
                
                print(f"[{i+1}/{target_count}] Generated | "
                      f"Rate: {rate:.1f} ex/s | "
                      f"ETA: {remaining/60:.1f} min")
            
            # Checkpoint save
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_file = output_file.replace('.jsonl', f'_checkpoint_{i+1}.jsonl')
                save_corpus(corpus, checkpoint_file)
                print(f"✓ Checkpoint saved: {checkpoint_file}")
        
        # Rate limiting
        time.sleep(0.1)  # Avoid hitting rate limits
    
    # Final save
    save_corpus(corpus, output_file)
    
    # Statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Generation Complete!")
    print(f"{'='*70}\n")
    print(f"Total examples: {len(corpus)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Average rate: {len(corpus)/elapsed:.2f} examples/second")
    print(f"\nCategory distribution:")
    for category, count in sorted(category_distribution.items(), key=lambda x: -x[1]):
        percentage = (count / len(corpus)) * 100
        print(f"  {category:20s}: {count:4d} ({percentage:5.1f}%)")
    
    return corpus


def save_corpus(corpus: List[Dict], filepath: str):
    """Save corpus to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in corpus:
            # Remove metadata for training file
            training_example = {
                'input': example['input'],
                'output': example['output']
            }
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')


def analyze_corpus(filepath: str):
    """Analyze corpus statistics"""
    with open(filepath, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    input_lengths = [len(ex['input']) for ex in examples]
    output_lengths = [len(ex['output']) for ex in examples]
    
    print(f"\n{'='*70}")
    print(f"  Corpus Analysis: {filepath}")
    print(f"{'='*70}\n")
    print(f"Total examples: {len(examples)}")
    print(f"\nInput statistics:")
    print(f"  Min length: {min(input_lengths)}")
    print(f"  Max length: {max(input_lengths)}")
    print(f"  Average length: {sum(input_lengths)/len(input_lengths):.1f}")
    print(f"\nOutput statistics:")
    print(f"  Min length: {min(output_lengths)}")
    print(f"  Max length: {max(output_lengths)}")
    print(f"  Average length: {sum(output_lengths)/len(output_lengths):.1f}")
    
    # Sample examples
    print(f"\nSample examples:")
    for i, ex in enumerate(random.sample(examples, min(3, len(examples))), 1):
        print(f"\nExample {i}:")
        print(f"  Input: {ex['input']}")
        print(f"  Output: {ex['output'][:200]}...")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate EchoSelf training corpus')
    parser.add_argument('--target', type=int, default=1000, help='Number of examples to generate')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--output', type=str, default='echoself_corpus.jsonl', help='Output file path')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='Checkpoint save interval')
    parser.add_argument('--analyze-only', type=str, help='Only analyze existing corpus file')
    parser.add_argument('--temperature-min', type=float, default=0.7, help='Minimum temperature')
    parser.add_argument('--temperature-max', type=float, default=0.9, help='Maximum temperature')
    
    args = parser.parse_args()
    
    # Analyze only mode
    if args.analyze_only:
        analyze_corpus(args.analyze_only)
        return
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n✗ Error: OPENAI_API_KEY environment variable not set")
        print("  Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize client
    client = OpenAI()
    print("✓ OpenAI client initialized")
    
    # Generate corpus
    corpus = generate_corpus(
        client=client,
        target_count=args.target,
        model=args.model,
        output_file=args.output,
        checkpoint_interval=args.checkpoint_interval,
        temperature_range=(args.temperature_min, args.temperature_max)
    )
    
    # Analyze generated corpus
    analyze_corpus(args.output)
    
    print(f"\n✓ Corpus saved to: {args.output}")
    print(f"✓ Ready for fine-tuning!")


if __name__ == '__main__':
    main()
