#!/usr/bin/env python3.11
"""
Dataset Splitter Utility
Splits a JSONL dataset into training and validation sets
"""

import json
import random
import argparse

def split_dataset(input_file, output_prefix, train_ratio=0.8, seed=42):
    """
    Split dataset into training and validation sets
    
    Args:
        input_file: Path to input JSONL file
        output_prefix: Prefix for output files (e.g., 'data' -> 'data_train.jsonl', 'data_val.jsonl')
        train_ratio: Ratio of training examples (default: 0.8 = 80%)
        seed: Random seed for reproducibility
    """
    # Read all examples
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total examples: {len(examples)}")
    
    # Shuffle for random split
    random.seed(seed)
    random.shuffle(examples)
    
    # Calculate split point
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Generate output filenames
    train_file = f"{output_prefix}_train.jsonl"
    val_file = f"{output_prefix}_val.jsonl"
    
    # Write training file
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Write validation file
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print summary
    print(f"\n✓ Split complete!")
    print(f"  Training:   {len(train_examples):3d} examples ({len(train_examples)/len(examples)*100:.1f}%) → {train_file}")
    print(f"  Validation: {len(val_examples):3d} examples ({len(val_examples)/len(examples)*100:.1f}%) → {val_file}")
    
    # Analyze data
    print(f"\nDataset Statistics:")
    
    for name, data in [("Training", train_examples), ("Validation", val_examples)]:
        input_lengths = [len(ex['input']) for ex in data]
        output_lengths = [len(ex['output']) for ex in data]
        
        print(f"\n{name} Set:")
        print(f"  Input length  - Min: {min(input_lengths):,}, Max: {max(input_lengths):,}, Avg: {sum(input_lengths)/len(input_lengths):,.0f}")
        print(f"  Output length - Min: {min(output_lengths):,}, Max: {max(output_lengths):,}, Avg: {sum(output_lengths)/len(output_lengths):,.0f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split JSONL dataset into training and validation sets')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('--output-prefix', '-o', default='dataset', help='Output file prefix (default: dataset)')
    parser.add_argument('--train-ratio', '-r', type=float, default=0.8, help='Training set ratio (default: 0.8)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_dataset(args.input_file, args.output_prefix, args.train_ratio, args.seed)
