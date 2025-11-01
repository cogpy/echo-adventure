#!/usr/bin/env python3.11
"""
Convert training dataset from prompt-completion format to input-output format.
This fixes the error: "Invalid file format. Example 1 is missing key 'input'."
"""

import json
import sys

def convert_format(input_file, output_file):
    """
    Convert JSONL file from prompt-completion format to input-output format.
    
    Args:
        input_file: Path to input JSONL file with prompt-completion format
        output_file: Path to output JSONL file with input-output format
    """
    converted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse the JSON object
                data = json.loads(line)
                
                # Convert format: prompt -> input, completion -> output
                if 'prompt' in data and 'completion' in data:
                    converted_data = {
                        'input': data['prompt'],
                        'output': data['completion']
                    }
                    
                    # Write the converted line
                    outfile.write(json.dumps(converted_data, ensure_ascii=False) + '\n')
                    converted_count += 1
                else:
                    print(f"Warning: Line {line_num} missing 'prompt' or 'completion' keys", file=sys.stderr)
                    error_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                error_count += 1
                continue
    
    print(f"Conversion complete!")
    print(f"  Converted: {converted_count} examples")
    print(f"  Errors: {error_count} examples")
    print(f"  Output file: {output_file}")
    
    return converted_count, error_count

if __name__ == '__main__':
    input_file = '/home/ubuntu/upload/training_dataset_5.jsonl'
    output_file = '/home/ubuntu/training_dataset_5_fixed.jsonl'
    
    convert_format(input_file, output_file)
