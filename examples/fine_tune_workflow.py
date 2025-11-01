#!/usr/bin/env python3.11
"""
Complete OpenAI Fine-Tuning Workflow Script
For Deep Tree Echo Training Dataset

This script handles the entire fine-tuning process:
1. Dataset splitting (training/validation)
2. File upload to OpenAI
3. Fine-tuning job creation
4. Progress monitoring
5. Model testing and evaluation
"""

import json
import time
import random
import os
from openai import OpenAI
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
INPUT_FILE = 'training_dataset_5_fixed.jsonl'
TRAINING_FILE = 'training_set.jsonl'
VALIDATION_FILE = 'validation_set.jsonl'
MODEL_INFO_FILE = 'model_info.json'

# Fine-tuning parameters
BASE_MODEL = "gpt-4o-mini-2024-07-18"  # Options: gpt-4o-mini-2024-07-18, gpt-4o-2024-08-06, gpt-3.5-turbo
MODEL_SUFFIX = "deep-tree-echo-v1"
N_EPOCHS = 3  # Or use "auto"
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# OpenAI API key (set via environment variable)
# export OPENAI_API_KEY='your-api-key-here'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def split_dataset(input_file, train_file, val_file, train_ratio=0.8, seed=42):
    """Split dataset into training and validation sets"""
    print_section("STEP 1: SPLITTING DATASET")
    
    # Read all examples
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total examples: {len(examples)}")
    
    # Shuffle for random split
    random.seed(seed)
    random.shuffle(examples)
    
    # Split
    split_idx = int(len(examples) * train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    # Write training file
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Write validation file
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✓ Training examples: {len(train_examples)} → {train_file}")
    print(f"✓ Validation examples: {len(val_examples)} → {val_file}")
    
    return len(train_examples), len(val_examples)


def analyze_dataset(file_path):
    """Analyze dataset statistics"""
    with open(file_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    input_lengths = [len(ex['input']) for ex in examples]
    output_lengths = [len(ex['output']) for ex in examples]
    
    print(f"\nDataset: {file_path}")
    print(f"  Examples: {len(examples)}")
    print(f"  Input length  - Min: {min(input_lengths):,}, Max: {max(input_lengths):,}, Avg: {sum(input_lengths)/len(input_lengths):,.0f}")
    print(f"  Output length - Min: {min(output_lengths):,}, Max: {max(output_lengths):,}, Avg: {sum(output_lengths)/len(output_lengths):,.0f}")


def upload_file(client, file_path, purpose='fine-tune'):
    """Upload a file to OpenAI"""
    print(f"\nUploading {file_path}...")
    
    with open(file_path, 'rb') as f:
        file_obj = client.files.create(file=f, purpose=purpose)
    
    print(f"✓ File uploaded: {file_obj.id}")
    print(f"  Status: {file_obj.status}")
    print(f"  Filename: {file_obj.filename}")
    print(f"  Bytes: {file_obj.bytes:,}")
    
    return file_obj


def wait_for_file_processing(client, file_id, max_wait=300):
    """Wait for file to be processed"""
    print(f"\nWaiting for file {file_id} to be processed...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        file_status = client.files.retrieve(file_id)
        
        if file_status.status == 'processed':
            print(f"✓ File processed successfully")
            return True
        elif file_status.status == 'error':
            print(f"✗ File processing failed")
            return False
        
        print(f"  Status: {file_status.status} (waiting...)")
        time.sleep(5)
    
    print(f"✗ Timeout waiting for file processing")
    return False


def create_fine_tuning_job(client, training_file_id, validation_file_id, model, suffix, n_epochs):
    """Create a fine-tuning job"""
    print_section("STEP 3: CREATING FINE-TUNING JOB")
    
    print(f"Configuration:")
    print(f"  Base model: {model}")
    print(f"  Suffix: {suffix}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Training file: {training_file_id}")
    print(f"  Validation file: {validation_file_id}")
    
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model,
        hyperparameters={"n_epochs": n_epochs},
        suffix=suffix
    )
    
    print(f"\n✓ Fine-tuning job created!")
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Created at: {datetime.fromtimestamp(job.created_at)}")
    
    return job


def monitor_fine_tuning(client, job_id):
    """Monitor fine-tuning job progress"""
    print_section("STEP 4: MONITORING FINE-TUNING PROGRESS")
    
    print(f"Job ID: {job_id}")
    print(f"Monitoring started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis may take 10-30 minutes depending on dataset size...")
    print("Status updates every 30 seconds:\n")
    
    iteration = 0
    while True:
        iteration += 1
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Status: {job.status}", end="")
        
        # Show progress if available
        if hasattr(job, 'trained_tokens') and job.trained_tokens:
            print(f" | Trained tokens: {job.trained_tokens:,}", end="")
        
        print()  # New line
        
        # Check if completed
        if job.status == 'succeeded':
            print(f"\n✓ Fine-tuning completed successfully!")
            print(f"  Fine-tuned model: {job.fine_tuned_model}")
            print(f"  Finished at: {datetime.fromtimestamp(job.finished_at)}")
            return job
        elif job.status in ['failed', 'cancelled']:
            print(f"\n✗ Fine-tuning {job.status}")
            if hasattr(job, 'error') and job.error:
                print(f"  Error: {job.error}")
            return None
        
        time.sleep(30)  # Check every 30 seconds


def get_training_events(client, job_id, limit=20):
    """Get training events and metrics"""
    print_section("STEP 5: TRAINING METRICS")
    
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id,
        limit=limit
    )
    
    print(f"Recent training events (last {limit}):\n")
    for event in reversed(events.data):
        timestamp = datetime.fromtimestamp(event.created_at).strftime('%H:%M:%S')
        print(f"[{timestamp}] {event.message}")


def test_model(client, model_name, test_input, temperature=0.7, max_tokens=500):
    """Test the fine-tuned model"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": test_input}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def compare_models(client, base_model, fine_tuned_model, test_inputs):
    """Compare base model vs fine-tuned model"""
    print_section("STEP 6: MODEL COMPARISON")
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{'─' * 70}")
        print(f"Test {i}: {test_input}")
        print('─' * 70)
        
        # Base model
        print("\n[BASE MODEL]")
        base_response = test_model(client, base_model, test_input)
        print(base_response[:300] + "..." if len(base_response) > 300 else base_response)
        
        # Fine-tuned model
        print("\n[FINE-TUNED MODEL]")
        fine_tuned_response = test_model(client, fine_tuned_model, test_input)
        print(fine_tuned_response[:300] + "..." if len(fine_tuned_response) > 300 else fine_tuned_response)


def save_model_info(job, training_file, validation_file, train_count, val_count, output_file):
    """Save model information for future reference"""
    model_info = {
        "fine_tuned_model": job.fine_tuned_model,
        "job_id": job.id,
        "base_model": job.model,
        "status": job.status,
        "training_file_id": training_file.id,
        "validation_file_id": validation_file.id,
        "training_examples": train_count,
        "validation_examples": val_count,
        "hyperparameters": {
            "n_epochs": job.hyperparameters.n_epochs if hasattr(job, 'hyperparameters') else None
        },
        "created_at": datetime.fromtimestamp(job.created_at).isoformat(),
        "finished_at": datetime.fromtimestamp(job.finished_at).isoformat() if job.finished_at else None,
        "trained_tokens": job.trained_tokens if hasattr(job, 'trained_tokens') else None
    }
    
    with open(output_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✓ Model information saved to {output_file}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main fine-tuning workflow"""
    print("\n" + "=" * 70)
    print("  OpenAI Fine-Tuning Workflow")
    print("  Deep Tree Echo Training Dataset")
    print("=" * 70)
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n✗ Error: OPENAI_API_KEY environment variable not set")
        print("  Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize client
    client = OpenAI()
    print("\n✓ OpenAI client initialized")
    
    # Step 1: Split dataset
    train_count, val_count = split_dataset(
        INPUT_FILE, 
        TRAINING_FILE, 
        VALIDATION_FILE, 
        TRAIN_SPLIT
    )
    
    # Analyze datasets
    analyze_dataset(TRAINING_FILE)
    analyze_dataset(VALIDATION_FILE)
    
    # Step 2: Upload files
    print_section("STEP 2: UPLOADING FILES")
    
    training_file = upload_file(client, TRAINING_FILE)
    validation_file = upload_file(client, VALIDATION_FILE)
    
    # Wait for processing
    if not wait_for_file_processing(client, training_file.id):
        print("✗ Training file processing failed. Aborting.")
        return
    
    if not wait_for_file_processing(client, validation_file.id):
        print("✗ Validation file processing failed. Aborting.")
        return
    
    # Step 3: Create fine-tuning job
    job = create_fine_tuning_job(
        client,
        training_file.id,
        validation_file.id,
        BASE_MODEL,
        MODEL_SUFFIX,
        N_EPOCHS
    )
    
    # Step 4: Monitor progress
    completed_job = monitor_fine_tuning(client, job.id)
    
    if not completed_job:
        print("\n✗ Fine-tuning failed. Check the error messages above.")
        return
    
    # Step 5: Get training metrics
    get_training_events(client, job.id)
    
    # Step 6: Test and compare models
    test_inputs = [
        "Deep Tree Echo, can you explain your reservoir architecture?",
        "Deep Tree Echo, optimize your system for the latest workload.",
        "How do your child reservoirs work together?"
    ]
    
    compare_models(client, BASE_MODEL, completed_job.fine_tuned_model, test_inputs)
    
    # Save model info
    print_section("STEP 7: SAVING MODEL INFORMATION")
    save_model_info(
        completed_job,
        training_file,
        validation_file,
        train_count,
        val_count,
        MODEL_INFO_FILE
    )
    
    # Final summary
    print_section("FINE-TUNING COMPLETE!")
    print(f"Your fine-tuned model: {completed_job.fine_tuned_model}")
    print(f"\nTo use this model in your code:")
    print(f"""
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="{completed_job.fine_tuned_model}",
    messages=[
        {{"role": "user", "content": "Your message here"}}
    ]
)

print(response.choices[0].message.content)
    """)
    
    print("\n" + "=" * 70)
    print("  All done! 🚀")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
