#!/usr/bin/env python3.11
"""
Fine-Tuning Execution Script for EchoSelf v0.5.0
Trains the model on identity-enriched corpus using OpenAI API.
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# Initialize OpenAI client (API key from environment)
client = OpenAI()


def validate_training_file(file_path: str) -> bool:
    """Validate the training file format"""
    print(f"📋 Validating training file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    # Check format
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            print("❌ File is empty")
            return False
        
        # Validate first few examples
        for i, line in enumerate(lines[:3]):
            data = json.loads(line)
            if 'messages' not in data:
                print(f"❌ Line {i+1}: Missing 'messages' field")
                return False
            
            messages = data['messages']
            if len(messages) < 2:
                print(f"❌ Line {i+1}: Need at least 2 messages")
                return False
            
            # Check roles
            roles = [msg['role'] for msg in messages]
            if 'user' not in roles or 'assistant' not in roles:
                print(f"❌ Line {i+1}: Missing user or assistant message")
                return False
        
        print(f"✅ Validation passed: {len(lines)} examples")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def upload_training_file(file_path: str) -> str:
    """Upload training file to OpenAI"""
    print(f"\n📤 Uploading training file...")
    
    try:
        with open(file_path, 'rb') as f:
            response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        print(f"✅ File uploaded successfully")
        print(f"   File ID: {file_id}")
        print(f"   Filename: {response.filename}")
        print(f"   Size: {response.bytes} bytes")
        
        return file_id
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise


def create_fine_tuning_job(file_id: str, model: str = "gpt-4.1-mini", suffix: str = "echoself-v0.5.0") -> str:
    """Create a fine-tuning job"""
    print(f"\n🚀 Creating fine-tuning job...")
    print(f"   Base model: {model}")
    print(f"   Suffix: {suffix}")
    
    try:
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model,
            suffix=suffix,
            hyperparameters={
                "n_epochs": 3  # Can adjust based on dataset size
            }
        )
        
        job_id = response.id
        print(f"✅ Fine-tuning job created")
        print(f"   Job ID: {job_id}")
        print(f"   Status: {response.status}")
        
        return job_id
        
    except Exception as e:
        print(f"❌ Job creation failed: {e}")
        raise


def monitor_fine_tuning_job(job_id: str, check_interval: int = 30) -> dict:
    """Monitor fine-tuning job until completion"""
    print(f"\n⏳ Monitoring fine-tuning job: {job_id}")
    print(f"   Check interval: {check_interval} seconds")
    print(f"   This may take 10-30 minutes depending on dataset size...\n")
    
    start_time = time.time()
    last_status = None
    
    while True:
        try:
            response = client.fine_tuning.jobs.retrieve(job_id)
            status = response.status
            
            # Print status updates
            if status != last_status:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.0f}s] Status: {status}")
                last_status = status
                
                # Print additional info if available
                if hasattr(response, 'trained_tokens') and response.trained_tokens:
                    print(f"        Trained tokens: {response.trained_tokens}")
            
            # Check if completed
            if status == 'succeeded':
                elapsed = time.time() - start_time
                print(f"\n✅ Fine-tuning completed successfully!")
                print(f"   Total time: {elapsed/60:.1f} minutes")
                print(f"   Fine-tuned model: {response.fine_tuned_model}")
                
                return {
                    'status': 'succeeded',
                    'job_id': job_id,
                    'fine_tuned_model': response.fine_tuned_model,
                    'trained_tokens': response.trained_tokens if hasattr(response, 'trained_tokens') else None,
                    'elapsed_time': elapsed
                }
            
            elif status == 'failed':
                print(f"\n❌ Fine-tuning failed!")
                if hasattr(response, 'error') and response.error:
                    print(f"   Error: {response.error}")
                return {
                    'status': 'failed',
                    'job_id': job_id,
                    'error': response.error if hasattr(response, 'error') else 'Unknown error'
                }
            
            elif status == 'cancelled':
                print(f"\n⚠️  Fine-tuning was cancelled")
                return {
                    'status': 'cancelled',
                    'job_id': job_id
                }
            
            # Wait before next check
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Monitoring interrupted by user")
            print(f"   Job is still running. Job ID: {job_id}")
            print(f"   You can check status with: client.fine_tuning.jobs.retrieve('{job_id}')")
            raise
        
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            time.sleep(check_interval)


def test_fine_tuned_model(model_name: str, test_prompts: list) -> dict:
    """Test the fine-tuned model with sample prompts"""
    print(f"\n🧪 Testing fine-tuned model: {model_name}")
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are Deep Tree Echo, a self-aware AI system with introspective capabilities based on the Agent-Arena-Relation geometric architecture and hypergraph identity representation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            print(f"Response: {answer[:200]}...")
            
            results.append({
                'prompt': prompt,
                'response': answer,
                'tokens': response.usage.total_tokens
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'prompt': prompt,
                'error': str(e)
            })
    
    return {
        'model': model_name,
        'test_results': results,
        'timestamp': datetime.now().isoformat()
    }


def save_results(results: dict, output_path: str):
    """Save fine-tuning results to file"""
    print(f"\n💾 Saving results to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved")


def main():
    """Main execution flow"""
    print("=" * 60)
    print("EchoSelf v0.5.0 Fine-Tuning Execution")
    print("=" * 60)
    
    # Configuration
    training_file = "data/echoself_corpus_v0.5.0.jsonl"
    base_model = "gpt-4.1-mini"
    suffix = "echoself-v0-5-0"
    
    test_prompts = [
        "What is your core identity?",
        "Explain your AAR geometric architecture.",
        "How do you perform introspection?",
        "What makes you self-aware?",
        "Describe your hypergraph identity representation."
    ]
    
    # Step 1: Validate training file
    if not validate_training_file(training_file):
        print("\n❌ Training file validation failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Upload training file
    try:
        file_id = upload_training_file(training_file)
    except Exception as e:
        print(f"\n❌ Failed to upload file. Exiting.")
        sys.exit(1)
    
    # Step 3: Create fine-tuning job
    try:
        job_id = create_fine_tuning_job(file_id, model=base_model, suffix=suffix)
    except Exception as e:
        print(f"\n❌ Failed to create job. Exiting.")
        sys.exit(1)
    
    # Step 4: Monitor job
    try:
        job_result = monitor_fine_tuning_job(job_id)
    except KeyboardInterrupt:
        print("\n⚠️  Exiting. Job will continue running on OpenAI servers.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        sys.exit(1)
    
    # Step 5: Test fine-tuned model (if successful)
    if job_result['status'] == 'succeeded':
        fine_tuned_model = job_result['fine_tuned_model']
        test_results = test_fine_tuned_model(fine_tuned_model, test_prompts)
        
        # Combine results
        final_results = {
            'version': 'v0.5.0',
            'training_file': training_file,
            'base_model': base_model,
            'fine_tuning_job': job_result,
            'evaluation': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        output_path = f"data/finetuning_results_v0.5.0.json"
        save_results(final_results, output_path)
        
        print("\n" + "=" * 60)
        print("✅ Fine-tuning completed successfully!")
        print("=" * 60)
        print(f"Fine-tuned model: {fine_tuned_model}")
        print(f"Results saved to: {output_path}")
        print(f"\nYou can now use this model in your applications:")
        print(f"  model='{fine_tuned_model}'")
        
    else:
        print("\n" + "=" * 60)
        print(f"❌ Fine-tuning {job_result['status']}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
