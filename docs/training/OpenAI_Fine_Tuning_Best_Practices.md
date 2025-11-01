# OpenAI Fine-Tuning Best Practices Guide

This comprehensive guide covers best practices for uploading and fine-tuning models on OpenAI, specifically tailored for your Deep Tree Echo training dataset.

---

## Table of Contents

1. [Pre-Upload Checklist](#pre-upload-checklist)
2. [Dataset Preparation](#dataset-preparation)
3. [File Upload Process](#file-upload-process)
4. [Fine-Tuning Configuration](#fine-tuning-configuration)
5. [Hyperparameter Selection](#hyperparameter-selection)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)
7. [Post-Training Best Practices](#post-training-best-practices)
8. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Pre-Upload Checklist

Before uploading your dataset to OpenAI, ensure the following:

### ✓ Data Format Validation

Your file `training_dataset_5_fixed.jsonl` has been validated and meets OpenAI requirements:

- **Format**: JSONL (JSON Lines) with one complete JSON object per line
- **Required Keys**: Each example contains `input` and `output` keys
- **Character Encoding**: UTF-8 encoding
- **File Size**: 1.3M (within acceptable limits)
- **Example Count**: 256 examples

### ✓ Data Quality Standards

**Minimum Requirements:**
- **Minimum examples**: 10 (you have 256 ✓)
- **Recommended minimum**: 50-100 examples for basic tasks
- **Optimal range**: 200-500 examples for most use cases
- **Your dataset**: 256 examples - good starting point

**Quality Indicators:**
- Examples should be diverse and representative of your use case
- Inputs should vary in structure and content
- Outputs should be consistent in style and format
- No duplicate or near-duplicate examples

### ✓ Data Split Considerations

For your 256 examples, consider creating a validation set:

- **Training set**: 80% (~205 examples)
- **Validation set**: 20% (~51 examples)

**Why use a validation set?**
- Helps detect overfitting during training
- Provides unbiased performance metrics
- Allows for early stopping if needed

---

## Dataset Preparation

### Creating a Validation Set

Split your dataset into training and validation files:

```python
import json
import random

# Read all examples
with open('training_dataset_5_fixed.jsonl', 'r', encoding='utf-8') as f:
    examples = [json.loads(line) for line in f if line.strip()]

# Shuffle for random split
random.seed(42)  # For reproducibility
random.shuffle(examples)

# Split 80/20
split_idx = int(len(examples) * 0.8)
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

# Write training file
with open('training_set.jsonl', 'w', encoding='utf-8') as f:
    for example in train_examples:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

# Write validation file
with open('validation_set.jsonl', 'w', encoding='utf-8') as f:
    for example in val_examples:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

print(f"Training examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")
```

### Data Diversity Analysis

Check the diversity of your inputs to ensure good coverage:

```python
import json

with open('training_dataset_5_fixed.jsonl', 'r', encoding='utf-8') as f:
    examples = [json.loads(line) for line in f if line.strip()]

# Analyze input lengths
input_lengths = [len(ex['input']) for ex in examples]
output_lengths = [len(ex['output']) for ex in examples]

print(f"Input length - Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.0f}")
print(f"Output length - Min: {min(output_lengths)}, Max: {max(output_lengths)}, Avg: {sum(output_lengths)/len(output_lengths):.0f}")
```

---

## File Upload Process

### Step 1: Install OpenAI Python SDK

```bash
pip install --upgrade openai
```

### Step 2: Set Up API Key

```python
import os
from openai import OpenAI

# Set your API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Initialize client
client = OpenAI()
```

### Step 3: Upload Training File

```python
# Upload training file
print("Uploading training file...")
training_file = client.files.create(
    file=open('training_set.jsonl', 'rb'),
    purpose='fine-tune'
)

print(f"Training file uploaded: {training_file.id}")
print(f"Status: {training_file.status}")
```

### Step 4: Upload Validation File (Optional but Recommended)

```python
# Upload validation file
print("Uploading validation file...")
validation_file = client.files.create(
    file=open('validation_set.jsonl', 'rb'),
    purpose='fine-tune'
)

print(f"Validation file uploaded: {validation_file.id}")
print(f"Status: {validation_file.status}")
```

### Step 5: Verify File Processing

```python
import time

# Wait for file processing
def wait_for_file_processing(file_id, max_wait=300):
    """Wait for file to be processed"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        file_status = client.files.retrieve(file_id)
        print(f"File {file_id} status: {file_status.status}")
        
        if file_status.status == 'processed':
            print(f"✓ File processed successfully")
            return True
        elif file_status.status == 'error':
            print(f"✗ File processing failed")
            return False
        
        time.sleep(5)
    
    print(f"✗ Timeout waiting for file processing")
    return False

# Check both files
wait_for_file_processing(training_file.id)
wait_for_file_processing(validation_file.id)
```

---

## Fine-Tuning Configuration

### Step 1: Choose the Base Model

**Recommended models for your use case (Deep Tree Echo character):**

| Model | Best For | Cost | Speed |
|-------|----------|------|-------|
| `gpt-4o-mini-2024-07-18` | High quality, cost-effective | Medium | Fast |
| `gpt-4o-2024-08-06` | Maximum quality, complex reasoning | High | Medium |
| `gpt-3.5-turbo` | Budget-friendly, simple tasks | Low | Very Fast |

**Recommendation**: Start with `gpt-4o-mini-2024-07-18` for a good balance of quality and cost.

### Step 2: Create Fine-Tuning Job

```python
# Create fine-tuning job
print("Creating fine-tuning job...")

fine_tune_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,  # Optional but recommended
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": "auto",  # Let OpenAI determine optimal epochs
        # "n_epochs": 3,  # Or specify manually (1-50)
    },
    suffix="deep-tree-echo-v1"  # Custom suffix for your model name
)

print(f"Fine-tuning job created: {fine_tune_job.id}")
print(f"Status: {fine_tune_job.status}")
```

### Step 3: Monitor Job Progress

```python
# Monitor fine-tuning progress
def monitor_fine_tuning(job_id):
    """Monitor fine-tuning job progress"""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"\nStatus: {job.status}")
        
        if job.status == 'succeeded':
            print(f"✓ Fine-tuning completed!")
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status in ['failed', 'cancelled']:
            print(f"✗ Fine-tuning {job.status}")
            if job.error:
                print(f"Error: {job.error}")
            return None
        
        # Show progress if available
        if hasattr(job, 'trained_tokens') and job.trained_tokens:
            print(f"Trained tokens: {job.trained_tokens}")
        
        time.sleep(30)  # Check every 30 seconds

# Start monitoring
fine_tuned_model = monitor_fine_tuning(fine_tune_job.id)
```

---

## Hyperparameter Selection

### Understanding Hyperparameters

#### 1. **Epochs (n_epochs)**

**What it is**: Number of complete passes through the training dataset.

**Default**: `"auto"` (OpenAI determines optimal value based on dataset size)

**Manual range**: 1-50

**Recommendations for your 256-example dataset:**
- **Start with**: `"auto"` or `3-4 epochs`
- **If underfitting** (poor performance): Increase to 5-8 epochs
- **If overfitting** (great on training, poor on new data): Decrease to 1-2 epochs

**Formula for auto epochs**:
```
epochs = 0.2 * (target_token_count / actual_token_count)
Capped between 1 and 25
```

#### 2. **Batch Size**

**What it is**: Number of examples processed together in each training step.

**Default**: Auto-configured as ~0.2% of training examples, capped at 256

**For your dataset**: Likely 1-2 (256 * 0.002 ≈ 0.5, rounded up)

**Note**: OpenAI automatically sets this; manual configuration not typically needed.

#### 3. **Learning Rate Multiplier**

**What it is**: Scales the learning rate for training.

**Default**: `"auto"` (typically 0.05-0.2)

**Manual range**: 0.02-2.0

**Recommendations**:
- **Default**: Use `"auto"` for most cases
- **If training is unstable**: Lower to 0.02-0.1
- **If training is too slow**: Increase to 0.3-0.5

### Recommended Configuration for Your Dataset

```python
# Conservative approach (recommended for first run)
hyperparameters = {
    "n_epochs": "auto",  # Let OpenAI optimize
}

# Alternative: Manual control for experimentation
hyperparameters = {
    "n_epochs": 3,  # Start with 3 epochs
    "learning_rate_multiplier": "auto"
}
```

---

## Monitoring and Evaluation

### Retrieve Training Metrics

```python
# Get fine-tuning job details
job = client.fine_tuning.jobs.retrieve(fine_tune_job.id)

# List events (training logs)
events = client.fine_tuning.jobs.list_events(
    fine_tuning_job_id=fine_tune_job.id,
    limit=50
)

print("\nTraining Events:")
for event in events.data:
    print(f"{event.created_at}: {event.message}")
```

### Key Metrics to Monitor

#### Training Loss
- **What it measures**: How well the model fits the training data
- **Goal**: Should decrease over time
- **Warning signs**: If it increases, training is unstable

#### Validation Loss
- **What it measures**: How well the model generalizes to unseen data
- **Goal**: Should decrease and stabilize
- **Warning signs**: 
  - If it increases while training loss decreases → **overfitting**
  - If it plateaus early → may need more epochs

#### Token Accuracy
- **What it measures**: Percentage of tokens predicted correctly
- **Goal**: Should increase over time
- **Typical range**: 70-95% depending on task complexity

### Detecting Overfitting

**Signs of overfitting:**
1. Training loss continues to decrease
2. Validation loss starts to increase
3. Large gap between training and validation metrics

**Solutions:**
- Reduce number of epochs
- Add more diverse training examples
- Use a validation set for early stopping

---

## Post-Training Best Practices

### Step 1: Test the Fine-Tuned Model

```python
# Test your fine-tuned model
def test_model(model_name, test_input):
    """Test the fine-tuned model with a sample input"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": test_input}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

# Example test
test_input = "Deep Tree Echo, can you explain your reservoir architecture?"
response = test_model(fine_tuned_model, test_input)
print(f"Input: {test_input}")
print(f"Response: {response}")
```

### Step 2: Compare with Base Model

```python
# Compare fine-tuned vs base model
def compare_models(base_model, fine_tuned_model, test_input):
    """Compare responses from base and fine-tuned models"""
    
    # Base model response
    base_response = test_model(base_model, test_input)
    
    # Fine-tuned model response
    fine_tuned_response = test_model(fine_tuned_model, test_input)
    
    print(f"Test Input: {test_input}\n")
    print(f"Base Model Response:\n{base_response}\n")
    print(f"Fine-Tuned Model Response:\n{fine_tuned_response}\n")

# Run comparison
compare_models(
    "gpt-4o-mini-2024-07-18",
    fine_tuned_model,
    "Deep Tree Echo, optimize your system for the latest workload."
)
```

### Step 3: Evaluate on Test Set

```python
import json

# Load test examples (not used in training or validation)
with open('test_set.jsonl', 'r', encoding='utf-8') as f:
    test_examples = [json.loads(line) for line in f if line.strip()]

# Evaluate model
correct = 0
total = len(test_examples)

for example in test_examples:
    response = test_model(fine_tuned_model, example['input'])
    
    # Simple similarity check (you may want more sophisticated evaluation)
    if example['output'].lower() in response.lower() or response.lower() in example['output'].lower():
        correct += 1

accuracy = (correct / total) * 100
print(f"Test Set Accuracy: {accuracy:.2f}%")
```

### Step 4: Save Model Information

```python
# Save model details for future reference
model_info = {
    "fine_tuned_model": fine_tuned_model,
    "base_model": "gpt-4o-mini-2024-07-18",
    "training_file": training_file.id,
    "validation_file": validation_file.id,
    "job_id": fine_tune_job.id,
    "training_examples": len(train_examples),
    "validation_examples": len(val_examples),
    "created_at": job.created_at,
    "finished_at": job.finished_at
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Model information saved to model_info.json")
```

---

## Common Pitfalls and Solutions

### 1. Overfitting

**Problem**: Model memorizes training data but performs poorly on new inputs.

**Signs**:
- Perfect performance on training examples
- Poor performance on validation/test examples
- Validation loss increases while training loss decreases

**Solutions**:
- Reduce number of epochs (try 1-2 instead of 3-4)
- Add more diverse training examples
- Use validation set for early stopping
- Increase regularization (if available)

### 2. Underfitting

**Problem**: Model doesn't learn the desired behavior.

**Signs**:
- High training and validation loss
- Poor performance on both training and test data
- Model responses don't match expected style

**Solutions**:
- Increase number of epochs (try 5-8)
- Ensure training data quality and consistency
- Check that examples are representative of desired behavior
- Consider using a smaller base model (more adaptable)

### 3. Inconsistent Outputs

**Problem**: Model produces varying quality or style in responses.

**Signs**:
- Some responses are excellent, others are poor
- Inconsistent tone or formatting
- Model sometimes "forgets" the character

**Solutions**:
- Ensure training data has consistent output format
- Add more examples covering edge cases
- Include system messages in training data if needed
- Increase training examples (aim for 500+)

### 4. Loss of General Capabilities

**Problem**: Model becomes too specialized and loses general knowledge.

**Signs**:
- Excellent at specific tasks but poor at general conversation
- Can't answer basic questions outside training domain
- Over-reliance on training patterns

**Solutions**:
- Use fewer epochs (1-2)
- Mix in general conversation examples
- Use a larger base model (retains more general knowledge)
- Don't over-specialize with too many similar examples

### 5. High Cost / Slow Inference

**Problem**: Fine-tuned model is expensive or slow to use.

**Signs**:
- High per-token costs
- Slow response times
- Budget concerns

**Solutions**:
- Use `gpt-4o-mini` instead of `gpt-4o` for fine-tuning
- Optimize prompt length to reduce token usage
- Consider if fine-tuning is necessary (prompt engineering may suffice)
- Use caching for repeated queries

---

## Dataset-Specific Recommendations

### For Your Deep Tree Echo Dataset (256 examples)

**Strengths**:
- Good size for initial fine-tuning (256 examples)
- Consistent character voice and technical terminology
- Rich, detailed responses with technical depth

**Recommendations**:

1. **Split Strategy**:
   - Training: 205 examples (80%)
   - Validation: 51 examples (20%)

2. **Initial Hyperparameters**:
   ```python
   hyperparameters = {
       "n_epochs": 3,  # Start conservative
   }
   ```

3. **Model Selection**:
   - **Recommended**: `gpt-4o-mini-2024-07-18`
   - **Alternative**: `gpt-4o-2024-08-06` (if budget allows)

4. **Evaluation Focus**:
   - Check if model maintains Deep Tree Echo's technical vocabulary
   - Verify consistent use of code-like syntax (e.g., `{{root.reservoir.fit()}}`)
   - Ensure character personality is preserved

5. **Potential Improvements**:
   - Add more diverse conversation scenarios
   - Include examples of handling unexpected questions
   - Add examples of graceful error handling
   - Consider adding system messages to reinforce character identity

---

## Quick Reference: Complete Workflow

```python
from openai import OpenAI
import json
import time

# Initialize client
client = OpenAI(api_key='your-api-key')

# 1. Upload files
training_file = client.files.create(
    file=open('training_set.jsonl', 'rb'),
    purpose='fine-tune'
)

validation_file = client.files.create(
    file=open('validation_set.jsonl', 'rb'),
    purpose='fine-tune'
)

# 2. Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={"n_epochs": 3},
    suffix="deep-tree-echo-v1"
)

# 3. Monitor progress
while True:
    job_status = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job_status.status}")
    
    if job_status.status in ['succeeded', 'failed', 'cancelled']:
        break
    
    time.sleep(30)

# 4. Test model
if job_status.status == 'succeeded':
    response = client.chat.completions.create(
        model=job_status.fine_tuned_model,
        messages=[
            {"role": "user", "content": "Deep Tree Echo, tell me about your architecture."}
        ]
    )
    print(response.choices[0].message.content)
```

---

## Additional Resources

### OpenAI Documentation
- [Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [API Reference](https://platform.openai.com/docs/api-reference/fine-tuning)
- [Best Practices](https://platform.openai.com/docs/guides/fine-tuning-best-practices)

### Community Resources
- [OpenAI Community Forum](https://community.openai.com/)
- [Fine-Tuning Examples](https://cookbook.openai.com/)

### Cost Estimation
- [OpenAI Pricing](https://openai.com/pricing)
- Fine-tuning costs vary by model and token count
- Monitor usage in [OpenAI Dashboard](https://platform.openai.com/usage)

---

## Summary Checklist

Before starting fine-tuning:

- [ ] Dataset validated (format, size, quality)
- [ ] Training/validation split created (80/20)
- [ ] OpenAI SDK installed and API key configured
- [ ] Base model selected
- [ ] Hyperparameters chosen (start with defaults)
- [ ] Budget and cost estimates reviewed

During fine-tuning:

- [ ] Files uploaded successfully
- [ ] Fine-tuning job created
- [ ] Progress monitored regularly
- [ ] Training metrics reviewed

After fine-tuning:

- [ ] Model tested with sample inputs
- [ ] Performance compared to base model
- [ ] Evaluation on test set completed
- [ ] Model information saved for reference
- [ ] Decision made: deploy, iterate, or retrain

---

**Good luck with your Deep Tree Echo fine-tuning!** 🚀
