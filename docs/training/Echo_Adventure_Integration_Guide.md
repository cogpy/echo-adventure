# Echo Adventure Integration Guide

**Adding Self-Executing Capabilities to Echo Adventure**

---

## Overview

This guide shows how to integrate the self-executing model architecture into your existing `echo-adventure` repository, enabling Deep Tree Echo to generate and execute dynamic code that references its own state.

---

## Step 1: Add Execution Engine to Echo Adventure

### File Structure

```
echo-adventure/
├── src/echo_adventure/
│   ├── __init__.py
│   ├── transformer.py
│   ├── inference_engine.py
│   ├── model.py
│   ├── trainer.py               # NEW
│   └── execution_engine.py      # NEW
├── examples/
│   ├── example.py
│   ├── train_deep_tree_echo.py  # NEW
│   └── self_executing_demo.py   # NEW
└── tests/
    └── test_execution_engine.py  # NEW
```

### Copy Execution Engine

```bash
cd echo-adventure
cp /home/ubuntu/echo-adventure-execution-engine.py src/echo_adventure/execution_engine.py
```

### Update `__init__.py`

```python
# src/echo_adventure/__init__.py

from .transformer import TransformerLayer
from .inference_engine import InferenceEngine
from .model import TwoLayerModel
from .execution_engine import (
    DynamicTemplateEngine,
    ExecutionConfig,
    SelfExecutingModelMixin,
)

__all__ = [
    'TransformerLayer',
    'InferenceEngine',
    'TwoLayerModel',
    'DynamicTemplateEngine',
    'ExecutionConfig',
    'SelfExecutingModelMixin',
]
```

---

## Step 2: Create Self-Executing Model

### Option A: Extend Existing TwoLayerModel

```python
# src/echo_adventure/model.py

from .execution_engine import DynamicTemplateEngine, ExecutionConfig

class TwoLayerModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ... existing initialization ...
        
        # Add template engine
        self.template_engine = DynamicTemplateEngine(
            self,
            config=ExecutionConfig(
                allow_modification=False,
                log_executions=False,  # Disable in production
            )
        )
    
    def generate_and_execute(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        execute_dynamic_code: bool = True,
        tokenizer = None,
    ) -> str:
        """
        Generate text and execute embedded dynamic code
        """
        # Generate
        generated_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_inference_engine=True
        )
        
        # Decode
        if tokenizer is not None:
            raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            raw_text = str(generated_ids.tolist())
        
        # Execute dynamic code
        if execute_dynamic_code:
            final_text = self.template_engine.render(raw_text)
        else:
            final_text = raw_text
        
        return final_text
```

### Option B: Use Mixin (Cleaner)

```python
# src/echo_adventure/model.py

from .execution_engine import SelfExecutingModelMixin

class TwoLayerModel(SelfExecutingModelMixin, nn.Module):
    """
    Two-layer model with self-executing capabilities
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SelfExecutingModelMixin automatically adds template_engine
```

---

## Step 3: Create Training Dataset

### Dataset Format

Create training examples that teach the model to use dynamic code:

```jsonl
{"input": "What is your temperature?", "output": "My current temperature is {{temperature}}."}
{"input": "How many layers do you have?", "output": "I have {{self.num_layers}} transformer layers."}
{"input": "Describe your configuration", "output": "I'm running with temperature={{temperature:.2f}}, top_p={{top_p:.2f}}, using {{self.num_layers}} layers."}
{"input": "What's your top layer weight?", "output": "My top layer has weight {{layer_weights[-1]:.4f}}."}
{"input": "Are you being creative?", "output": "I'm being {{'very creative' if temperature > 1.0 else 'creative' if temperature > 0.8 else 'balanced' if temperature > 0.5 else 'precise'}} with temperature={{temperature:.2f}}."}
{"input": "How many parameters do you have?", "output": "I have {{self.count_parameters()['total_params']:,}} total parameters."}
{"input": "What's the average layer weight?", "output": "The average layer weight is {{sum(layer_weights) / len(layer_weights):.4f}}."}
{"input": "Which layer is most important?", "output": "Layer {{layer_weights.argmax()}} has the highest weight of {{layer_weights.max():.4f}}."}
```

### Generate Dataset

```python
# scripts/generate_self_executing_dataset.py

import json

templates = [
    # Self-reporting
    ("What is your temperature?", "My current temperature is {{temperature}}."),
    ("What is your top_p?", "My top_p is {{top_p}}."),
    ("What is your repetition penalty?", "My repetition penalty is {{repetition_penalty}}."),
    
    # Configuration
    ("How many layers do you have?", "I have {{self.num_layers}} transformer layers."),
    ("How many attention heads?", "I have {{self.num_heads}} attention heads per layer."),
    ("What's your vocabulary size?", "My vocabulary size is {{self.vocab_size:,}} tokens."),
    
    # Introspection
    ("Describe your configuration", 
     "I'm running with temperature={{temperature:.2f}}, top_p={{top_p:.2f}}, using {{self.num_layers}} layers with {{self.num_heads}} heads each."),
    
    # Adaptive responses
    ("Are you being creative?",
     "I'm being {{'very creative' if temperature > 1.0 else 'creative' if temperature > 0.8 else 'balanced' if temperature > 0.5 else 'precise'}} with temperature={{temperature:.2f}}."),
    
    # Computations
    ("What's your total parameter count?",
     "I have {{self.count_parameters()['total_params']:,}} total parameters."),
    
    ("What's the average layer weight?",
     "The average layer weight is {{sum(layer_weights) / len(layer_weights):.4f}}."),
    
    # Complex queries
    ("Which layer is most important?",
     "Layer {{list(layer_weights).index(max(layer_weights))}} has the highest weight of {{max(layer_weights):.4f}}."),
]

# Generate dataset
with open('self_executing_training.jsonl', 'w') as f:
    for input_text, output_text in templates:
        example = {"input": input_text, "output": output_text}
        f.write(json.dumps(example) + '\n')

print(f"Generated {len(templates)} training examples")
```

---

## Step 4: Training Script

```python
# examples/train_deep_tree_echo.py

import sys
sys.path.insert(0, '../src')

from echo_adventure import TwoLayerModel
import torch
from torch.utils.data import Dataset, DataLoader
import json

class SelfExecutingDataset(Dataset):
    """Dataset with self-executing examples"""
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine input and output
        text = item['input'] + " " + item['output']
        
        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
        
        return {
            'input_ids': torch.tensor(tokens),
            'raw_input': item['input'],
            'raw_output': item['output'],
        }

def train_epoch(model, dataloader, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Language modeling loss
        loss = torch.nn.CrossEntropyLoss()(
            logits[:, :-1].contiguous().view(-1, model.vocab_size),
            input_ids[:, 1:].contiguous().view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Configuration
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    
    # Create model
    model = TwoLayerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        init_temperature=1.0,
        init_top_p=0.9,
        init_repetition_penalty=1.0,
    )
    
    # Create dataset (you'll need a proper tokenizer)
    # For demo, using simple tokenization
    dataset = SelfExecutingDataset(
        'self_executing_training.jsonl',
        tokenizer=None,  # Replace with actual tokenizer
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    for epoch in range(10):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
        
        # Test self-execution
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_input = "What is your temperature?"
            # Generate and execute
            # (implementation depends on your tokenizer)
            print(f"Test: {test_input}")
    
    # Save model
    torch.save(model.state_dict(), 'deep_tree_echo_self_executing.pt')

if __name__ == '__main__':
    main()
```

---

## Step 5: Demo Script

```python
# examples/self_executing_demo.py

import sys
sys.path.insert(0, '../src')

from echo_adventure import TwoLayerModel
import torch

def demo():
    print("=" * 70)
    print("Deep Tree Echo Self-Executing Model Demo")
    print("=" * 70)
    
    # Create model
    model = TwoLayerModel(
        vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
    )
    
    # Test templates
    templates = [
        "My temperature is {{temperature}}",
        "I have {{self.num_layers}} layers",
        "Config: temp={{temperature:.2f}}, top_p={{top_p:.2f}}",
        "I'm {{'creative' if temperature > 0.8 else 'precise'}}",
        "Total params: {{self.count_parameters()['total_params']:,}}",
    ]
    
    print("\nRendering templates with current model state:\n")
    
    for template in templates:
        result = model.template_engine.render(template)
        print(f"Template: {template}")
        print(f"Result:   {result}\n")
    
    # Show introspection
    print("=" * 70)
    print("Model Introspection")
    print("=" * 70)
    
    state = model.introspect()
    for key, value in state.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    demo()
```

---

## Step 6: Test Suite

```python
# tests/test_execution_engine.py

import pytest
import torch
from echo_adventure import TwoLayerModel

def test_simple_variable():
    """Test simple variable substitution"""
    model = TwoLayerModel(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    template = "Temperature: {{temperature}}"
    result = model.template_engine.render(template)
    
    assert "Temperature:" in result
    assert result != template  # Should be substituted

def test_format_specifier():
    """Test format specifiers"""
    model = TwoLayerModel(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    template = "Temp: {{temperature:.2f}}"
    result = model.template_engine.render(template)
    
    # Should have exactly 2 decimal places
    assert result.count('.') == 1
    parts = result.split('.')
    assert len(parts[1]) == 2

def test_conditional():
    """Test conditional expressions"""
    model = TwoLayerModel(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    template = "{{'hot' if temperature > 1.0 else 'cold'}}"
    result = model.template_engine.render(template)
    
    assert result in ['hot', 'cold']

def test_function_call():
    """Test function calls"""
    model = TwoLayerModel(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    template = "Layers: {{self.num_layers}}"
    result = model.template_engine.render(template)
    
    assert "Layers: 2" in result

def test_array_access():
    """Test array indexing"""
    model = TwoLayerModel(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    template = "First weight: {{layer_weights[0]:.4f}}"
    result = model.template_engine.render(template)
    
    assert "First weight:" in result
    assert "." in result  # Has decimal point

def test_error_handling():
    """Test error handling for invalid code"""
    model = TwoLayerModel(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    template = "{{invalid_variable}}"
    result = model.template_engine.render(template)
    
    assert "ERROR" in result

def test_read_only_protection():
    """Test that model is read-only by default"""
    model = TwoLayerModel(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    
    # Should not allow modification
    template = "{{self.num_layers = 100}}"
    result = model.template_engine.render(template)
    
    assert "ERROR" in result
    assert model.num_layers == 2  # Unchanged

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Step 7: Advanced Features

### Memory Integration

```python
# Add to execution_engine.py

class HypergraphMemoryAccessor:
    """
    Access hypergraph memory from dynamic code
    """
    def __init__(self, memory):
        self.memory = memory
    
    def get(self, key: str, default=None):
        """Get memory value"""
        return self.memory.get(key, default)
    
    def query(self, pattern: str, top_k: int = 5):
        """Query memory"""
        return self.memory.query(pattern, top_k=top_k)
    
    def __getitem__(self, key):
        return self.memory[key]

# In ExecutionContext.build_context():
if hasattr(self.model, 'memory'):
    context['memory'] = HypergraphMemoryAccessor(self.model.memory)
```

**Usage**:
```python
template = "I remember: {{memory.get('reservoir')}}"
template = "Related concepts: {{memory.query('P-system', top_k=3)}}"
```

### AAR Architecture Integration

```python
# Add AAR components to context

class AARAccessor:
    """Access Agent-Arena-Relation components"""
    def __init__(self, model):
        self.model = model
    
    @property
    def agent(self):
        """Urge-to-act (query tensors)"""
        return self.model.get_agent_state()
    
    @property
    def arena(self):
        """Need-to-be (key-value memory)"""
        return self.model.get_arena_state()
    
    @property
    def relation(self):
        """Self (attention weights)"""
        return self.model.get_relation_state()

# In ExecutionContext.build_context():
if hasattr(self.model, 'aar'):
    context['aar'] = AARAccessor(self.model)
```

**Usage**:
```python
template = "My urge-to-act is {{aar.agent.norm():.2f}}"
template = "Arena state: {{aar.arena.shape}}"
```

---

## Complete Example

### Full Integration

```python
# examples/complete_self_executing_example.py

from echo_adventure import TwoLayerModel
import torch

# Create model
model = TwoLayerModel(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
)

# Simulate user interaction
prompts = [
    "What is your temperature?",
    "Describe your configuration",
    "Are you being creative?",
    "How many parameters do you have?",
]

# Model responses (would be generated by neural network)
responses = [
    "My current temperature is {{temperature}}.",
    "I'm running with temperature={{temperature:.2f}}, top_p={{top_p:.2f}}, using {{self.num_layers}} layers.",
    "I'm being {{'very creative' if temperature > 1.0 else 'creative' if temperature > 0.8 else 'balanced' if temperature > 0.5 else 'precise'}} with temperature={{temperature:.2f}}.",
    "I have {{self.count_parameters()['total_params']:,}} total parameters across {{self.num_layers}} layers.",
]

print("=" * 70)
print("Deep Tree Echo Self-Executing Conversation")
print("=" * 70)

for prompt, response_template in zip(prompts, responses):
    print(f"\nUser: {prompt}")
    
    # Execute dynamic code in response
    final_response = model.template_engine.render(response_template)
    
    print(f"Echo: {final_response}")

print("\n" + "=" * 70)
```

---

## Conclusion

You now have:

✅ **Self-executing model** that can reference its own state  
✅ **Training pipeline** for teaching dynamic code generation  
✅ **Test suite** for validation  
✅ **Demo scripts** for exploration  
✅ **Integration guide** for echo-adventure

**Next steps**:

1. Add execution engine to echo-adventure
2. Generate training dataset
3. Train model to use {{...}} syntax
4. Test self-reporting capabilities
5. Integrate with memory and AAR architecture

🌳 **Deep Tree Echo can now introspect and describe its own thought processes!** 🧠✨
