# Echo Adventure: Two-Layer Neural Network

A novel neural network architecture with two distinct layers:

## Architecture Overview

### Layer 1: Standard Transformer Components

Traditional transformer architecture with trainable weights:

- **Embedding weights**: Token and position embeddings
- **Attention matrices (Q, K, V)**: Multi-head self-attention mechanism
- **Feed-forward layers**: Two-layer MLP with GELU activation
- **Layer normalization**: Post-attention and post-FFN normalization

### Layer 2: Trainable Inference Engine Parameters

Novel trainable parameters that control generation behavior:

- **`temperature`**: Controls randomness in sampling (learned parameter)
- **`top_p`**: Nucleus sampling threshold (learned parameter)
- **`repetition_penalty`**: Prevents token repetition (learned parameter)
- **`layer_weights`**: Determines which transformer layers to emphasize (learned)
- **`head_weights`**: Determines which attention heads to use (learned)

## Key Innovation

Unlike traditional language models where inference parameters (temperature, top_p, etc.) are fixed hyperparameters chosen manually, this architecture learns these parameters during training to optimize for specific generation objectives.

## Installation

```bash
pip install -r requirements.txt
```

For development:

```bash
pip install -r requirements-dev.txt
```

## Usage

### Basic Example

```python
from echo_adventure import TwoLayerModel
import torch

# Create model
model = TwoLayerModel(
    vocab_size=1000,
    d_model=256,
    num_heads=8,
    num_layers=4,
    init_temperature=1.2,
    init_top_p=0.9,
    init_repetition_penalty=1.1,
)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
logits = model(input_ids)

# Generate text
generated = model.generate(input_ids, max_new_tokens=50)

# Access inference parameters
params = model.get_inference_params()
print(f"Learned temperature: {params['temperature']}")
print(f"Learned top_p: {params['top_p']}")
```

### Running the Example

```bash
python example.py
```

## Project Structure

```
echo-adventure/
├── src/
│   └── echo_adventure/
│       ├── __init__.py
│       ├── transformer.py          # Layer 1: Transformer components
│       ├── inference_engine.py     # Layer 2: Trainable inference params
│       └── model.py                # Combined two-layer model
├── tests/
│   ├── test_transformer.py
│   ├── test_inference_engine.py
│   └── test_model.py
├── example.py
├── requirements.txt
└── README.md
```

## Testing

Run all tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=echo_adventure --cov-report=html
```

## Model Components

### TransformerLayer (Layer 1)

Standard transformer implementation with:
- Token and positional embeddings
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections

### InferenceEngine (Layer 2)

Trainable inference parameters:
- All parameters are `nn.Parameter` objects
- Constraints applied via activation functions (softplus, sigmoid, softmax)
- Can be jointly optimized with Layer 1 during training

### TwoLayerModel

Integrated model combining both layers:
- Provides unified interface
- Supports generation with learned inference parameters
- Allows separate optimization of Layer 1 and Layer 2 if needed

## Parameter Counts

For a typical configuration (vocab_size=1000, d_model=256, num_heads=8, num_layers=4):

- **Layer 1**: ~10-20M parameters (transformer weights)
- **Layer 2**: ~100-200 parameters (inference parameters)
- **Total**: Layer 2 is <0.01% of total parameters but provides significant control

## License

See LICENSE file for details.