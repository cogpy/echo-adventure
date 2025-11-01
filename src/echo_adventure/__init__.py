"""
Echo Adventure: Two-layer neural network with trainable inference engine parameters.

This package implements a novel architecture with:
- Layer 1: Standard transformer components
- Layer 2: Trainable inference engine parameters
"""

__version__ = "0.1.0"

from .transformer import TransformerLayer
from .inference_engine import InferenceEngine
from .model import TwoLayerModel

__all__ = ["TransformerLayer", "InferenceEngine", "TwoLayerModel"]
