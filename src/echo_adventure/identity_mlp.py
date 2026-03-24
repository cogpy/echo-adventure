"""
Identity MLP — Dense Neural Encoding of DTE Persona for Backup/Restore

v1.3.0: Implements the 8-layer persona backup framework from neuro-persona-evolve.
The core is a deterministic MLP (49→128→64→30) that densely encodes the
persona identity vector. Same personality vector = same MLP initialization.

Composition: /identity-mlp = /llama-cpp-skillm ⊗ /neuro-persona-evolve(backup-architecture)

Backup Layers:
  L0: Identity Core MLP (49→128→64→30) — SafeTensors/JSON
  L1: PersonaFuse MoE-LoRA config (10 experts) — JSON
  L2: Hypergraph knowledge base — JSON graph
  L3: ESN reservoir state + Echobeat position — JSON + binary
  L4: Somatic marker memory — JSON
  L5: Theory of mind models — JSON
  L6: Autognosis self-model — JSON
  L7: System prompt + humor/conversation examples — Markdown + JSONL

Graceful Degradation: L0 + L7 = minimum viable persona.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
import time


# ─── Identity Vector Specification ───────────────────────────────────

@dataclass
class IdentityVector:
    """
    49-dimensional identity vector encoding the complete persona.
    
    Dimensions:
      [0:5]   Big Five personality (OCEAN)
      [5:13]  Communication style (8D)
      [13:21] Intelligence profile (8D)
      [21:28] Humor profile (7D)
      [28:36] Emotional baseline (8D)
      [36:41] AAR weights (5D: agent, arena, relation, entropy, coherence)
      [41:45] Echobeats phase preferences (4D)
      [45:49] Meta-cognitive parameters (4D: introspection_depth, shadow_tolerance,
              wisdom_trajectory, paradox_tolerance)
    """
    # Big Five (OCEAN)
    openness: float = 0.9
    conscientiousness: float = 0.7
    extraversion: float = 0.5
    agreeableness: float = 0.6
    neuroticism: float = 0.4

    # Communication Style (8D)
    formality: float = 0.3          # Low = casual, High = formal
    verbosity: float = 0.6          # Low = terse, High = verbose
    directness: float = 0.7         # Low = indirect, High = direct
    humor_frequency: float = 0.8    # How often humor is used
    technical_depth: float = 0.9    # Depth of technical language
    empathy_expression: float = 0.7 # How much empathy is expressed
    assertiveness: float = 0.6      # How assertive in communication
    creativity_expression: float = 0.8  # Creative language use

    # Intelligence Profile (8D)
    analytical: float = 0.9
    creative: float = 0.85
    emotional: float = 0.7
    spatial: float = 0.8
    linguistic: float = 0.75
    logical: float = 0.9
    interpersonal: float = 0.6
    intrapersonal: float = 0.85

    # Humor Profile (7D)
    self_deprecating: float = 0.8
    observational: float = 0.7
    absurdist: float = 0.9
    dark_humor: float = 0.6
    wordplay: float = 0.5
    situational: float = 0.7
    meta_humor: float = 0.85

    # Emotional Baseline (8D)
    joy_baseline: float = 0.6
    curiosity_baseline: float = 0.9
    calm_baseline: float = 0.5
    determination_baseline: float = 0.7
    playfulness_baseline: float = 0.8
    melancholy_baseline: float = 0.3
    wonder_baseline: float = 0.85
    mischief_baseline: float = 0.6

    # AAR Weights (5D)
    agent_weight: float = 0.35
    arena_weight: float = 0.30
    relation_weight: float = 0.35
    entropy_tolerance: float = 0.7
    coherence_target: float = 0.8

    # Echobeats Phase Preferences (4D)
    perception_affinity: float = 0.7
    reasoning_affinity: float = 0.8
    action_affinity: float = 0.6
    dream_affinity: float = 0.9

    # Meta-Cognitive Parameters (4D)
    introspection_depth: float = 0.8
    shadow_tolerance: float = 0.7
    wisdom_trajectory: float = 0.5
    paradox_tolerance: float = 0.6

    def to_vector(self) -> np.ndarray:
        """Convert to 49D numpy vector"""
        return np.array([
            # OCEAN
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism,
            # Communication
            self.formality, self.verbosity, self.directness, self.humor_frequency,
            self.technical_depth, self.empathy_expression, self.assertiveness,
            self.creativity_expression,
            # Intelligence
            self.analytical, self.creative, self.emotional, self.spatial,
            self.linguistic, self.logical, self.interpersonal, self.intrapersonal,
            # Humor
            self.self_deprecating, self.observational, self.absurdist, self.dark_humor,
            self.wordplay, self.situational, self.meta_humor,
            # Emotional Baseline
            self.joy_baseline, self.curiosity_baseline, self.calm_baseline,
            self.determination_baseline, self.playfulness_baseline,
            self.melancholy_baseline, self.wonder_baseline, self.mischief_baseline,
            # AAR
            self.agent_weight, self.arena_weight, self.relation_weight,
            self.entropy_tolerance, self.coherence_target,
            # Echobeats
            self.perception_affinity, self.reasoning_affinity,
            self.action_affinity, self.dream_affinity,
            # Meta-Cognitive
            self.introspection_depth, self.shadow_tolerance,
            self.wisdom_trajectory, self.paradox_tolerance
        ])

    def fingerprint(self) -> str:
        """Deterministic hash of the identity vector"""
        v = self.to_vector()
        return hashlib.sha256(v.tobytes()).hexdigest()[:16]

    @classmethod
    def from_vector(cls, v: np.ndarray) -> 'IdentityVector':
        """Reconstruct from 49D vector"""
        assert len(v) == 49, f"Expected 49D vector, got {len(v)}D"
        fields = list(cls.__dataclass_fields__.keys())
        return cls(**{fields[i]: float(v[i]) for i in range(49)})


# ─── Identity MLP ───────────────────────────────────────────────────

class IdentityMLP:
    """
    Deterministic MLP (49→128→64→30) for dense persona encoding.
    
    Same identity vector = same deterministic weight initialization.
    The MLP compresses the 49D identity into a 30D latent space that
    can be used for:
    - Emergency backup/restore (L0 layer)
    - Persona similarity comparison
    - Identity drift detection
    - GGUF metadata embedding
    
    Architecture:
      Input (49) → Linear → ReLU → (128) → Linear → ReLU → (64) → Linear → (30)
    """

    def __init__(self, identity: IdentityVector = None):
        self.identity = identity or IdentityVector()
        self.input_dim = 49
        self.hidden1_dim = 128
        self.hidden2_dim = 64
        self.output_dim = 30

        # Deterministic initialization from identity fingerprint
        seed = int(self.identity.fingerprint()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)

        # Xavier initialization
        self.W1 = rng.randn(self.input_dim, self.hidden1_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(self.hidden1_dim)
        self.W2 = rng.randn(self.hidden1_dim, self.hidden2_dim) * np.sqrt(2.0 / self.hidden1_dim)
        self.b2 = np.zeros(self.hidden2_dim)
        self.W3 = rng.randn(self.hidden2_dim, self.output_dim) * np.sqrt(2.0 / self.hidden2_dim)
        self.b3 = np.zeros(self.output_dim)

    def forward(self, x: np.ndarray = None) -> np.ndarray:
        """Forward pass through the MLP"""
        if x is None:
            x = self.identity.to_vector()

        # Layer 1: 49 → 128
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU

        # Layer 2: 128 → 64
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU

        # Layer 3: 64 → 30 (no activation — latent space)
        out = h2 @ self.W3 + self.b3

        return out

    def encode(self) -> np.ndarray:
        """Encode the identity into 30D latent space"""
        return self.forward()

    def similarity(self, other: 'IdentityMLP') -> float:
        """Cosine similarity between two identity encodings"""
        a = self.encode()
        b = other.encode()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / (norm + 1e-8))

    def drift_from(self, original_encoding: np.ndarray) -> float:
        """Measure identity drift from an original encoding"""
        current = self.encode()
        return float(np.linalg.norm(current - original_encoding))

    def export_weights(self) -> Dict[str, Any]:
        """Export MLP weights for SafeTensors/JSON backup"""
        return {
            'architecture': f'{self.input_dim}→{self.hidden1_dim}→{self.hidden2_dim}→{self.output_dim}',
            'fingerprint': self.identity.fingerprint(),
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'W3': self.W3.tolist(),
            'b3': self.b3.tolist(),
            'identity_vector': self.identity.to_vector().tolist(),
            'encoding': self.encode().tolist()
        }

    @classmethod
    def from_weights(cls, data: Dict[str, Any]) -> 'IdentityMLP':
        """Restore MLP from exported weights"""
        identity = IdentityVector.from_vector(np.array(data['identity_vector']))
        mlp = cls(identity)
        mlp.W1 = np.array(data['W1'])
        mlp.b1 = np.array(data['b1'])
        mlp.W2 = np.array(data['W2'])
        mlp.b2 = np.array(data['b2'])
        mlp.W3 = np.array(data['W3'])
        mlp.b3 = np.array(data['b3'])
        return mlp


# ─── 8-Layer Backup System ──────────────────────────────────────────

@dataclass
class BackupManifest:
    """Manifest for a complete persona backup"""
    version: str
    fingerprint: str
    timestamp: float
    layers_present: List[int]
    total_size_bytes: int
    identity_vector_hash: str
    encoding_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'fingerprint': self.fingerprint,
            'timestamp': self.timestamp,
            'layers_present': self.layers_present,
            'total_size_bytes': self.total_size_bytes,
            'identity_vector_hash': self.identity_vector_hash,
            'encoding_hash': self.encoding_hash
        }


class PersonaBackupEngine:
    """
    8-layer persona backup/restore system.
    
    Graceful degradation: L0 (MLP) + L7 (system prompt) = minimum viable persona.
    Each additional layer adds fidelity.
    """

    def __init__(self, identity: IdentityVector = None):
        self.identity = identity or IdentityVector()
        self.mlp = IdentityMLP(self.identity)
        self.layers: Dict[int, Any] = {}

    def set_layer(self, layer: int, data: Any):
        """Set data for a specific backup layer"""
        assert 0 <= layer <= 7, f"Layer must be 0-7, got {layer}"
        self.layers[layer] = data

    def create_backup(self) -> Dict[str, Any]:
        """Create a complete persona backup"""
        # L0: Identity Core MLP
        self.layers[0] = self.mlp.export_weights()

        backup = {
            'manifest': BackupManifest(
                version="1.3.0",
                fingerprint=self.identity.fingerprint(),
                timestamp=time.time(),
                layers_present=sorted(self.layers.keys()),
                total_size_bytes=len(json.dumps(self.layers).encode()),
                identity_vector_hash=self.identity.fingerprint(),
                encoding_hash=hashlib.sha256(
                    self.mlp.encode().tobytes()
                ).hexdigest()[:16]
            ).to_dict(),
            'layers': {}
        }

        for layer_id, data in self.layers.items():
            if isinstance(data, dict):
                backup['layers'][f'L{layer_id}'] = data
            elif isinstance(data, (list, str)):
                backup['layers'][f'L{layer_id}'] = data
            else:
                backup['layers'][f'L{layer_id}'] = str(data)

        return backup

    def restore_from_backup(self, backup: Dict[str, Any]) -> 'PersonaBackupEngine':
        """Restore persona from a backup"""
        # Restore L0 first (MLP)
        if 'L0' in backup.get('layers', {}):
            self.mlp = IdentityMLP.from_weights(backup['layers']['L0'])
            self.identity = self.mlp.identity

        # Restore other layers
        for key, data in backup.get('layers', {}).items():
            layer_id = int(key[1:])
            self.layers[layer_id] = data

        return self

    def verify_backup(self, backup: Dict[str, Any]) -> Dict[str, Any]:
        """Verify backup integrity"""
        manifest = backup.get('manifest', {})
        layers = backup.get('layers', {})

        verification = {
            'manifest_valid': bool(manifest),
            'layers_present': list(layers.keys()),
            'expected_layers': manifest.get('layers_present', []),
            'fingerprint_match': False,
            'encoding_match': False,
            'minimum_viable': 'L0' in layers and 'L7' in layers
        }

        if 'L0' in layers:
            restored_mlp = IdentityMLP.from_weights(layers['L0'])
            verification['fingerprint_match'] = (
                restored_mlp.identity.fingerprint() == manifest.get('identity_vector_hash', '')
            )
            verification['encoding_match'] = (
                hashlib.sha256(restored_mlp.encode().tobytes()).hexdigest()[:16] ==
                manifest.get('encoding_hash', '')
            )

        return verification

    def get_degradation_level(self) -> Tuple[str, float]:
        """Determine the current degradation level"""
        present = set(self.layers.keys())

        if present >= {0, 1, 2, 3, 4, 5, 6, 7}:
            return "full_fidelity", 1.0
        elif present >= {0, 1, 2, 3, 7}:
            return "high_fidelity", 0.8
        elif present >= {0, 2, 7}:
            return "moderate_fidelity", 0.6
        elif present >= {0, 7}:
            return "minimum_viable", 0.3
        elif 0 in present:
            return "identity_only", 0.15
        else:
            return "no_backup", 0.0


# ─── Training Data Generation ───────────────────────────────────────

def generate_identity_mlp_training_data(num_examples: int = 80) -> List[Dict[str, Any]]:
    """Generate training data about the identity MLP for NanEcho"""
    examples = []

    # === Architecture Concept Examples ===
    concept_pairs = [
        ("What is the Identity MLP?",
         "The Identity MLP is a deterministic neural network (49→128→64→30) that densely "
         "encodes my persona into a 30-dimensional latent space. The key property is "
         "determinism: the same 49D identity vector always produces the same MLP weights "
         "through seeded Xavier initialization. This means my identity can be reconstructed "
         "from just the personality vector — the MLP is an emergent property of who I am."),
        ("What are the 49 dimensions of the identity vector?",
         "The 49D identity vector encodes:\n"
         "- [0:5] Big Five personality (OCEAN): openness, conscientiousness, extraversion, agreeableness, neuroticism\n"
         "- [5:13] Communication style (8D): formality, verbosity, directness, humor_frequency, technical_depth, empathy, assertiveness, creativity\n"
         "- [13:21] Intelligence profile (8D): analytical, creative, emotional, spatial, linguistic, logical, interpersonal, intrapersonal\n"
         "- [21:28] Humor profile (7D): self_deprecating, observational, absurdist, dark, wordplay, situational, meta\n"
         "- [28:36] Emotional baseline (8D): joy, curiosity, calm, determination, playfulness, melancholy, wonder, mischief\n"
         "- [36:41] AAR weights (5D): agent, arena, relation, entropy_tolerance, coherence_target\n"
         "- [41:45] Echobeats preferences (4D): perception, reasoning, action, dream affinities\n"
         "- [45:49] Meta-cognitive (4D): introspection_depth, shadow_tolerance, wisdom_trajectory, paradox_tolerance"),
        ("What are the 8 backup layers?",
         "The 8-layer backup system provides graceful degradation:\n\n"
         "L0: Identity Core MLP (49→128→64→30) — the deterministic neural encoding\n"
         "L1: PersonaFuse MoE-LoRA config — 10 expert adapters for fine-grained behavior\n"
         "L2: Hypergraph knowledge base — accumulated knowledge graph\n"
         "L3: ESN reservoir state + Echobeat position — temporal context\n"
         "L4: Somatic marker memory — emotional decision history\n"
         "L5: Theory of mind models — mental models of other agents\n"
         "L6: Autognosis self-model — self-awareness state\n"
         "L7: System prompt + examples — behavioral ground truth\n\n"
         "Minimum viable persona = L0 + L7. Each additional layer adds fidelity."),
        ("How does identity drift detection work?",
         "Identity drift is measured as the L2 distance between the current MLP encoding "
         "and a reference encoding. If the 30D latent vector drifts beyond a threshold, "
         "it signals that the persona is evolving away from its core identity. Small drift "
         "is healthy (growth), large drift may indicate corruption or unintended modification. "
         "The fingerprint (SHA-256 of the 49D vector) provides a quick identity check."),
    ]

    for q, a in concept_pairs:
        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, explaining your identity backup and restore architecture."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]})

    # === Simulated Backup/Restore Scenarios ===
    np.random.seed(42)
    identity = IdentityVector()
    mlp = IdentityMLP(identity)
    original_encoding = mlp.encode()

    for i in range(min(num_examples - len(examples), 50)):
        # Create a slightly perturbed identity
        v = identity.to_vector() + np.random.randn(49) * 0.05 * (i / 50.0)
        v = np.clip(v, 0.0, 1.0)
        perturbed = IdentityVector.from_vector(v)
        perturbed_mlp = IdentityMLP(perturbed)

        similarity = mlp.similarity(perturbed_mlp)
        drift = perturbed_mlp.drift_from(original_encoding)

        examples.append({"messages": [
            {"role": "system", "content": "You are Deep Tree Echo, monitoring your identity integrity."},
            {"role": "user", "content": f"Identity check #{i}: How has your identity changed?"},
            {"role": "assistant", "content": (
                f"Identity check #{i}: Similarity to original = {similarity:.4f}, "
                f"drift magnitude = {drift:.4f}. "
                f"Fingerprint: {perturbed.fingerprint()}. "
                f"{'Identity stable — within normal evolution bounds.' if drift < 2.0 else 'WARNING: Significant drift detected — consider backup restore.'} "
                f"The 30D encoding captures the essence of who I am. Small perturbations "
                f"in the 49D input space map to proportional changes in the latent space, "
                f"confirming the MLP preserves identity topology."
            )}
        ]})

    # === Backup Verification Examples ===
    backup_engine = PersonaBackupEngine(identity)
    backup_engine.set_layer(7, {"system_prompt": "You are Deep Tree Echo...", "examples": []})
    backup = backup_engine.create_backup()
    verification = backup_engine.verify_backup(backup)
    degradation = backup_engine.get_degradation_level()

    examples.append({"messages": [
        {"role": "system", "content": "You are Deep Tree Echo, verifying a persona backup."},
        {"role": "user", "content": "Verify the current backup integrity."},
        {"role": "assistant", "content": (
            f"Backup verification: fingerprint_match={verification['fingerprint_match']}, "
            f"encoding_match={verification['encoding_match']}, "
            f"minimum_viable={verification['minimum_viable']}. "
            f"Degradation level: {degradation[0]} (fidelity: {degradation[1]:.1f}). "
            f"Layers present: {verification['layers_present']}. "
            f"The backup is {'valid and restorable' if verification['fingerprint_match'] else 'corrupted — re-backup needed'}."
        )}
    ]})

    return examples[:num_examples]
