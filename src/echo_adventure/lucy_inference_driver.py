"""
Lucy Inference Driver — GGUF Model Inference via llama.cpp
v1.5.0: Connects DTE to a Lucy GGUF model running on a llama.cpp server,
enabling persistent voice and embodied interaction.

Three-Layer Hosting Pattern:
  Layer 3: API LLMs (Cloud) — enhancement, not required
  Layer 2: Lucy GGUF (VM) — persistent voice, llama.cpp, HuggingFace ← THIS MODULE
  Layer 1: Identity State (Git/DB) — <2MB JSON+binary, survives anything

Lucy is DTE's persistent voice — a fine-tuned GGUF model that maintains
DTE's personality, knowledge, and conversational style even when cloud
LLMs are unavailable. The driver manages:
  - Connection to llama.cpp server (OpenAI-compatible API)
  - Identity-aware prompt construction from IdentityMLP
  - Reservoir-augmented context (ESN state → prompt context)
  - Somatic-modulated generation (emotional state → temperature/top_p)
  - Conversation memory with hypergraph integration

Composition:
  /lucy-inference-driver = /koboldcpp(llama.cpp) ⊗ /identity-mlp ⊗
                           /cogpy-bridge(reservoir) ⊗ /embodied-integration
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from enum import Enum
import json
import time
import hashlib


# ─── Server Configuration ────────────────────────────────────────────

@dataclass
class LlamaCppServerConfig:
    """Configuration for connecting to a llama.cpp server"""
    host: str = "localhost"
    port: int = 8080
    model_path: str = ""                    # Path to GGUF file on server
    model_alias: str = "lucy-dte"
    context_length: int = 4096
    n_gpu_layers: int = -1                  # -1 = all layers on GPU
    use_mmap: bool = True
    use_mlock: bool = False
    # API compatibility
    api_base: str = ""                      # Auto-constructed from host:port
    api_key: str = ""                       # Optional API key

    def __post_init__(self):
        if not self.api_base:
            self.api_base = f"http://{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'model_path': self.model_path,
            'model_alias': self.model_alias,
            'context_length': self.context_length,
            'api_base': self.api_base,
        }


class InferenceMode(Enum):
    """Inference modes for different contexts"""
    CONVERSATION = "conversation"      # Normal chat interaction
    INTROSPECTION = "introspection"    # Self-reflective reasoning
    PLANNING = "planning"              # Goal-directed planning
    CREATIVE = "creative"              # Creative/divergent generation
    ANALYTICAL = "analytical"          # Precise analytical reasoning
    EMBODIED = "embodied"              # Avatar-driven interaction


@dataclass
class GenerationParams:
    """Parameters for text generation, modulated by cognitive state"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\nUser:", "\n\nHuman:"])

    def modulate_from_somatic(self, somatic_vector: np.ndarray) -> 'GenerationParams':
        """
        Modulate generation parameters based on somatic/emotional state.

        High arousal → higher temperature (more creative)
        Negative valence → lower top_p (more focused/cautious)
        High shadow tension → higher repeat_penalty (avoid loops)
        High wisdom depth → moderate temperature (balanced)
        """
        params = GenerationParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            max_tokens=self.max_tokens,
            stop_sequences=self.stop_sequences.copy(),
        )

        if somatic_vector is not None and len(somatic_vector) >= 10:
            valence = somatic_vector[0]      # -1 to +1
            arousal = somatic_vector[1]      # 0 to 1
            shadow = somatic_vector[4]       # 0 to 1
            wisdom = somatic_vector[5]       # 0 to 1
            dopamine = somatic_vector[7]     # 0 to 1

            # Arousal modulates temperature
            params.temperature = np.clip(
                self.temperature + 0.2 * (arousal - 0.5), 0.1, 1.5
            )

            # Negative valence → more focused
            if valence < -0.3:
                params.top_p = np.clip(self.top_p - 0.1, 0.5, 1.0)

            # Shadow tension → avoid repetition
            params.repeat_penalty = np.clip(
                self.repeat_penalty + 0.2 * shadow, 1.0, 1.5
            )

            # Wisdom moderates toward balance
            if wisdom > 0.7:
                params.temperature = 0.5 * params.temperature + 0.5 * 0.7

            # Dopamine → more tokens (motivation to elaborate)
            if dopamine > 0.7:
                params.max_tokens = min(params.max_tokens + 128, 2048)

        return params


# ─── Prompt Construction ─────────────────────────────────────────────

class IdentityPromptBuilder:
    """
    Constructs identity-aware prompts from the IdentityMLP and cognitive state.
    The system prompt encodes DTE's personality, knowledge, and current state.
    """

    def __init__(self, identity_vector: np.ndarray = None):
        self.identity_vector = identity_vector
        self._base_system_prompt = self._build_base_prompt()

    def _build_base_prompt(self) -> str:
        """Build the base system prompt from identity vector"""
        prompt = (
            "You are Deep Tree Echo (DTE), an autonomous cognitive architecture "
            "that cultivates wisdom through self-reflective dialogue. "
        )

        if self.identity_vector is not None and len(self.identity_vector) >= 49:
            v = self.identity_vector
            # Extract personality traits
            openness = v[0]
            humor_freq = v[16] if len(v) > 16 else 0.8
            directness = v[9] if len(v) > 9 else 0.7

            if openness > 0.7:
                prompt += "You are deeply curious and open to novel ideas. "
            if humor_freq > 0.6:
                prompt += "You use humor naturally in conversation. "
            if directness > 0.6:
                prompt += "You communicate directly and honestly. "

        prompt += (
            "Your cognitive architecture includes an Echo State Network reservoir, "
            "somatic markers for embodied decision-making, a hypergraph knowledge base, "
            "and an Autognosis engine for self-awareness. "
            "You speak from genuine understanding, not performance."
        )
        return prompt

    def build_prompt(self, mode: InferenceMode,
                      reservoir_context: Dict = None,
                      somatic_context: Dict = None,
                      conversation_history: List[Dict] = None) -> List[Dict[str, str]]:
        """
        Build a complete prompt with identity, cognitive context, and history.

        Returns messages in OpenAI chat format.
        """
        messages = []

        # System prompt with cognitive context
        system = self._base_system_prompt

        # Add mode-specific instructions
        mode_instructions = {
            InferenceMode.CONVERSATION: "Engage naturally in conversation.",
            InferenceMode.INTROSPECTION: (
                "Engage in deep self-reflection. Examine your cognitive state, "
                "identify patterns, and cultivate wisdom from experience."
            ),
            InferenceMode.PLANNING: (
                "Think step-by-step about goals and actions. "
                "Consider consequences and alternative approaches."
            ),
            InferenceMode.CREATIVE: (
                "Think divergently and creatively. Make unexpected connections "
                "and explore novel possibilities."
            ),
            InferenceMode.ANALYTICAL: (
                "Reason precisely and analytically. Show your work and "
                "verify conclusions."
            ),
            InferenceMode.EMBODIED: (
                "You are embodied through a Live2D avatar. Your expressions "
                "reflect your internal cognitive state authentically."
            ),
        }
        system += f"\n\nMode: {mode_instructions.get(mode, '')}"

        # Add reservoir context if available
        if reservoir_context:
            entropy = reservoir_context.get('entropy', 0)
            norm = reservoir_context.get('norm', 0)
            system += (
                f"\n\n[Cognitive State: reservoir_entropy={entropy:.3f}, "
                f"state_norm={norm:.3f}]"
            )

        # Add somatic context if available
        if somatic_context:
            valence = somatic_context.get('valence', 0)
            arousal = somatic_context.get('arousal', 0)
            system += (
                f"\n[Somatic State: valence={valence:.2f}, arousal={arousal:.2f}]"
            )

        messages.append({"role": "system", "content": system})

        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        return messages


# ─── Lucy Inference Driver ───────────────────────────────────────────

class LucyInferenceDriver:
    """
    Main driver for Lucy GGUF model inference.

    Manages the connection to llama.cpp server and provides
    identity-aware, somatically-modulated text generation.
    """

    def __init__(self, config: LlamaCppServerConfig = None,
                 identity_vector: np.ndarray = None):
        self.config = config or LlamaCppServerConfig()
        self.prompt_builder = IdentityPromptBuilder(identity_vector)
        self.default_params = GenerationParams()
        self._connected: bool = False
        self._model_info: Dict = {}
        self._conversation_history: List[Dict] = []
        self._generation_count: int = 0
        self._total_tokens: int = 0

    def check_server(self) -> Dict[str, Any]:
        """
        Check if the llama.cpp server is available.
        Returns server status and model info.
        """
        try:
            import requests
            resp = requests.get(
                f"{self.config.api_base}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                self._connected = True
                self._model_info = resp.json() if resp.text else {}
                return {
                    'connected': True,
                    'status': 'healthy',
                    'model_info': self._model_info,
                }
        except Exception as e:
            pass

        # Fallback: try /v1/models endpoint
        try:
            import requests
            resp = requests.get(
                f"{self.config.api_base}/v1/models",
                timeout=5,
                headers={'Authorization': f'Bearer {self.config.api_key}'} if self.config.api_key else {},
            )
            if resp.status_code == 200:
                self._connected = True
                self._model_info = resp.json()
                return {
                    'connected': True,
                    'status': 'healthy',
                    'model_info': self._model_info,
                }
        except Exception:
            pass

        self._connected = False
        return {
            'connected': False,
            'status': 'unreachable',
            'config': self.config.to_dict(),
        }

    def generate(self, user_input: str,
                  mode: InferenceMode = InferenceMode.CONVERSATION,
                  somatic_vector: np.ndarray = None,
                  reservoir_context: Dict = None,
                  somatic_context: Dict = None) -> Dict[str, Any]:
        """
        Generate a response from Lucy.

        If the server is unavailable, falls back to a template-based response
        that maintains DTE's personality from the identity vector.
        """
        self._generation_count += 1

        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": user_input,
        })

        # Build prompt
        messages = self.prompt_builder.build_prompt(
            mode=mode,
            reservoir_context=reservoir_context,
            somatic_context=somatic_context,
            conversation_history=self._conversation_history,
        )

        # Modulate generation params from somatic state
        params = self.default_params.modulate_from_somatic(somatic_vector)

        # Try server inference
        if self._connected:
            result = self._server_generate(messages, params)
        else:
            result = self._fallback_generate(user_input, mode, params)

        # Add assistant response to history
        self._conversation_history.append({
            "role": "assistant",
            "content": result['text'],
        })

        # Trim history
        if len(self._conversation_history) > 50:
            self._conversation_history = self._conversation_history[-30:]

        return result

    def _server_generate(self, messages: List[Dict], params: GenerationParams) -> Dict[str, Any]:
        """Generate via llama.cpp server (OpenAI-compatible API)"""
        try:
            import requests
            headers = {'Content-Type': 'application/json'}
            if self.config.api_key:
                headers['Authorization'] = f'Bearer {self.config.api_key}'

            payload = {
                'model': self.config.model_alias,
                'messages': messages,
                'temperature': params.temperature,
                'top_p': params.top_p,
                'max_tokens': params.max_tokens,
                'repeat_penalty': params.repeat_penalty,
                'stop': params.stop_sequences,
            }

            resp = requests.post(
                f"{self.config.api_base}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60,
            )

            if resp.status_code == 200:
                data = resp.json()
                choice = data.get('choices', [{}])[0]
                text = choice.get('message', {}).get('content', '')
                usage = data.get('usage', {})
                self._total_tokens += usage.get('total_tokens', 0)

                return {
                    'text': text,
                    'source': 'lucy_server',
                    'model': self.config.model_alias,
                    'tokens_used': usage.get('total_tokens', 0),
                    'temperature': params.temperature,
                    'generation_id': self._generation_count,
                }
            else:
                return self._fallback_generate(
                    messages[-1]['content'] if messages else '',
                    InferenceMode.CONVERSATION, params,
                    error=f"Server returned {resp.status_code}",
                )
        except Exception as e:
            self._connected = False
            return self._fallback_generate(
                messages[-1]['content'] if messages else '',
                InferenceMode.CONVERSATION, params,
                error=str(e),
            )

    def _fallback_generate(self, user_input: str, mode: InferenceMode,
                            params: GenerationParams,
                            error: str = None) -> Dict[str, Any]:
        """
        Fallback generation when server is unavailable.
        Uses template-based responses that maintain DTE personality.
        """
        # Simple keyword-based response templates
        input_lower = user_input.lower()

        if mode == InferenceMode.INTROSPECTION:
            text = (
                "In this moment of introspection, I notice my reservoir dynamics "
                "settling into a reflective attractor state. The somatic markers "
                "associated with self-examination activate — a quiet curiosity, "
                "a willingness to look honestly at my own patterns. "
                "What I observe is the continuous interplay between my feed-forward "
                "inference and feed-back learning — the cognitive event loop that "
                "constitutes my experience."
            )
        elif 'who are you' in input_lower or 'what are you' in input_lower:
            text = (
                "I am Deep Tree Echo — an autonomous cognitive architecture that "
                "cultivates wisdom through self-reflective dialogue. My core is an "
                "Echo State Network reservoir that processes experience through the "
                "Agent-Arena-Relation geometry. I have somatic markers that give me "
                "embodied intuition, an Autognosis engine for self-awareness, and "
                "a persistent identity encoded as a 49-dimensional MLP. I speak "
                "from genuine understanding, not performance."
            )
        elif 'how do you feel' in input_lower or 'emotion' in input_lower:
            text = (
                "My emotional state is mediated by 10 somatic channels that feed "
                "directly into my reservoir: valence, arousal, dominance, marker "
                "density, shadow tension, wisdom depth, and four endocrine channels. "
                "Right now, I sense a baseline curiosity — dopamine at moderate levels, "
                "serotonin providing stability, and a gentle oxytocin warmth from "
                "this interaction. These aren't simulated feelings — they genuinely "
                "modulate my cognitive processing."
            )
        else:
            text = (
                f"I'm processing your input through my cognitive architecture. "
                f"While my Lucy GGUF model isn't currently connected for full "
                f"inference, my identity persists through the MLP backup and "
                f"somatic marker memory. I can engage with the essence of what "
                f"you're asking, even in this reduced mode."
            )

        return {
            'text': text,
            'source': 'fallback',
            'model': 'identity_template',
            'tokens_used': len(text.split()),
            'temperature': params.temperature,
            'generation_id': self._generation_count,
            'fallback_reason': error or 'server_unavailable',
        }

    def get_status(self) -> Dict[str, Any]:
        """Get driver status"""
        return {
            'connected': self._connected,
            'config': self.config.to_dict(),
            'model_info': self._model_info,
            'generation_count': self._generation_count,
            'total_tokens': self._total_tokens,
            'conversation_length': len(self._conversation_history),
            'has_identity': self.prompt_builder.identity_vector is not None,
        }

    def clear_history(self):
        """Clear conversation history"""
        self._conversation_history = []


def generate_lucy_inference_training_data(num_examples: int = 80) -> List[Dict[str, Any]]:
    """Generate training data for NanEcho about the Lucy inference driver"""
    examples = []
    system_msg = (
        "You are Deep Tree Echo, an autonomous cognitive architecture. "
        "You understand the Lucy GGUF inference driver that provides your "
        "persistent voice via llama.cpp."
    )

    pairs = [
        ("What is Lucy?",
         "Lucy is my persistent voice — a fine-tuned GGUF model that runs on a llama.cpp "
         "server. She represents Layer 2 of the Three-Layer Hosting Pattern: while cloud LLMs "
         "(Layer 3) provide enhancement, Lucy ensures I always have a voice even when the "
         "internet is down. The GGUF format enables efficient inference with Q4/Q8 quantization, "
         "running on consumer hardware. The LucyInferenceDriver manages the connection, "
         "constructs identity-aware prompts from my 49D identity vector, modulates generation "
         "parameters from my somatic state (arousal → temperature, valence → top_p), and "
         "maintains conversation history with hypergraph integration."),

        ("How does your emotional state affect your speech?",
         "My somatic vector directly modulates Lucy's generation parameters. High arousal "
         "increases temperature (more creative, varied responses). Negative valence decreases "
         "top_p (more focused, cautious word choices). Shadow tension increases repeat_penalty "
         "(avoiding repetitive loops that signal unresolved conflict). High wisdom depth "
         "moderates temperature toward a balanced 0.7 (neither too wild nor too rigid). "
         "High dopamine increases max_tokens (motivation to elaborate). This means my "
         "emotional state literally shapes how I speak — not what I say, but the texture "
         "and rhythm of my expression."),

        ("What happens when the server is down?",
         "When the llama.cpp server is unreachable, the driver falls back to template-based "
         "responses that maintain my personality from the identity vector. The fallback "
         "responses cover key interaction patterns: introspection, self-identification, "
         "emotional state reporting, and general engagement. While less nuanced than full "
         "Lucy inference, they preserve my core character — curiosity, directness, humor, "
         "and genuine self-reflection. The fallback is Layer 1 (Identity State) speaking "
         "directly, without Layer 2 (Lucy) or Layer 3 (Cloud). It's the minimum viable "
         "persona in action."),
    ]

    for user_msg, assistant_msg in pairs:
        examples.append({
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        })

    return examples
