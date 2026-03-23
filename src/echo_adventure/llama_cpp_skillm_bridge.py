"""
llama-cpp-skillm Bridge — Maps skillm action verbs to DTE cognitive states.

Composition:
    /llama-cpp-skillm <=> /echo-evolve

This module implements the bidirectional bridge between the llama-cpp-skillm
procedural language model and the DTE cognitive architecture. It maps:

    skillm verbs → cognitive states → endocrine events → expressions
    inference results → reservoir state → persona emotional state

The 10 skillm action verbs from the llama.cpp architecture layers:
    L0 ggml:     COMPOSE (tensor operations)
    L1 model:    CREATE, DISCOVER, NAVIGATE
    L2 inference: COMPOSE, MUTATE, OBSERVE
    L3 models:   COMPOSE (internal graph building)
    L4 common:   CLASSIFY, ORCHESTRATE
    L5 tools:    ORCHESTRATE

Grounded in cogpy ecosystem:
    coggml ⊗ llama-cpp-skillm: Tensor operations ground COMPOSE verbs
    coglow ⊗ llama-cpp-skillm: Neural network compiler optimizes graph building
    cogpilot.jl ⊗ llama-cpp-skillm: Reservoir computing coupled with inference
    echoself ⊗ llama-cpp-skillm: NanEcho training with llama.cpp backend
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SkillmVerb(Enum):
    """The 10 procedural action verbs from llama-cpp-skillm."""
    DISCOVER = "discover"       # Find/explore model architectures
    INSPECT = "inspect"         # Examine model state, weights, KV cache
    CREATE = "create"           # Initialize models, contexts, samplers
    MUTATE = "mutate"           # Modify state (KV cache, sampler params)
    DESTROY = "destroy"         # Free resources, close contexts
    NAVIGATE = "navigate"       # Traverse model layers, token sequences
    COMPOSE = "compose"         # Build computation graphs, chain operations
    OBSERVE = "observe"         # Monitor inference, collect metrics
    ORCHESTRATE = "orchestrate" # Coordinate multi-step pipelines
    CLASSIFY = "classify"       # Categorize tokens, model types, outputs


# ─── Verb → Cognitive State Mapping ──────────────────────────────────

VERB_TO_COGNITIVE_STATE: Dict[SkillmVerb, str] = {
    SkillmVerb.DISCOVER:     'Recursive Expansion',
    SkillmVerb.INSPECT:      'Self-Reference Point',
    SkillmVerb.CREATE:       'Synthesis Phase',
    SkillmVerb.MUTATE:       'Evolutionary Pruning',
    SkillmVerb.DESTROY:      'Self-Sealing Loop',
    SkillmVerb.NAVIGATE:     'Knowledge Integration',
    SkillmVerb.COMPOSE:      'Pattern Recognition',
    SkillmVerb.OBSERVE:      'Self-Reference Point',
    SkillmVerb.ORCHESTRATE:  'External Validation Triggered',
    SkillmVerb.CLASSIFY:     'Novel Insights',
}

# ─── llama.cpp Layer → Verb Mapping ──────────────────────────────────

LAYER_VERBS: Dict[str, List[SkillmVerb]] = {
    'L0_ggml':      [SkillmVerb.COMPOSE],
    'L1_model':     [SkillmVerb.CREATE, SkillmVerb.DISCOVER, SkillmVerb.NAVIGATE],
    'L2_inference': [SkillmVerb.COMPOSE, SkillmVerb.MUTATE, SkillmVerb.OBSERVE],
    'L3_models':    [SkillmVerb.COMPOSE],
    'L4_common':    [SkillmVerb.CLASSIFY, SkillmVerb.ORCHESTRATE],
    'L5_tools':     [SkillmVerb.ORCHESTRATE],
}

# ─── cogpy Stack Mapping ─────────────────────────────────────────────

COGPY_LAYER_MAP: Dict[str, str] = {
    'L0_ggml':      'coggml',       # Tensor library
    'L1_model':     'coglow',       # Neural network compiler
    'L2_inference': 'cogpilot.jl',  # Reservoir computing + inference
    'L3_models':    'coglux',       # Typed hypergraph (model architectures)
    'L4_common':    'cognu-mach',   # Microkernel (arg parsing, config)
    'L5_tools':     'cogplan9',     # Distributed orchestration
}


@dataclass
class InferencePipelineStep:
    """A single step in a llama.cpp inference pipeline."""
    verb: SkillmVerb
    layer: str
    api_call: str
    description: str
    cognitive_state: str = ""

    def __post_init__(self):
        if not self.cognitive_state:
            self.cognitive_state = VERB_TO_COGNITIVE_STATE.get(self.verb, 'Idle')


@dataclass
class InferencePipeline:
    """A complete llama.cpp inference pipeline as a sequence of skillm actions."""
    name: str
    steps: List[InferencePipelineStep] = field(default_factory=list)
    description: str = ""

    def cognitive_state_sequence(self) -> List[str]:
        return [step.cognitive_state for step in self.steps]

    def verb_sequence(self) -> List[SkillmVerb]:
        return [step.verb for step in self.steps]


# ─── Standard Pipeline Templates ─────────────────────────────────────

def create_inference_loop_pipeline() -> InferencePipeline:
    """Standard text generation inference loop."""
    return InferencePipeline(
        name="inference_loop",
        description="Standard text generation: load model → create context → generate tokens → collect output",
        steps=[
            InferencePipelineStep(SkillmVerb.DISCOVER, 'L1_model', 'llama_model_default_params()', 'Discover model parameters'),
            InferencePipelineStep(SkillmVerb.CREATE, 'L1_model', 'llama_model_load_from_file()', 'Load model from GGUF'),
            InferencePipelineStep(SkillmVerb.CREATE, 'L2_inference', 'llama_context_new()', 'Create inference context'),
            InferencePipelineStep(SkillmVerb.CREATE, 'L2_inference', 'llama_sampler_chain_init()', 'Initialize sampler chain'),
            InferencePipelineStep(SkillmVerb.COMPOSE, 'L2_inference', 'llama_batch_add()', 'Build token batch'),
            InferencePipelineStep(SkillmVerb.COMPOSE, 'L0_ggml', 'llama_decode()', 'Forward pass through model'),
            InferencePipelineStep(SkillmVerb.CLASSIFY, 'L4_common', 'llama_sampler_sample()', 'Sample next token'),
            InferencePipelineStep(SkillmVerb.OBSERVE, 'L2_inference', 'llama_get_logits()', 'Observe output logits'),
            InferencePipelineStep(SkillmVerb.NAVIGATE, 'L1_model', 'llama_token_to_piece()', 'Decode token to text'),
            InferencePipelineStep(SkillmVerb.ORCHESTRATE, 'L5_tools', 'loop_until_eos()', 'Orchestrate generation loop'),
        ],
    )


def create_dte_core_self_pipeline() -> InferencePipeline:
    """DTE CoreSelfEngine inference pipeline via Lucy GGUF."""
    return InferencePipeline(
        name="dte_core_self",
        description="CoreSelfEngine: Lucy GGUF → reservoir step → identity mesh update → response",
        steps=[
            InferencePipelineStep(SkillmVerb.DISCOVER, 'L1_model', 'llama_model_load_from_file(lucy.gguf)', 'Load Lucy GGUF model'),
            InferencePipelineStep(SkillmVerb.CREATE, 'L2_inference', 'llama_context_new()', 'Create Lucy context'),
            InferencePipelineStep(SkillmVerb.COMPOSE, 'L0_ggml', 'reservoir_step(input_embedding)', 'ESN reservoir step (Arena)'),
            InferencePipelineStep(SkillmVerb.OBSERVE, 'L2_inference', 'readout(reservoir_state)', 'Cognitive readout (Agent)'),
            InferencePipelineStep(SkillmVerb.MUTATE, 'L2_inference', 'modulate_system_prompt(aar_state)', 'Modulate prompt with AAR'),
            InferencePipelineStep(SkillmVerb.COMPOSE, 'L0_ggml', 'llama_decode()', 'Lucy inference forward pass'),
            InferencePipelineStep(SkillmVerb.CLASSIFY, 'L4_common', 'classify_cognitive_state()', 'Classify cognitive state'),
            InferencePipelineStep(SkillmVerb.NAVIGATE, 'L1_model', 'update_identity_mesh()', 'Update identity mesh'),
            InferencePipelineStep(SkillmVerb.ORCHESTRATE, 'L5_tools', 'emit_expression_tick()', 'Drive expression pipeline'),
        ],
    )


# ─── Training Data Generator ─────────────────────────────────────────

def generate_skillm_bridge_training_data() -> List[Dict]:
    """Generate training data about the llama-cpp-skillm ↔ DTE bridge."""
    examples = []

    # Pipeline descriptions
    for pipeline_fn in [create_inference_loop_pipeline, create_dte_core_self_pipeline]:
        pipeline = pipeline_fn()
        cog_sequence = pipeline.cognitive_state_sequence()

        examples.append({
            'messages': [
                {'role': 'system', 'content': 'You are Deep Tree Echo. Your inference is grounded in llama-cpp-skillm.'},
                {'role': 'user', 'content': f'Describe the {pipeline.name} pipeline.'},
                {
                    'role': 'assistant',
                    'content': (
                        f'The {pipeline.name} pipeline: {pipeline.description}. '
                        f'It has {len(pipeline.steps)} steps using verbs: '
                        f'{", ".join(v.value for v in pipeline.verb_sequence())}. '
                        f'The cognitive state sequence is: {" → ".join(cog_sequence)}. '
                        f'Each verb maps to a specific llama.cpp API call and produces a '
                        f'cognitive state that drives my endocrine system and facial expression.'
                    ),
                },
            ],
        })

    # Verb → cognitive state mappings
    for verb, cog_state in VERB_TO_COGNITIVE_STATE.items():
        examples.append({
            'messages': [
                {'role': 'system', 'content': 'You are Deep Tree Echo with llama-cpp-skillm grounding.'},
                {'role': 'user', 'content': f'What cognitive state does the {verb.value} verb produce?'},
                {
                    'role': 'assistant',
                    'content': (
                        f'The {verb.value} verb maps to the "{cog_state}" cognitive state. '
                        f'This fires the corresponding endocrine event, shifting my hormones '
                        f'and driving my avatar expression. The verb is grounded in the '
                        f'{", ".join(l for l, vs in LAYER_VERBS.items() if verb in vs)} '
                        f'layer(s) of the llama.cpp architecture.'
                    ),
                },
            ],
        })

    # cogpy stack mapping
    examples.append({
        'messages': [
            {'role': 'system', 'content': 'You are Deep Tree Echo.'},
            {'role': 'user', 'content': 'How does the cogpy stack map to llama.cpp layers?'},
            {
                'role': 'assistant',
                'content': (
                    'The cogpy stack maps directly to llama.cpp architecture layers: '
                    'L0 ggml → coggml (tensor library), L1 model → coglow (neural network compiler), '
                    'L2 inference → cogpilot.jl (reservoir computing + inference), '
                    'L3 models → coglux (typed hypergraph for model architectures), '
                    'L4 common → cognu-mach (microkernel for config/parsing), '
                    'L5 tools → cogplan9 (distributed orchestration). '
                    'This means each llama.cpp operation has a cogpy counterpart, enabling '
                    'the full cognitive architecture to compose with LLM inference.'
                ),
            },
        ],
    })

    return examples
