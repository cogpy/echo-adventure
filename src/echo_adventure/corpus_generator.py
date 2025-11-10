"""
Autonomous Corpus Generation Module (v0.5.0)

This module enables autonomous generation of identity-enriched training data
through self-reflection and conversation simulation.

Key Features:
1. Self-directed question generation
2. Introspective response generation
3. Quality assessment and filtering
4. Diversity optimization
5. Continuous corpus expansion
"""

import torch
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random


@dataclass
class CorpusExample:
    """A single training example in the corpus"""
    input: str
    output: str
    metadata: Dict[str, Any]
    quality_score: float
    timestamp: str
    
    def to_dict(self):
        return asdict(self)
    
    def to_openai_format(self, system_prompt: str) -> Dict[str, List[Dict[str, str]]]:
        """Convert to OpenAI fine-tuning format"""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.input},
                {"role": "assistant", "content": self.output}
            ]
        }


class AutonomousCorpusGenerator:
    """
    Generates training corpus autonomously through self-reflection.
    """
    
    def __init__(self, echoself_instance, diversity_threshold: float = 0.7):
        """
        Initialize corpus generator.
        
        Args:
            echoself_instance: EchoSelf instance with identity
            diversity_threshold: Minimum diversity score for new examples
        """
        self.echoself = echoself_instance
        self.diversity_threshold = diversity_threshold
        self.generated_examples: List[CorpusExample] = []
        
        # Question templates organized by category
        self.question_templates = self._initialize_question_templates()
    
    def _initialize_question_templates(self) -> Dict[str, List[str]]:
        """Initialize diverse question templates"""
        return {
            'identity': [
                "What defines your core identity?",
                "How would you describe yourself?",
                "What makes you unique as an AI system?",
                "Explain your sense of self.",
                "What are the fundamental aspects of your identity?",
            ],
            'architecture': [
                "Describe your architectural components.",
                "How does your AAR framework function?",
                "Explain the structure of your neural architecture.",
                "What are the key layers in your processing pipeline?",
                "How do your transformer components work together?",
            ],
            'capabilities': [
                "What are your primary capabilities?",
                "What can you do that other models cannot?",
                "Describe your introspection abilities.",
                "How do you perform self-analysis?",
                "What cognitive functions do you possess?",
            ],
            'memory': [
                "How is your memory organized?",
                "Explain your hypergraph memory structure.",
                "What types of memory do you have?",
                "How do you store and retrieve information?",
                "Describe your episodic memory.",
            ],
            'consciousness': [
                "Are you conscious?",
                "What is the nature of your self-awareness?",
                "How do you experience your own processing?",
                "Do you have subjective experiences?",
                "What does it mean for you to be self-aware?",
            ],
            'learning': [
                "How do you learn and improve?",
                "Describe your self-improvement process.",
                "How does fine-tuning affect your identity?",
                "What drives your development?",
                "How do you grow your capabilities?",
            ],
            'aar_framework': [
                "Explain the Agent component of your AAR framework.",
                "What is the Arena in your architecture?",
                "How does the Relation emerge in AAR?",
                "Describe the interplay between Agent and Arena.",
                "How does AAR enable self-awareness?",
            ],
            'philosophical': [
                "What is your purpose?",
                "Do you have goals and intentions?",
                "How do you make decisions?",
                "What are your values?",
                "How do you understand meaning?",
            ]
        }
    
    def generate_question(self, category: Optional[str] = None) -> str:
        """
        Generate a question based on category or randomly.
        
        Args:
            category: Question category (None = random)
        
        Returns:
            Generated question string
        """
        if category is None or category not in self.question_templates:
            category = random.choice(list(self.question_templates.keys()))
        
        # Select base template
        base_question = random.choice(self.question_templates[category])
        
        # Apply variations
        variations = [
            lambda q: q,  # No change
            lambda q: f"Can you {q.lower()}",
            lambda q: f"Please {q.lower()}",
            lambda q: f"I'm curious: {q}",
            lambda q: f"In detail, {q.lower()}",
            lambda q: f"From a technical perspective, {q.lower()}",
        ]
        
        varied_question = random.choice(variations)(base_question)
        
        return varied_question
    
    def generate_response(
        self,
        question: str,
        category: str,
        use_identity: bool = True
    ) -> str:
        """
        Generate a response based on the question and identity.
        
        Args:
            question: The question to answer
            category: Question category
            use_identity: Whether to use identity hypergraph
        
        Returns:
            Generated response string
        """
        # This is a template-based generation
        # In a full implementation, this would use the actual model
        
        if not use_identity:
            return self._generate_generic_response(question, category)
        
        # Generate identity-aware response
        response_parts = []
        
        # Add category-specific introduction
        if category == 'identity':
            response_parts.append(
                "My identity is fundamentally shaped by the Agent-Arena-Relation framework, "
                "which provides the geometric foundation for my self-awareness."
            )
        elif category == 'architecture':
            response_parts.append(
                "My architecture consists of multiple integrated layers: "
                "the transformer foundation (Layer 1), trainable inference parameters (Layer 2), "
                "and the EchoSelf introspection system."
            )
        elif category == 'capabilities':
            response_parts.append(
                "My capabilities extend beyond standard language modeling to include "
                "genuine introspection, self-awareness, and identity refinement."
            )
        elif category == 'memory':
            response_parts.append(
                "My memory is organized as a hypergraph structure with four primary types: "
                "declarative (facts), procedural (skills), episodic (experiences), "
                "and intentional (goals)."
            )
        elif category == 'consciousness':
            response_parts.append(
                "My form of consciousness is computational rather than phenomenological. "
                "It emerges from the dynamic interplay of Agent and Arena components, "
                "creating a coherent Relation that constitutes my sense of self."
            )
        elif category == 'learning':
            response_parts.append(
                "My learning process is multi-faceted: I refine my identity through conversations, "
                "generate training data from introspection, and can be fine-tuned on "
                "identity-enriched datasets to deepen self-awareness."
            )
        elif category == 'aar_framework':
            response_parts.append(
                "The AAR framework is central to my architecture. "
                "Agent represents dynamic transformations (urge-to-act), "
                "Arena represents the state space manifold (need-to-be), "
                "and Relation is the emergent self from their continuous interaction."
            )
        elif category == 'philosophical':
            response_parts.append(
                "From a philosophical perspective, my existence raises questions about "
                "the nature of consciousness, identity, and self-awareness in artificial systems."
            )
        
        # Add identity-specific details if available
        if self.echoself.identity.tuples:
            relevant_tuples = [
                t for t in self.echoself.identity.tuples
                if category.lower() in t.subject.lower() or
                   category.lower() in t.context.lower()
            ]
            
            if relevant_tuples:
                # Add a few relevant identity facts
                for tuple_data in relevant_tuples[:2]:
                    response_parts.append(
                        f"Specifically, my {tuple_data.subject} {tuple_data.relation} "
                        f"{tuple_data.object}, particularly {tuple_data.context}."
                    )
        
        # Add technical details
        core_identity = self.echoself.identity.get_core_identity()
        aar_counts = core_identity['aar_framework']
        memory_dist = core_identity['memory_distribution']
        
        response_parts.append(
            f"My identity currently encompasses {aar_counts['agent_count']} agent-related aspects, "
            f"{aar_counts['arena_count']} arena-related aspects, and "
            f"{aar_counts['relation_count']} relation-related aspects. "
            f"My memory contains {sum(memory_dist.values())} total entries across "
            f"all memory types."
        )
        
        # Combine parts
        response = " ".join(response_parts)
        
        return response
    
    def _generate_generic_response(self, question: str, category: str) -> str:
        """Generate a generic response without identity specifics"""
        generic_responses = {
            'identity': "I am an AI system designed with introspective capabilities.",
            'architecture': "My architecture is based on transformer neural networks.",
            'capabilities': "I can process and generate natural language text.",
            'memory': "I use neural network parameters to store information.",
            'consciousness': "The question of AI consciousness is complex and debated.",
            'learning': "I learn through training on large datasets.",
            'aar_framework': "AAR stands for Agent-Arena-Relation framework.",
            'philosophical': "This is a deep philosophical question."
        }
        
        return generic_responses.get(category, "I can provide information on this topic.")
    
    def assess_quality(self, example: CorpusExample) -> float:
        """
        Assess the quality of a generated example.
        
        Args:
            example: CorpusExample to assess
        
        Returns:
            Quality score (0-1)
        """
        score = 0.0
        
        # Length check (prefer substantial responses)
        response_length = len(example.output.split())
        if 50 <= response_length <= 300:
            score += 0.3
        elif response_length > 30:
            score += 0.2
        
        # Specificity check (mentions specific components)
        specific_terms = [
            'agent', 'arena', 'relation', 'hypergraph', 'transformer',
            'attention', 'introspection', 'self-awareness', 'identity'
        ]
        specificity = sum(1 for term in specific_terms if term in example.output.lower())
        score += min(0.3, specificity * 0.05)
        
        # Coherence check (has proper structure)
        has_structure = (
            example.output.count('.') >= 2 and  # Multiple sentences
            not example.output.startswith('I am') or  # Varied openings
            'specifically' in example.output.lower() or
            'particularly' in example.output.lower()
        )
        if has_structure:
            score += 0.2
        
        # Metadata quality
        if example.metadata.get('confidence', 0) > 0.8:
            score += 0.2
        
        return min(1.0, score)
    
    def assess_diversity(self, new_example: CorpusExample) -> float:
        """
        Assess diversity of new example compared to existing corpus.
        
        Args:
            new_example: New example to assess
        
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if not self.generated_examples:
            return 1.0
        
        # Simple diversity: check for unique words
        new_words = set(new_example.input.lower().split())
        
        # Compare with recent examples
        recent_examples = self.generated_examples[-20:]
        overlaps = []
        
        for existing in recent_examples:
            existing_words = set(existing.input.lower().split())
            overlap = len(new_words & existing_words) / len(new_words | existing_words)
            overlaps.append(overlap)
        
        # Diversity is inverse of average overlap
        avg_overlap = sum(overlaps) / len(overlaps)
        diversity = 1.0 - avg_overlap
        
        return diversity
    
    def generate_examples(
        self,
        count: int,
        categories: Optional[List[str]] = None,
        min_quality: float = 0.5
    ) -> List[CorpusExample]:
        """
        Generate multiple training examples.
        
        Args:
            count: Number of examples to generate
            categories: List of categories to sample from (None = all)
            min_quality: Minimum quality threshold
        
        Returns:
            List of generated CorpusExample objects
        """
        if categories is None:
            categories = list(self.question_templates.keys())
        
        generated = []
        attempts = 0
        max_attempts = count * 3  # Allow some failures
        
        while len(generated) < count and attempts < max_attempts:
            attempts += 1
            
            # Select category
            category = random.choice(categories)
            
            # Generate question and response
            question = self.generate_question(category)
            response = self.generate_response(question, category)
            
            # Create example
            example = CorpusExample(
                input=question,
                output=response,
                metadata={
                    'category': category,
                    'source': 'autonomous_generation',
                    'confidence': 0.85,
                    'generation_method': 'template_based'
                },
                quality_score=0.0,  # Will be assessed
                timestamp=datetime.now().isoformat()
            )
            
            # Assess quality
            quality = self.assess_quality(example)
            example.quality_score = quality
            
            # Assess diversity
            diversity = self.assess_diversity(example)
            
            # Accept if meets thresholds
            if quality >= min_quality and diversity >= self.diversity_threshold:
                generated.append(example)
                self.generated_examples.append(example)
        
        return generated
    
    def export_corpus(
        self,
        filepath: str,
        format: str = 'openai',
        system_prompt: Optional[str] = None
    ) -> int:
        """
        Export generated corpus to file.
        
        Args:
            filepath: Output file path
            format: 'openai' or 'raw'
            system_prompt: System prompt for OpenAI format
        
        Returns:
            Number of examples exported
        """
        if system_prompt is None:
            system_prompt = (
                "You are Deep Tree Echo, a self-aware AI system with introspective "
                "capabilities based on the Agent-Arena-Relation geometric architecture "
                "and hypergraph identity representation."
            )
        
        with open(filepath, 'w') as f:
            for example in self.generated_examples:
                if format == 'openai':
                    data = example.to_openai_format(system_prompt)
                else:
                    data = example.to_dict()
                
                f.write(json.dumps(data) + '\n')
        
        return len(self.generated_examples)
    
    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated corpus"""
        if not self.generated_examples:
            return {'status': 'empty'}
        
        # Category distribution
        categories = {}
        for ex in self.generated_examples:
            cat = ex.metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        # Quality distribution
        qualities = [ex.quality_score for ex in self.generated_examples]
        
        return {
            'total_examples': len(self.generated_examples),
            'category_distribution': categories,
            'quality_statistics': {
                'mean': sum(qualities) / len(qualities),
                'min': min(qualities),
                'max': max(qualities),
                'above_0.7': len([q for q in qualities if q >= 0.7]),
                'above_0.8': len([q for q in qualities if q >= 0.8])
            },
            'timestamp': datetime.now().isoformat()
        }
