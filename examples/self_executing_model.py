#!/usr/bin/env python3.11
"""
Self-Executing Model Implementation for Echo Adventure

Adds meta-circular evaluation capabilities where the model can:
1. Generate text with embedded dynamic code {{...}}
2. Execute that code referencing its own state
3. Introspect and report on its own parameters
4. Optionally modify its own configuration

This file can be added to echo-adventure/src/echo_adventure/execution_engine.py
"""

import torch
import torch.nn as nn
import re
import ast
import logging
from typing import Any, Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for dynamic code execution"""
    allow_modification: bool = False  # Allow self-modification
    timeout_seconds: int = 1  # Execution timeout
    max_iterations: int = 100  # Max loop iterations
    log_executions: bool = True  # Log all executions


class DynamicCodeParser:
    """
    Parse text containing {{...}} dynamic code blocks
    
    Supports:
    - Simple variables: {{temperature}}
    - Attribute access: {{self.temperature}}
    - Function calls: {{len(self.layer_weights)}}
    - Indexing: {{self.layer_weights[0]}}
    - Expressions: {{self.temperature * 2}}
    - Conditionals: {{value if condition else other}}
    - Format specifiers: {{temperature:.2f}}
    """
    
    # Pattern to match {{...}} blocks with optional format specifier
    PATTERN = re.compile(r'\{\{(.+?)(?::([^}]+))?\}\}', re.DOTALL)
    
    def parse(self, text: str) -> List[Tuple[str, str, Optional[str]]]:
        """
        Parse text into alternating text and code segments
        
        Args:
            text: Input text with {{...}} blocks
            
        Returns:
            List of (type, content, format_spec) tuples where type is 'text' or 'code'
        """
        tokens = []
        last_end = 0
        
        for match in self.PATTERN.finditer(text):
            # Add text before this match
            if match.start() > last_end:
                tokens.append(('text', text[last_end:match.start()], None))
            
            # Add code block with optional format specifier
            code = match.group(1).strip()
            format_spec = match.group(2)  # May be None
            tokens.append(('code', code, format_spec))
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            tokens.append(('text', text[last_end:], None))
        
        return tokens
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract just the code blocks"""
        return [code for typ, code, _ in self.parse(text) if typ == 'code']


class SafeEvaluator:
    """
    Safely evaluate Python expressions with restricted AST nodes
    
    Only allows safe operations:
    - Literals (numbers, strings, booleans)
    - Variables and attribute access
    - Function calls (to whitelisted functions)
    - Basic operators (+, -, *, /, ==, !=, <, >, etc.)
    - Conditionals (if-else expressions)
    - Indexing and slicing
    - Lists, tuples, dicts
    """
    
    # Allowed AST node types
    ALLOWED_NODES = {
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Attribute,
        ast.Call,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.IfExp,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.List,
        ast.Tuple,
        ast.Dict,
        # Operators
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
        ast.And, ast.Or, ast.Not,
        ast.UAdd, ast.USub,
        # Python 3.9+ compatibility
        getattr(ast, 'Num', type(None)),
        getattr(ast, 'Str', type(None)),
    }
    
    def is_safe(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code only uses allowed operations
        
        Returns:
            (is_safe, error_message) tuple
        """
        try:
            tree = ast.parse(code, mode='eval')
            
            for node in ast.walk(tree):
                if type(node) not in self.ALLOWED_NODES:
                    return False, f"Disallowed node type: {type(node).__name__}"
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
    
    def evaluate(self, code: str, context: dict) -> Any:
        """
        Safely evaluate code in given context
        
        Args:
            code: Python expression to evaluate
            context: Dictionary of available variables and functions
            
        Returns:
            Result of evaluation
            
        Raises:
            ValueError: If code is unsafe
            RuntimeError: If execution fails
        """
        # Check safety
        is_safe, error = self.is_safe(code)
        if not is_safe:
            raise ValueError(f"Unsafe code: {error}")
        
        # Evaluate with restricted builtins
        try:
            result = eval(code, {"__builtins__": {}}, context)
            return result
        except Exception as e:
            raise RuntimeError(f"Execution error in '{code}': {e}")


class ExecutionContext:
    """
    Build execution context with model state and safe functions
    """
    
    # Default safe functions available in all contexts
    SAFE_FUNCTIONS = {
        'len': len,
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'tuple': tuple,
        'dict': dict,
        'sorted': sorted,
        'enumerate': enumerate,
        'zip': zip,
        'range': range,
    }
    
    def __init__(
        self,
        model,
        config: ExecutionConfig,
        custom_functions: Optional[Dict[str, Callable]] = None
    ):
        self.model = model
        self.config = config
        self.custom_functions = custom_functions or {}
    
    def build_context(self) -> dict:
        """
        Build execution context with model state
        
        Returns:
            Dictionary of variables and functions available to dynamic code
        """
        context = {}
        
        # Add safe built-in functions
        context.update(self.SAFE_FUNCTIONS)
        
        # Add custom functions
        context.update(self.custom_functions)
        
        # Add model reference (read-only if modification not allowed)
        if self.config.allow_modification:
            context['self'] = self.model
        else:
            context['self'] = ReadOnlyProxy(self.model)
        
        # Add convenient shortcuts to inference parameters
        if hasattr(self.model, 'inference_engine'):
            engine = self.model.inference_engine
            context.update({
                'temperature': engine.temperature.item(),
                'top_p': engine.top_p.item(),
                'repetition_penalty': engine.repetition_penalty.item(),
                'layer_weights': engine.layer_weights.detach().cpu().numpy(),
                'head_weights': engine.head_weights.detach().cpu().numpy(),
            })
        
        # Add model metadata
        if hasattr(self.model, 'num_layers'):
            context['num_layers'] = self.model.num_layers
        if hasattr(self.model, 'num_heads'):
            context['num_heads'] = self.model.num_heads
        if hasattr(self.model, 'vocab_size'):
            context['vocab_size'] = self.model.vocab_size
        
        return context


class ReadOnlyProxy:
    """
    Proxy that prevents modifications to wrapped object
    
    Allows attribute access but prevents method calls and assignments
    """
    
    def __init__(self, obj):
        object.__setattr__(self, '_obj', obj)
        object.__setattr__(self, '_allowed_methods', {
            'get_inference_params',
            'count_parameters',
            'get_layer1_params',
            'get_layer2_params',
        })
    
    def __getattr__(self, name):
        obj = object.__getattribute__(self, '_obj')
        allowed = object.__getattribute__(self, '_allowed_methods')
        
        value = getattr(obj, name)
        
        # Allow read-only methods
        if callable(value) and name in allowed:
            return value
        
        # Prevent calling other methods
        if callable(value):
            raise AttributeError(f"Cannot call method '{name}' on read-only model")
        
        # Allow attribute access
        return value
    
    def __setattr__(self, name, value):
        raise AttributeError(f"Cannot modify read-only model")


class DynamicTemplateEngine:
    """
    Complete template engine with dynamic code execution
    
    Parses text, identifies {{...}} blocks, executes them safely,
    and substitutes results back into the text.
    """
    
    def __init__(
        self,
        model,
        config: Optional[ExecutionConfig] = None,
        custom_functions: Optional[Dict[str, Callable]] = None
    ):
        self.model = model
        self.config = config or ExecutionConfig()
        self.parser = DynamicCodeParser()
        self.evaluator = SafeEvaluator()
        self.context_builder = ExecutionContext(
            model, self.config, custom_functions
        )
        
        # Execution statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_time': 0.0,
        }
    
    def render(self, text: str) -> str:
        """
        Render template by executing dynamic code blocks
        
        Args:
            text: Input text with {{...}} blocks
            
        Returns:
            Text with {{...}} blocks replaced by execution results
        """
        # Parse text into tokens
        tokens = self.parser.parse(text)
        
        # Build execution context
        context = self.context_builder.build_context()
        
        # Process each token
        result_parts = []
        for token_type, content, format_spec in tokens:
            if token_type == 'text':
                # Plain text, pass through
                result_parts.append(content)
                
            elif token_type == 'code':
                # Dynamic code, execute it
                try:
                    value = self._execute_code(content, context)
                    
                    # Apply format specifier if provided
                    if format_spec:
                        formatted_value = format(value, format_spec)
                        result_parts.append(formatted_value)
                    else:
                        result_parts.append(str(value))
                    
                except Exception as e:
                    # Handle errors gracefully
                    error_msg = f"{{ERROR: {e}}}"
                    result_parts.append(error_msg)
                    
                    if self.config.log_executions:
                        logger.error(f"Execution failed: {content} -> {e}")
        
        return ''.join(result_parts)
    
    def _execute_code(self, code: str, context: dict) -> Any:
        """
        Execute a single code block
        
        Args:
            code: Python expression to execute
            context: Execution context
            
        Returns:
            Result of execution
        """
        import time
        
        start_time = time.time()
        
        try:
            # Log execution if enabled
            if self.config.log_executions:
                logger.info(f"Executing: {code}")
            
            # Execute
            result = self.evaluator.evaluate(code, context)
            
            # Update statistics
            self.stats['total_executions'] += 1
            self.stats['successful_executions'] += 1
            
            # Log result
            if self.config.log_executions:
                logger.info(f"Result: {result}")
            
            return result
            
        except Exception as e:
            self.stats['total_executions'] += 1
            self.stats['failed_executions'] += 1
            raise
            
        finally:
            elapsed = time.time() - start_time
            self.stats['total_time'] += elapsed
    
    def get_stats(self) -> dict:
        """Get execution statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset execution statistics"""
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_time': 0.0,
        }


# Example integration with TwoLayerModel
class SelfExecutingModelMixin:
    """
    Mixin to add self-executing capabilities to any model
    
    Usage:
        class MyModel(SelfExecutingModelMixin, TwoLayerModel):
            pass
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize template engine
        self.template_engine = DynamicTemplateEngine(
            self,
            config=ExecutionConfig(
                allow_modification=False,  # Safe by default
                log_executions=True,
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
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            execute_dynamic_code: Whether to execute {{...}} blocks
            tokenizer: Tokenizer for decoding (optional)
            
        Returns:
            Final text with dynamic code executed
        """
        # Generate raw text (may contain {{...}} blocks)
        generated_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_inference_engine=True
        )
        
        # Decode to text
        if tokenizer is not None:
            raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # Fallback: just return token IDs as string
            raw_text = str(generated_ids.tolist())
        
        # Execute dynamic code if enabled
        if execute_dynamic_code:
            final_text = self.template_engine.render(raw_text)
        else:
            final_text = raw_text
        
        return final_text
    
    def introspect(self) -> dict:
        """
        Get complete model state for self-reference
        
        Returns:
            Dictionary with all introspectable model state
        """
        state = {}
        
        # Inference parameters
        if hasattr(self, 'get_inference_params'):
            state['inference_params'] = self.get_inference_params()
        
        # Model architecture
        if hasattr(self, 'num_layers'):
            state['num_layers'] = self.num_layers
        if hasattr(self, 'num_heads'):
            state['num_heads'] = self.num_heads
        if hasattr(self, 'vocab_size'):
            state['vocab_size'] = self.vocab_size
        if hasattr(self, 'd_model'):
            state['d_model'] = self.d_model
        
        # Parameter counts
        if hasattr(self, 'count_parameters'):
            state['parameter_counts'] = self.count_parameters()
        
        # Execution statistics
        if hasattr(self, 'template_engine'):
            state['execution_stats'] = self.template_engine.get_stats()
        
        return state
    
    def enable_self_modification(self):
        """
        Enable self-modification (use with caution!)
        """
        self.template_engine.config.allow_modification = True
        logger.warning("Self-modification enabled - model can now modify its own parameters")
    
    def disable_self_modification(self):
        """
        Disable self-modification (safe mode)
        """
        self.template_engine.config.allow_modification = False


# Demo
def demo():
    """
    Demonstrate self-executing model capabilities
    """
    print("=" * 70)
    print("Self-Executing Model Demo")
    print("=" * 70)
    
    # Create mock model
    class MockInferenceEngine:
        def __init__(self):
            self.temperature = nn.Parameter(torch.tensor(0.743))
            self.top_p = nn.Parameter(torch.tensor(0.891))
            self.repetition_penalty = nn.Parameter(torch.tensor(1.123))
            self.layer_weights = nn.Parameter(torch.randn(12))
            self.head_weights = nn.Parameter(torch.randn(12, 8))
    
    class MockModel:
        def __init__(self):
            self.inference_engine = MockInferenceEngine()
            self.num_layers = 12
            self.num_heads = 8
            self.vocab_size = 10000
            self.d_model = 512
            
            # Add template engine
            self.template_engine = DynamicTemplateEngine(self)
        
        def get_inference_params(self):
            return {
                'temperature': self.inference_engine.temperature.item(),
                'top_p': self.inference_engine.top_p.item(),
                'repetition_penalty': self.inference_engine.repetition_penalty.item(),
            }
        
        def count_parameters(self):
            return {
                'layer1_params': 15234567,
                'layer2_params': 156,
                'total_params': 15234723,
            }
    
    model = MockModel()
    
    # Test cases
    test_cases = [
        ("Simple variable", "Temperature: {{temperature}}"),
        ("Attribute access", "My temp is {{self.inference_engine.temperature.item():.3f}}"),
        ("Function call", "I have {{self.num_layers}} layers"),
        ("Expression", "Double temp: {{temperature * 2:.3f}}"),
        ("Conditional", "I'm {{'creative' if temperature > 0.8 else 'precise'}}"),
        ("Complex", "Config: temp={{temperature:.2f}}, layers={{self.num_layers}}, params={{self.count_parameters()['total_params']:,}}"),
        ("Array access", "Top layer weight: {{layer_weights[-1]:.4f}}"),
        ("Nested", "Status: {{'hot' if temperature > 1.0 else 'warm' if temperature > 0.5 else 'cold'}}"),
    ]
    
    for name, template in test_cases:
        print(f"\n{name}:")
        print(f"  Template: {template}")
        result = model.template_engine.render(template)
        print(f"  Result:   {result}")
    
    # Show statistics
    print(f"\n{'Execution Statistics':-^70}")
    stats = model.template_engine.get_stats()
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Successful: {stats['successful_executions']}")
    print(f"  Failed: {stats['failed_executions']}")
    print(f"  Total time: {stats['total_time']:.4f}s")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    demo()
