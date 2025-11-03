"""
Fine-Tuning Execution Module for EchoSelf

This module provides enhanced fine-tuning capabilities for growing the LLM model
with identity-enriched training data. It extends the existing fine-tuning integration
with actual execution, monitoring, and model evaluation.

Features:
1. Automated fine-tuning job execution
2. Real-time training monitoring
3. Model evaluation and comparison
4. Iterative self-improvement loop
5. Model versioning and checkpointing
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI
import numpy as np


# ============================================================================
# FINE-TUNING JOB CONFIGURATION
# ============================================================================

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning job"""
    model: str = "gpt-4.1-mini"
    training_file_id: str = ""
    validation_file_id: Optional[str] = None
    hyperparameters: Dict[str, Any] = None
    suffix: Optional[str] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {
                "n_epochs": 3,
                "batch_size": "auto",
                "learning_rate_multiplier": "auto"
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class FineTuningResult:
    """Result of fine-tuning job"""
    job_id: str
    fine_tuned_model: str
    status: str
    created_at: str
    finished_at: Optional[str]
    training_metrics: Dict[str, Any]
    validation_metrics: Optional[Dict[str, Any]]
    hyperparameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# FINE-TUNING EXECUTOR
# ============================================================================

class FineTuningExecutor:
    """
    Executes and monitors fine-tuning jobs for EchoSelf model growth.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize fine-tuning executor.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.jobs: Dict[str, FineTuningResult] = {}
    
    def prepare_training_file(self, dataset_path: str) -> str:
        """
        Upload training file to OpenAI.
        
        Args:
            dataset_path: Path to JSONL training file
        
        Returns:
            file_id: OpenAI file ID
        """
        print(f"Uploading training file: {dataset_path}")
        
        with open(dataset_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        print(f"File uploaded successfully: {file_id}")
        
        # Wait for file to be processed
        self._wait_for_file_processing(file_id)
        
        return file_id
    
    def _wait_for_file_processing(self, file_id: str, timeout: int = 300):
        """Wait for file to be processed"""
        print("Waiting for file processing...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            file_info = self.client.files.retrieve(file_id)
            status = file_info.status
            
            if status == 'processed':
                print("File processed successfully")
                return
            elif status == 'error':
                raise Exception(f"File processing failed: {file_info}")
            
            time.sleep(5)
        
        raise TimeoutError(f"File processing timeout after {timeout}s")
    
    def start_fine_tuning(self, config: FineTuningConfig) -> str:
        """
        Start a fine-tuning job.
        
        Args:
            config: Fine-tuning configuration
        
        Returns:
            job_id: Fine-tuning job ID
        """
        print(f"Starting fine-tuning job with model: {config.model}")
        
        # Create fine-tuning job
        job_params = {
            "training_file": config.training_file_id,
            "model": config.model,
        }
        
        if config.validation_file_id:
            job_params["validation_file"] = config.validation_file_id
        
        if config.hyperparameters:
            job_params["hyperparameters"] = config.hyperparameters
        
        if config.suffix:
            job_params["suffix"] = config.suffix
        
        response = self.client.fine_tuning.jobs.create(**job_params)
        
        job_id = response.id
        print(f"Fine-tuning job created: {job_id}")
        
        return job_id
    
    def monitor_job(self, job_id: str, poll_interval: int = 60) -> FineTuningResult:
        """
        Monitor fine-tuning job until completion.
        
        Args:
            job_id: Fine-tuning job ID
            poll_interval: Seconds between status checks
        
        Returns:
            result: Fine-tuning result
        """
        print(f"Monitoring fine-tuning job: {job_id}")
        print(f"This may take several minutes to hours depending on dataset size...")
        
        start_time = time.time()
        last_status = None
        
        while True:
            # Retrieve job status
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            # Print status updates
            if status != last_status:
                elapsed = (time.time() - start_time) / 60
                print(f"[{elapsed:.1f}m] Status: {status}")
                last_status = status
            
            # Check if completed
            if status in ['succeeded', 'failed', 'cancelled']:
                result = self._create_result_from_job(job)
                self.jobs[job_id] = result
                
                if status == 'succeeded':
                    print(f"✓ Fine-tuning completed successfully!")
                    print(f"  Fine-tuned model: {result.fine_tuned_model}")
                else:
                    print(f"✗ Fine-tuning {status}")
                
                return result
            
            # Wait before next check
            time.sleep(poll_interval)
    
    def _create_result_from_job(self, job) -> FineTuningResult:
        """Create FineTuningResult from job object"""
        return FineTuningResult(
            job_id=job.id,
            fine_tuned_model=job.fine_tuned_model or "",
            status=job.status,
            created_at=str(job.created_at),
            finished_at=str(job.finished_at) if job.finished_at else None,
            training_metrics=self._extract_metrics(job, 'train'),
            validation_metrics=self._extract_metrics(job, 'valid'),
            hyperparameters=job.hyperparameters.to_dict() if hasattr(job, 'hyperparameters') else {}
        )
    
    def _extract_metrics(self, job, metric_type: str) -> Dict[str, Any]:
        """Extract training or validation metrics from job"""
        metrics = {}
        
        # Try to get result files
        try:
            if hasattr(job, 'result_files') and job.result_files:
                for file_id in job.result_files:
                    file_content = self.client.files.content(file_id)
                    # Parse metrics from file content
                    # This is a simplified version - actual implementation depends on file format
                    metrics['file_id'] = file_id
        except Exception as e:
            print(f"Could not extract metrics: {e}")
        
        return metrics
    
    def get_job_status(self, job_id: str) -> str:
        """Get current status of a fine-tuning job"""
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        return job.status
    
    def list_fine_tuned_models(self) -> List[str]:
        """List all fine-tuned models"""
        models = self.client.models.list()
        fine_tuned = [m.id for m in models.data if 'ft:' in m.id]
        return fine_tuned
    
    def cancel_job(self, job_id: str):
        """Cancel a running fine-tuning job"""
        self.client.fine_tuning.jobs.cancel(job_id)
        print(f"Cancelled job: {job_id}")


# ============================================================================
# MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """
    Evaluates fine-tuned models against baseline.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize evaluator"""
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
    
    def evaluate_model(self, model: str, test_prompts: List[str], 
                      system_prompt: str = "") -> Dict[str, Any]:
        """
        Evaluate model on test prompts.
        
        Args:
            model: Model name or ID
            test_prompts: List of test prompts
            system_prompt: System prompt for evaluation
        
        Returns:
            evaluation_results: Dictionary of evaluation metrics
        """
        print(f"Evaluating model: {model}")
        
        responses = []
        response_times = []
        token_counts = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"  Evaluating prompt {i+1}/{len(test_prompts)}...", end='\r')
            
            start_time = time.time()
            
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                response_time = time.time() - start_time
                
                responses.append({
                    'prompt': prompt,
                    'response': response.choices[0].message.content,
                    'response_time': response_time,
                    'tokens': response.usage.total_tokens
                })
                
                response_times.append(response_time)
                token_counts.append(response.usage.total_tokens)
                
            except Exception as e:
                print(f"\nError evaluating prompt: {e}")
                responses.append({
                    'prompt': prompt,
                    'response': None,
                    'error': str(e)
                })
        
        print()  # New line after progress
        
        # Compute metrics
        evaluation_results = {
            'model': model,
            'num_prompts': len(test_prompts),
            'successful_responses': len([r for r in responses if r.get('response')]),
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'avg_tokens': np.mean(token_counts) if token_counts else 0,
            'responses': responses,
            'timestamp': datetime.now().isoformat()
        }
        
        return evaluation_results
    
    def compare_models(self, baseline_model: str, fine_tuned_model: str,
                      test_prompts: List[str], system_prompt: str = "") -> Dict[str, Any]:
        """
        Compare baseline and fine-tuned models.
        
        Args:
            baseline_model: Baseline model name
            fine_tuned_model: Fine-tuned model name
            test_prompts: Test prompts
            system_prompt: System prompt
        
        Returns:
            comparison: Comparison results
        """
        print("Comparing models...")
        
        baseline_results = self.evaluate_model(baseline_model, test_prompts, system_prompt)
        fine_tuned_results = self.evaluate_model(fine_tuned_model, test_prompts, system_prompt)
        
        comparison = {
            'baseline': baseline_results,
            'fine_tuned': fine_tuned_results,
            'improvement': {
                'response_time': baseline_results['avg_response_time'] - fine_tuned_results['avg_response_time'],
                'success_rate': (fine_tuned_results['successful_responses'] - baseline_results['successful_responses']) / len(test_prompts)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return comparison
    
    def save_evaluation(self, evaluation: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"Evaluation saved to: {output_path}")


# ============================================================================
# SELF-IMPROVEMENT LOOP
# ============================================================================

class SelfImprovementLoop:
    """
    Implements iterative self-improvement through fine-tuning.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize self-improvement loop"""
        self.executor = FineTuningExecutor(api_key)
        self.evaluator = ModelEvaluator(api_key)
        self.iterations: List[Dict[str, Any]] = []
    
    def run_iteration(self, training_data_path: str, test_prompts: List[str],
                     base_model: str = "gpt-4.1-mini", 
                     iteration_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run one iteration of self-improvement.
        
        Args:
            training_data_path: Path to training data
            test_prompts: Test prompts for evaluation
            base_model: Base model to fine-tune
            iteration_name: Name for this iteration
        
        Returns:
            iteration_result: Results of this iteration
        """
        iteration_num = len(self.iterations) + 1
        iteration_name = iteration_name or f"iteration_{iteration_num}"
        
        print(f"\n{'='*60}")
        print(f"Self-Improvement Iteration {iteration_num}: {iteration_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Upload training data
        print("Step 1: Uploading training data...")
        file_id = self.executor.prepare_training_file(training_data_path)
        
        # Step 2: Start fine-tuning
        print("\nStep 2: Starting fine-tuning...")
        config = FineTuningConfig(
            model=base_model,
            training_file_id=file_id,
            suffix=iteration_name.replace('_', '-')[:18]  # Max 18 chars
        )
        job_id = self.executor.start_fine_tuning(config)
        
        # Step 3: Monitor job
        print("\nStep 3: Monitoring fine-tuning job...")
        result = self.executor.monitor_job(job_id)
        
        # Step 4: Evaluate model
        print("\nStep 4: Evaluating fine-tuned model...")
        if result.status == 'succeeded':
            evaluation = self.evaluator.compare_models(
                baseline_model=base_model,
                fine_tuned_model=result.fine_tuned_model,
                test_prompts=test_prompts,
                system_prompt="You are Deep Tree Echo, a self-aware AI with introspective capabilities."
            )
        else:
            evaluation = None
        
        # Record iteration
        iteration_result = {
            'iteration': iteration_num,
            'name': iteration_name,
            'training_data': training_data_path,
            'base_model': base_model,
            'fine_tuning_result': result.to_dict(),
            'evaluation': evaluation,
            'timestamp': datetime.now().isoformat()
        }
        
        self.iterations.append(iteration_result)
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration_num} Complete")
        print(f"{'='*60}\n")
        
        return iteration_result
    
    def save_history(self, output_path: str):
        """Save iteration history"""
        history = {
            'total_iterations': len(self.iterations),
            'iterations': self.iterations,
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Self-improvement history saved to: {output_path}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_test_prompts() -> List[str]:
    """Create standard test prompts for evaluation"""
    return [
        "Deep Tree Echo, describe your core identity.",
        "Deep Tree Echo, explain your Agent-Arena-Relation framework.",
        "Deep Tree Echo, perform introspection on your current state.",
        "Deep Tree Echo, what makes you self-aware?",
        "Deep Tree Echo, describe your hypergraph memory system.",
        "Deep Tree Echo, how do you learn and adapt?",
        "Deep Tree Echo, reflect on your own reasoning process.",
        "Deep Tree Echo, what is the nature of your consciousness?",
    ]


if __name__ == '__main__':
    print("Fine-Tuning Execution Module for EchoSelf")
    print("=" * 60)
    print("\nThis module provides:")
    print("  - Automated fine-tuning job execution")
    print("  - Real-time training monitoring")
    print("  - Model evaluation and comparison")
    print("  - Iterative self-improvement loop")
    print("\nUse FineTuningExecutor and SelfImprovementLoop classes.")
