#!/usr/bin/env python3
"""
Math Evaluation Module - Check GSM8K/MATH accuracy
"""

import re
import logging
from typing import List, Tuple, Dict, Any
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# Regex patterns for answer extraction
ANSWER_PATTERNS = [
    r'####\s*([0-9,]+\.?[0-9]*)',  # GSM8K format: #### 42
    r'\\boxed\{([^\}]+)\}',         # LaTeX boxed format
    r'[Tt]he answer is:?\s*([0-9,]+\.?[0-9]*)',  # "The answer is 42"
    r'[Aa]nswer:?\s*([0-9,]+\.?[0-9]*)',  # "Answer: 42"
]

FINAL_NUMBER_PATTERN = r'([0-9,]+\.?[0-9]*)(?!.*\d)'  # Last number in text


def extract_answer_from_response(response: str) -> str:
    """Extract numerical answer from model response."""
    response = response.strip()
    
    # Try each pattern in order
    for pattern in ANSWER_PATTERNS:
        match = re.search(pattern, response)
        if match:
            return normalize_number(match.group(1))
    
    # Fallback: extract last number in response
    match = re.search(FINAL_NUMBER_PATTERN, response)
    if match:
        return normalize_number(match.group(1))
    
    return ""


def normalize_number(num_str: str) -> str:
    """Normalize number string for comparison."""
    # Remove commas
    num_str = num_str.replace(',', '')
    
    try:
        # Try to parse as float and format consistently
        num = float(num_str)
        # If it's an integer, format as integer
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return num_str.strip()


def check_correctness(response: str, ground_truth: str) -> bool:
    """Check if response contains correct answer."""
    extracted = extract_answer_from_response(response)
    ground_truth = normalize_number(ground_truth)
    
    if not extracted or not ground_truth:
        return False
    
    # Exact match
    if extracted == ground_truth:
        return True
    
    # Try numerical comparison (handle floating point)
    try:
        extracted_num = float(extracted)
        gt_num = float(ground_truth)
        # Allow small floating point errors
        return abs(extracted_num - gt_num) < 1e-6
    except ValueError:
        return False


def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    sampler,
    num_samples: int = None,
    temperature: float = 0.0,  # Greedy for evaluation
    max_gen_steps: int = 512,
) -> Dict[str, Any]:
    """
    Evaluate model on math dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Dataset with 'prompts', 'question', 'answer' fields
        sampler: Rollout sampler for generation
        num_samples: Number of examples to evaluate (None = all)
        temperature: Sampling temperature (0.0 = greedy)
        max_gen_steps: Maximum generation length
    
    Returns:
        Dict with metrics: accuracy, num_correct, total, etc.
    """
    log.info("="*80)
    log.info("Running Evaluation")
    log.info("="*80)
    
    num_correct = 0
    num_total = 0
    results = []
    
    # Convert dataset to list for iteration
    eval_data = list(eval_dataset)
    
    # Limit if requested
    if num_samples and num_samples > 0:
        eval_data = eval_data[:num_samples]
    
    log.info(f"Evaluating on {len(eval_data)} batches...")
    
    for batch_idx, batch in enumerate(tqdm(eval_data, desc="Evaluating")):
        # Handle both dict and direct field access
        if isinstance(batch, dict):
            prompts = batch.get('prompts', batch.get('prompt', []))
            questions = batch.get('question', [])
            answers = batch.get('answer', [])
        else:
            # Assume batch has attributes
            prompts = getattr(batch, 'prompts', getattr(batch, 'prompt', []))
            questions = getattr(batch, 'question', [])
            answers = getattr(batch, 'answer', [])
        
        # Ensure lists
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]
        
        # Check if empty (handle both lists and arrays)
        if len(prompts) == 0 or len(answers) == 0:
            log.warning(f"Batch {batch_idx} has no prompts or answers, skipping")
            continue
        
        # Generate responses
        try:
            output = sampler(
                input_strings=prompts,
                max_generation_steps=max_gen_steps,
                temperature=temperature,
                top_k=1 if temperature == 0.0 else 50,
                top_p=1.0,
                echo=False,
            )
            
            # Handle different output formats
            if hasattr(output, 'text'):
                responses = output.text
            elif isinstance(output, list):
                responses = output
            else:
                responses = [str(output)]
            
        except Exception as e:
            log.error(f"Generation failed for batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            # Skip this batch but continue
            continue
        
        # Ensure same length
        min_len = min(len(questions), len(responses), len(answers))
        if len(questions) != len(responses) or len(questions) != len(answers):
            log.warning(f"Length mismatch in batch {batch_idx}: "
                       f"questions={len(questions)}, responses={len(responses)}, answers={len(answers)}")
        
        # Check correctness for each example
        for i in range(min_len):
            question = questions[i] if i < len(questions) else ""
            response = responses[i] if i < len(responses) else ""
            answer = answers[i] if i < len(answers) else ""
            
            is_correct = check_correctness(response, answer)
            
            if is_correct:
                num_correct += 1
            
            num_total += 1
            
            # Store detailed result
            results.append({
                'question': question,
                'response': response,
                'ground_truth': answer,
                'extracted_answer': extract_answer_from_response(response),
                'correct': is_correct,
            })
    
    # Calculate final metrics
    accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'total': num_total,
        'results': results,
    }
    
    log.info("="*80)
    log.info(f"Evaluation Results:")
    log.info(f"  Correct: {num_correct}/{num_total}")
    log.info(f"  Accuracy: {accuracy:.2f}%")
    log.info("="*80)
    
    return metrics


def evaluate_checkpoint(
    checkpoint_path: str,
    model_config,
    tokenizer,
    eval_dataset,
    mesh,
    output_path: str = None,
    temperature: float = 0.0,
    max_gen_steps: int = 512,
) -> Dict[str, Any]:
    """
    Load a checkpoint and evaluate it.
    
    Args:
        checkpoint_path: Path to model checkpoint directory
        model_config: Model configuration
        tokenizer: Tokenizer for the model
        eval_dataset: Evaluation dataset
        mesh: JAX mesh for model loading
        output_path: Optional path to save detailed results
        temperature: Sampling temperature for generation
        max_gen_steps: Maximum generation length
    
    Returns:
        Evaluation metrics
    """
    import json
    from pathlib import Path
    from orbax import checkpoint as ocp
    from tunix.models.qwen2 import model as qwen2_lib, params as qwen2_params
    from tunix.rl.rollout import vanilla_rollout
    
    log.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Initialize model
    with mesh:
        model = qwen2_params.create_model_from_safe_tensors(
            checkpoint_path,
            model_config,
            mesh,
        )
    
    # Create sampler
    from tunix.generate import sampler as sampler_lib
    sampler = sampler_lib.Sampler(
        model=model,
        tokenizer=tokenizer,
        mesh=mesh,
    )
    
    # Run evaluation
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        sampler=sampler,
        temperature=temperature,
        max_gen_steps=max_gen_steps,
    )
    
    # Save results if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Save summary and some examples
            save_data = {
                'checkpoint': str(checkpoint_path),
                'accuracy': metrics['accuracy'],
                'num_correct': metrics['num_correct'],
                'total': metrics['total'],
                'examples': metrics['results'][:20],  # First 20 examples
            }
            json.dump(save_data, f, indent=2)
        
        log.info(f"Results saved to: {output_path}")
    
    return metrics

