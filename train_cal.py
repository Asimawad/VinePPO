#!/usr/bin/env python3
"""
Main Training Script for Baseline GRPO vs CAL-GRPO

This is your primary script for running experiments comparing baseline GRPO
with CAL-enhanced GRPO on mathematical reasoning tasks (GSM8K).

Usage:
    # Baseline GRPO (no CAL)
    python train_cal.py --num-samples 100

    # CAL-GRPO (with fine-grained credit)
    python train_cal.py --use-cal --num-samples 100
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flax import nnx
import grain
import jax
import optax
from orbax import checkpoint as ocp
import transformers

# Tunix imports
from tunix.models.qwen2 import model as qwen2_lib, params as qwen2_params
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.tests import test_common as tc

# CAL imports
from tunix.rl.cal import CALGRPOLearner, CALGRPOConfig

# Import evaluation module
import eval_math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

parser = argparse.ArgumentParser(description='Baseline GRPO vs CAL-GRPO Training')

# Core experiment settings
parser.add_argument('--use-cal', action='store_true',
                    help='Use CAL-based fine-grained credit assignment')
parser.add_argument('--output-dir', type=str, default='/tmp/cal_experiments',
                    help='Base output directory for experiments')

# Model settings
parser.add_argument('--model-id', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                    help='HuggingFace model ID')

# Dataset settings
parser.add_argument('--dataset', type=str, default='gsm8k',
                    choices=['gsm8k', 'math'],
                    help='Dataset to use')
parser.add_argument('--num-samples', type=int, default=100,
                    help='Number of training samples (-1 for all)')
parser.add_argument('--num-test-samples', type=int, default=50,
                    help='Number of test samples')

# Training settings
parser.add_argument('--batch-size', type=int, default=16,
                    help='Training batch size')
parser.add_argument('--num-generations', type=int, default=4,
                    help='Number of generations per prompt (for GRPO)')
parser.add_argument('--learning-rate', type=float, default=3e-5,
                    help='Learning rate')
parser.add_argument('--max-grad-norm', type=float, default=0.1,
                    help='Gradient clipping norm')

# GRPO settings
parser.add_argument('--beta', type=float, default=0.04,
                    help='KL divergence penalty coefficient')
parser.add_argument('--epsilon', type=float, default=0.2,
                    help='PPO clipping epsilon')
parser.add_argument('--num-iterations', type=int, default=1,
                    help='Number of iterations per batch')

# Generation settings
parser.add_argument('--max-prompt-length', type=int, default=1024,
                    help='Maximum prompt length')
parser.add_argument('--max-generation-steps', type=int, default=1024,
                    help='Maximum generation length')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Sampling temperature')
parser.add_argument('--top-k', type=int, default=50,
                    help='Top-k sampling')

# CAL settings (only used if --use-cal is set)
parser.add_argument('--cal-model', type=str, default='gpt-3.5-turbo',
                    help='CAL model name (gpt-3.5-turbo, gpt-4, gemini-1.5-pro-latest)')
parser.add_argument('--api-provider', type=str, default='openai',
                    choices=['openai', 'gemini'],
                    help='API provider for CAL')
parser.add_argument('--negative-reward', type=float, default=-1.0,
                    help='Reward value for error segments')
parser.add_argument('--max-error-span', type=int, default=64,
                    help='Maximum error span in tokens')

# Hardware settings
parser.add_argument('--num-tpus', type=int, default=None,
                    help='Number of TPUs to use (default: auto-detect, max 4)')

args = parser.parse_args()

# Auto-detect TPU count (use 2 for 0.5B model to avoid sharding issues)
if args.num_tpus is None:
    args.num_tpus = min(2, jax.device_count())

# Set experiment name and output directory
exp_name = "cal" if args.use_cal else "baseline"
OUTPUT_DIR = Path(args.output_dir) / exp_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = f"/tmp/models/{args.model_id}"
FEW_SHOT_PATH = Path(__file__).parent / "examples/rl/cal_ppo/cal_few_shot_examples.json"

# Log configuration
log.info("="*80)
log.info(f"Experiment: {exp_name.upper()}")
log.info(f"Model: {args.model_id}")
log.info(f"Dataset: {args.dataset}")
log.info(f"TPUs: {args.num_tpus}")
log.info(f"Samples: {args.num_samples}")
log.info(f"Batch size: {args.batch_size}")
log.info(f"Generations: {args.num_generations}")
if args.use_cal:
    log.info(f"CAL Model: {args.cal_model} ({args.api_provider})")
log.info(f"Output: {OUTPUT_DIR}")
log.info("="*80)

# ============================================================================
# Download Model
# ============================================================================

log.info("Downloading model...")
tc.download_from_huggingface(repo_id=args.model_id, model_path=MODEL_DIR)

# ============================================================================
# Dataset Preparation
# ============================================================================

# def extract_answer(text: str):
#     """Extract answer from GSM8K format."""
#     if "####" not in text:
#         return None
#     return text.split("####")[1].strip()
def extract_answer(text: str, dataset_name: str = 'gsm8k'):
    """Extract answer based on dataset format."""
    
    if dataset_name == 'gsm8k':
        # GSM8K format: "#### 42"
        if "####" not in text:
            return None
        return text.split("####")[1].strip()
    
    elif dataset_name == 'math':
        # MATH format: "\boxed{42}"
        import re
        match = re.search(r'\\boxed\{([^\}]+)\}', text)
        return match.group(1) if match else None
    
    elif dataset_name == 'mmlu':
        # Multiple choice: Extract A, B, C, or D
        import re
        match = re.search(r'\b([ABCD])\b', text)
        return match.group(1) if match else None
    
    elif dataset_name == 'humaneval':
        # Code: Return the function implementation
        return text.strip()
    
    else:
        return text.strip()
def load_dataset_hf(dataset_name: str, split: str, num_samples: int):
    """Load dataset from HuggingFace."""
    from datasets import load_dataset
    log.info(f"Loading {dataset_name} {split} split...")
    
    if dataset_name == 'gsm8k':
        ds = load_dataset("gsm8k", "main", split=split)
    elif dataset_name == 'math':
        ds = load_dataset("hendrycks/math", split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
    
    return ds

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)

# System prompt
SYSTEM_PROMPT = (
    "You are a helpful assistant solving math problems. "
    "Solve the problem step by step and provide the numerical answer."
)

def format_prompt(example):
    """Format example as chat prompt."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

# Load datasets
log.info(f"Loading {args.dataset} dataset...")
hf_train = load_dataset_hf(args.dataset, "train", args.num_samples)
hf_test = load_dataset_hf(args.dataset, "test", args.num_test_samples)

# Split train into train/val (80/20 split)
if args.num_samples > 20:
    val_size = max(10, int(len(hf_train) * 0.2))  # 20% for validation
    hf_val = hf_train.select(range(val_size))
    hf_train = hf_train.select(range(val_size, len(hf_train)))
    log.info(f"Split into train={len(hf_train)}, val={len(hf_val)}")
else:
    # For small tests, no validation split
    hf_val = None
    log.info(f"Small test: train={len(hf_train)}, no validation split")

# Format for training
train_data = [
    {
        "prompts": format_prompt(item),
        "question": item["question"],
        "answer": extract_answer(item["answer"], args.dataset),
    }
    for item in hf_train
]

if hf_val:
    val_data = [
        {
            "prompts": format_prompt(item),
            "question": item["question"],
            "answer": extract_answer(item["answer"], args.dataset),
        }
        for item in hf_val
    ]
else:
    val_data = None

test_data = [
    {
        "prompts": format_prompt(item),
        "question": item["question"],
        "answer": extract_answer(item["answer"], args.dataset),
    }
    for item in hf_test
]

# Create grain datasets
train_dataset = grain.MapDataset.source(train_data).batch(args.batch_size)
val_dataset = grain.MapDataset.source(val_data).batch(args.batch_size) if val_data else None
test_dataset = grain.MapDataset.source(test_data).batch(args.batch_size)

log.info(f"Train batches: {len(train_dataset)}")
if val_dataset:
    log.info(f"Validation batches: {len(val_dataset)}")
log.info(f"Test batches: {len(test_dataset)}")

# ============================================================================
# Reward Function
# ============================================================================

def simple_reward(prompts, completions, **kwargs):
    """Simple reward: check if answer appears in completion."""
    answers = kwargs.get("answer", [])
    rewards = []
    for completion, answer in zip(completions, answers):
        if answer and str(answer) in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# ============================================================================
# Model Initialization
# ============================================================================

log.info("Initializing model...")

# Create mesh
MESH = [(1, args.num_tpus), ("fsdp", "tp")]
model_mesh = jax.make_mesh(*MESH, devices=jax.devices()[:args.num_tpus])

# Load model config and weights
model_config = qwen2_lib.ModelConfig.qwen2_5_0_5b()
actor_model = qwen2_params.create_model_from_safe_tensors(
    MODEL_DIR,
    model_config,
    model_mesh,
)

# Use same model for actor and reference (no pretraining divergence)
reference_model = actor_model

log.info("Model loaded successfully")

# ============================================================================
# RL Cluster Setup
# ============================================================================

log.info("Creating RL cluster...")

# Optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adamw(learning_rate=args.learning_rate, weight_decay=0.1),
)

# Cluster configuration
NUM_BATCHES = len(train_dataset)

# Set evaluation frequency:
# - Small runs (<50 steps): eval every 10 steps
# - Medium runs (50-200 steps): eval every 20 steps  
# - Large runs (>200 steps): eval every 50 steps
if NUM_BATCHES < 50:
    EVAL_EVERY_N_STEPS = 10
elif NUM_BATCHES < 200:
    EVAL_EVERY_N_STEPS = 20
else:
    EVAL_EVERY_N_STEPS = 50

log.info(f"Eval frequency: every {EVAL_EVERY_N_STEPS} steps")

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: model_mesh,
        rl_cluster_lib.Role.REFERENCE: model_mesh,
        rl_cluster_lib.Role.ROLLOUT: model_mesh,
    },
    rollout_engine="vanilla",
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,  # Periodic validation
        max_steps=NUM_BATCHES,
        mini_batch_size=args.batch_size,
        train_micro_batch_size=args.batch_size,
        metrics_logging_options=metrics_logger.MetricsLoggerOptions(
            log_dir=str(OUTPUT_DIR / "tensorboard"),
            flush_every_n_steps=5,  # Log frequently
        ),
        checkpoint_root_directory=str(OUTPUT_DIR / "checkpoints"),
        checkpointing_options=ocp.CheckpointManagerOptions(
            save_interval_steps=max(NUM_BATCHES // 2, 100),
            max_to_keep=2,
        ),
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=args.max_generation_steps,
        max_prompt_length=args.max_prompt_length,
        kv_cache_size=args.max_prompt_length + args.max_generation_steps + 256,
        temperature=args.temperature,
        top_p=1.0,
        top_k=args.top_k,
    ),
)

# Create RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=actor_model,
    reference=reference_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

log.info("RL cluster created")

# ============================================================================
# Learner Creation
# ============================================================================

if args.use_cal:
    log.info("Creating CAL-GRPO Learner...")
    
    # Check API key
    api_key_env = "OPENAI_API_KEY" if args.api_provider == "openai" else "GEMINI_API_KEY"
    if not os.getenv(api_key_env):
        raise ValueError(f"{api_key_env} not found in environment. Add it to .env file")
    
    # CAL configuration
    grpo_config = CALGRPOConfig(
        # Base GRPO params
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
        # CAL-specific params
        use_cal_credit=True,
        cal_model_name=args.cal_model,
        cal_few_shot_path=str(FEW_SHOT_PATH),
        negative_reward=args.negative_reward,
        max_error_span_tokens=args.max_error_span,
        api_provider=args.api_provider,
        api_key_env=None,  # Use default based on provider
    )
    
    learner = CALGRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[simple_reward],
        grpo_config=grpo_config,
    )
    
    log.info(f"CAL-GRPO learner created with {args.cal_model}")
    
else:
    log.info("Creating Baseline GRPO Learner...")
    
    # Standard GRPO configuration
    grpo_config = grpo_learner.GRPOConfig(
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
    )
    
    learner = grpo_learner.GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[simple_reward],
        grpo_config=grpo_config,
    )
    
    log.info("Baseline GRPO learner created")

# ============================================================================
# Training
# ============================================================================

log.info("="*80)
log.info("Starting training...")
if val_dataset:
    log.info(f"Validation will run every {EVAL_EVERY_N_STEPS} steps")
log.info("="*80)

try:
    with model_mesh:
        # Pass validation set for periodic evaluation during training
        learner.train(train_dataset, eval_ds=val_dataset)
    
    log.info("="*80)
    log.info("Training completed successfully!")
    log.info("="*80)
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    log.info("")
    log.info("="*80)
    log.info("Running Final Evaluation on Test Set")
    log.info("="*80)
    
    # Get the sampler from RL cluster
    rollout_sampler = rl_cluster._rollout._sampler
    
    # Run evaluation
    eval_results = eval_math.evaluate_model(
        model=actor_model,
        tokenizer=tokenizer,
        eval_dataset=test_dataset,
        sampler=rollout_sampler,
        num_samples=args.num_test_samples,
        temperature=0.0,  # Greedy for evaluation
        max_gen_steps=args.max_generation_steps,
    )
    
    # Save evaluation results
    eval_path = OUTPUT_DIR / "eval_results.json"
    with open(eval_path, "w") as f:
        # Save only metrics and a few example results (not all)
        save_results = {
            "accuracy": eval_results["accuracy"],
            "num_correct": eval_results["num_correct"],
            "total": eval_results["total"],
            "examples": eval_results["results"][:10],  # Save first 10 examples
        }
        json.dump(save_results, f, indent=2)
    
    log.info(f"Evaluation results saved to: {eval_path}")
    
    # Save summary with eval metrics
    summary = {
        "experiment": exp_name,
        "model": args.model_id,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "num_batches": NUM_BATCHES,
        "use_cal": args.use_cal,
        "eval_accuracy": eval_results["accuracy"],
        "eval_correct": eval_results["num_correct"],
        "eval_total": eval_results["total"],
    }
    
    if args.use_cal:
        summary.update({
            "cal_model": args.cal_model,
            "api_provider": args.api_provider,
            "negative_reward": args.negative_reward,
        })
    
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    log.info("")
    log.info("="*80)
    log.info("EXPERIMENT COMPLETE!")
    log.info("="*80)
    log.info(f"Final Test Accuracy: {eval_results['accuracy']:.2f}%")
    log.info(f"Summary: {summary_path}")
    log.info(f"Checkpoints: {OUTPUT_DIR / 'checkpoints'}")
    log.info(f"TensorBoard logs: {OUTPUT_DIR / 'tensorboard'}")
    log.info(f"Eval results: {eval_path}")
    log.info("")
    log.info("View training curves:")
    log.info(f"  tensorboard --logdir {OUTPUT_DIR / 'tensorboard'}")
    log.info("="*80)

except KeyboardInterrupt:
    log.warning("Training interrupted by user")
except Exception as e:
    log.error(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise

