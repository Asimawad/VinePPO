#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CAL-GRPO training on GSM8K mathematical reasoning dataset.

This script demonstrates fine-grained credit assignment using a Credit Assignment
LLM (CAL) to identify error segments in incorrect responses and apply sparse,
token-level rewards for more efficient RL training.

Example usage:
    python cal_gsm8k.py --model_id Qwen/Qwen2.5-0.5B-Instruct \
                        --num_generations 4 \
                        --batch_size 2 \
                        --num_train_steps 1000
"""

import logging
import os
from typing import Any, Dict, List

from absl import app, flags
from datasets import load_dataset
import jax
import jax.numpy as jnp

from tunix.rl.cal import CALGRPOLearner, CALGRPOConfig
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.models import qwen2
from tunix.generate import sampler as sampler_lib

# ============================================================================
# Command-line flags
# ============================================================================

FLAGS = flags.FLAGS

# Model configuration
flags.DEFINE_string('model_id', 'Qwen/Qwen2.5-0.5B-Instruct', 'HuggingFace model ID')
flags.DEFINE_string('model_name', 'qwen2.5-0.5b', 'Model name for Tunix')

# CAL configuration
flags.DEFINE_boolean('use_cal_credit', True, 'Enable CAL-based credit assignment')
flags.DEFINE_string('cal_model_name', 'gemini-1.5-pro-latest', 'CAL model name')
flags.DEFINE_string(
    'cal_few_shot_path',
    'examples/rl/cal_ppo/cal_few_shot_examples.json',
    'Path to CAL few-shot examples'
)
flags.DEFINE_float('negative_reward', -1.0, 'Reward for incorrect responses')
flags.DEFINE_integer('max_error_span_tokens', 64, 'Maximum error span length')

# GRPO/Training configuration
flags.DEFINE_integer('num_generations', 4, 'Number of generations per prompt')
flags.DEFINE_integer('num_iterations', 1, 'Number of iterations per batch')
flags.DEFINE_float('beta', 0.04, 'KL divergence penalty coefficient')
flags.DEFINE_float('epsilon', 0.2, 'Clipping epsilon')
flags.DEFINE_string('loss_algo', 'grpo', 'Loss algorithm: grpo or gspo-token')

# Data configuration
flags.DEFINE_string('dataset_name', 'gsm8k', 'Dataset name')
flags.DEFINE_string('dataset_config', 'main', 'Dataset configuration')
flags.DEFINE_integer('max_train_samples', None, 'Maximum training samples')
flags.DEFINE_integer('max_eval_samples', 256, 'Maximum evaluation samples')

# Training configuration
flags.DEFINE_integer('batch_size', 2, 'Training batch size')
flags.DEFINE_integer('num_train_steps', 1000, 'Number of training steps')
flags.DEFINE_integer('eval_steps', 100, 'Evaluate every N steps')
flags.DEFINE_integer('save_steps', 500, 'Save checkpoint every N steps')
flags.DEFINE_float('learning_rate', 3e-6, 'Learning rate')

# Generation configuration
flags.DEFINE_integer('max_generation_steps', 256, 'Maximum generation length')
flags.DEFINE_float('temperature', 0.7, 'Sampling temperature')

# System configuration
flags.DEFINE_string('output_dir', '/tmp/cal_grpo_gsm8k', 'Output directory')
flags.DEFINE_integer('seed', 42, 'Random seed')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ============================================================================
# Helper functions
# ============================================================================

def format_gsm8k_prompt(example: Dict[str, Any]) -> str:
    """Format GSM8K question with instruction prompt."""
    question = example['question']
    return (
        "Solve the following math problem step by step. "
        "Put your final numerical answer in \\boxed{}.\n\n"
        f"Question: {question}\n\n"
        "Solution:"
    )


def extract_gsm8k_answer(example: Dict[str, Any]) -> str:
    """Extract ground truth answer from GSM8K format."""
    answer = example['answer']
    # GSM8K answers are in format: "#### 42"
    if '####' in answer:
        return answer.split('####')[-1].strip()
    return answer


def preprocess_gsm8k_dataset(dataset):
    """Preprocess GSM8K dataset for CAL training."""
    def preprocess_fn(example):
        return {
            'prompt': format_gsm8k_prompt(example),
            'question': example['question'],
            'answer': extract_gsm8k_answer(example),
        }
    return dataset.map(preprocess_fn)


# ============================================================================
# Reward function (fallback if CAL is disabled)
# ============================================================================

def simple_correctness_reward(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """Simple correctness-based reward (1.0 if correct, 0.0 otherwise)."""
    answers = kwargs.get('answer', [])
    rewards = []
    for completion, answer in zip(completions, answers):
        # Simple substring check (replace with proper math grading)
        if answer in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ============================================================================
# Main training function
# ============================================================================

def main(argv):
    del argv  # Unused
    
    log.info("=" * 80)
    log.info("CAL-GRPO Training on GSM8K")
    log.info("=" * 80)
    
    # Set random seed
    jax.random.PRNGKey(FLAGS.seed)
    
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # ========================================================================
    # 1. Load and preprocess dataset
    # ========================================================================
    log.info(f"Loading dataset: {FLAGS.dataset_name}/{FLAGS.dataset_config}")
    dataset = load_dataset(FLAGS.dataset_name, FLAGS.dataset_config)
    
    train_dataset = preprocess_gsm8k_dataset(dataset['train'])
    test_dataset = preprocess_gsm8k_dataset(dataset['test'])
    
    if FLAGS.max_train_samples:
        train_dataset = train_dataset.select(range(FLAGS.max_train_samples))
    if FLAGS.max_eval_samples:
        test_dataset = test_dataset.select(range(FLAGS.max_eval_samples))
    
    log.info(f"Train samples: {len(train_dataset)}")
    log.info(f"Test samples: {len(test_dataset)}")
    
    # ========================================================================
    # 2. Initialize RLCluster (model, tokenizer, etc.)
    # ========================================================================
    log.info(f"Initializing model: {FLAGS.model_id}")
    
    # Note: This is a simplified example. In practice, you'll need to:
    # - Create proper model config
    # - Set up mesh for distributed training
    # - Configure LoRA adapters if needed
    # - Initialize reference model
    
    # Example (adjust based on actual Tunix API):
    # rl_cluster = rl_cluster_lib.RLCluster(
    #     actor_model_config=...,
    #     reference_model_config=...,
    #     tokenizer_config=...,
    # )
    
    log.warning(
        "RLCluster initialization not shown in this template. "
        "Please refer to Tunix examples (e.g., examples/grpo_demo.ipynb) "
        "for proper cluster setup."
    )
    
    # ========================================================================
    # 3. Configure CAL-GRPO
    # ========================================================================
    log.info("Configuring CAL-GRPO")
    
    cal_config = CALGRPOConfig(
        # GRPO parameters
        num_generations=FLAGS.num_generations,
        num_iterations=FLAGS.num_iterations,
        beta=FLAGS.beta,
        epsilon=FLAGS.epsilon,
        loss_algo=FLAGS.loss_algo,
        # CAL parameters
        use_cal_credit=FLAGS.use_cal_credit,
        cal_model_name=FLAGS.cal_model_name,
        cal_few_shot_path=FLAGS.cal_few_shot_path,
        negative_reward=FLAGS.negative_reward,
        max_error_span_tokens=FLAGS.max_error_span_tokens,
    )
    
    log.info(f"CAL credit enabled: {cal_config.use_cal_credit}")
    log.info(f"Number of generations per prompt: {cal_config.num_generations}")
    
    # ========================================================================
    # 4. Create CAL-GRPO Learner
    # ========================================================================
    log.info("Creating CAL-GRPO Learner")
    
    # Note: Replace with actual rl_cluster once initialized
    # learner = CALGRPOLearner(
    #     rl_cluster=rl_cluster,
    #     reward_fns=simple_correctness_reward,  # Fallback reward
    #     grpo_config=cal_config,
    # )
    
    log.warning(
        "CALGRPOLearner instantiation not shown. "
        "Complete the RLCluster setup first, then create the learner."
    )
    
    # ========================================================================
    # 5. Train
    # ========================================================================
    log.info("Starting training")
    log.info(f"Training steps: {FLAGS.num_train_steps}")
    log.info(f"Batch size: {FLAGS.batch_size}")
    log.info(f"Learning rate: {FLAGS.learning_rate}")
    
    # learner.train(
    #     train_ds=train_dataset,
    #     eval_ds=test_dataset,
    #     num_steps=FLAGS.num_train_steps,
    # )
    
    log.info("Training complete!")
    log.info(f"Results saved to: {FLAGS.output_dir}")
    
    # ========================================================================
    # 6. Evaluate
    # ========================================================================
    log.info("Running final evaluation")
    
    # results = learner.evaluate(test_dataset)
    # log.info(f"Final test accuracy: {results.get('accuracy', 'N/A')}")


if __name__ == '__main__':
    app.run(main)
