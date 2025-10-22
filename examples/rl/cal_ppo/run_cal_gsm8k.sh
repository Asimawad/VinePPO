#!/bin/bash
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

# CAL-GRPO Training Script for GSM8K
# This script launches fine-grained credit assignment RL training using CAL.

set -e

# ============================================================================
# Configuration (can be overridden via environment variables)
# ============================================================================

# Model settings
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-0.5b}"

# CAL settings
USE_CAL="${USE_CAL:-true}"
CAL_MODEL="${CAL_MODEL:-gemini-1.5-pro-latest}"
CAL_FEW_SHOT="${CAL_FEW_SHOT:-examples/rl/cal_ppo/cal_few_shot_examples.json}"
NEGATIVE_REWARD="${NEGATIVE_REWARD:--1.0}"
MAX_ERROR_SPAN="${MAX_ERROR_SPAN:-64}"

# Training settings
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"

# Data settings
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-256}"

# System settings
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/cal_grpo_gsm8k}"
SEED="${SEED:-42}"

# ============================================================================
# Environment checks
# ============================================================================

echo "========================================"
echo "CAL-GRPO Training on GSM8K"
echo "========================================"
echo ""

# Check for Gemini API key
if [ -z "${GEMINI_API_KEY}" ]; then
    echo "WARNING: GEMINI_API_KEY not set!"
    echo "CAL oracle will fail if use_cal_credit=true."
    echo "Please set: export GEMINI_API_KEY='your-api-key'"
    echo ""
fi

# Check for JAX installation
if ! python3 -c "import jax" 2>/dev/null; then
    echo "ERROR: JAX not installed!"
    echo "Please install JAX: pip install jax"
    exit 1
fi

# Check for Tunix installation
if ! python3 -c "import tunix" 2>/dev/null; then
    echo "ERROR: Tunix not installed!"
    echo "Please install Tunix: pip install google-tunix"
    exit 1
fi

echo "Environment checks passed."
echo ""

# ============================================================================
# Display configuration
# ============================================================================

echo "Configuration:"
echo "  Model: ${MODEL_ID}"
echo "  CAL Enabled: ${USE_CAL}"
echo "  CAL Model: ${CAL_MODEL}"
echo "  Num Generations: ${NUM_GENERATIONS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Training Steps: ${NUM_TRAIN_STEPS}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""

# ============================================================================
# Launch training
# ============================================================================

echo "Starting training..."
echo ""

python3 examples/rl/cal_ppo/cal_gsm8k.py \
    --model_id="${MODEL_ID}" \
    --model_name="${MODEL_NAME}" \
    --use_cal_credit="${USE_CAL}" \
    --cal_model_name="${CAL_MODEL}" \
    --cal_few_shot_path="${CAL_FEW_SHOT}" \
    --negative_reward="${NEGATIVE_REWARD}" \
    --max_error_span_tokens="${MAX_ERROR_SPAN}" \
    --num_generations="${NUM_GENERATIONS}" \
    --batch_size="${BATCH_SIZE}" \
    --num_train_steps="${NUM_TRAIN_STEPS}" \
    --learning_rate="${LEARNING_RATE}" \
    ${MAX_TRAIN_SAMPLES:+--max_train_samples="${MAX_TRAIN_SAMPLES}"} \
    --max_eval_samples="${MAX_EVAL_SAMPLES}" \
    --output_dir="${OUTPUT_DIR}" \
    --seed="${SEED}"

echo ""
echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"
