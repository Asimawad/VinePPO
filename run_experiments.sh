#!/bin/bash
# Quick launcher for baseline vs CAL experiments

set -e

# Configuration
NUM_SAMPLES=500       # Default: 100 samples
NUM_GENS=8          # Default: 4 generations per prompt

echo ""
echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Generations: $NUM_GENS"
echo ""

# Activate environment
source .venv/bin/activate

# Run baseline
echo "=========================================="
echo "1. Running BASELINE GRPO"
echo "=========================================="
python train_cal.py \
  --num-samples $NUM_SAMPLES \
  --num-generations $NUM_GENS \
  --cal-model gpt-3.5-turbo \
  --output-dir /tmp/cal_experiments \
  --num-tpus 2 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --max-grad-norm 0.1 \
  --beta 0.04 \
  --epsilon 0.2 \
  --num-iterations 1 \
  --max-prompt-length 1024 \
  --max-generation-steps 1024 \
  --temperature 0.3 \
  --num-test-samples 50 \
  --api-provider openai \
  --negative-reward -1.0 \
  --max-error-span 64 \
  --top-k 50 \


# Run CAL
echo "=========================================="
echo "2. Running CAL-GRPO"
echo "=========================================="
python train_cal.py \
  --use-cal \
  --num-samples $NUM_SAMPLES \
  --num-generations $NUM_GENS \
  --cal-model gpt-3.5-turbo \
  --output-dir /tmp/cal_experiments \
  --num-tpus 2 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --max-grad-norm 0.1 \
  --beta 0.04 \
  --epsilon 0.2 \
  --num-iterations 1 \
  --max-prompt-length 1024 \
  --max-generation-steps 1024 \
  --temperature 0.3 \
  --num-test-samples 50 \
  --api-provider openai \
  --negative-reward -1.0 \
  --max-error-span 64 \
  --top-k 50 \






# Compare results
echo "Comparing results..."
python compare_results.py --exp-dir /tmp/cal_experiments

echo ""
echo "Full results:"
echo "  Baseline: /tmp/cal_experiments/baseline/"
echo "  CAL:      /tmp/cal_experiments/cal/"
echo ""
echo "View training curves:"
echo "  tensorboard --logdir /tmp/cal_experiments/"
echo ""
echo "=========================================="

