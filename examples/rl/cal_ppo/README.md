# CAL-GRPO: Fine-Grained Credit Assignment for LLM RL Training

This directory contains an implementation of **Fine-Grained Credit PPO (FGC-PPO)** using a **Credit Assignment LLM (CAL)** to provide token-level reward signals for more efficient reinforcement learning.

## Overview

### The Problem

Standard RL algorithms like PPO and GRPO use coarse, sequence-level feedback. They treat an entire generated answer as uniformly "good" or "bad" by assigning it a single score. If a long, mostly correct answer has one small error, all the correct parts are penalized along with the error, creating a noisy, high-variance learning signal.

### The Solution

CAL-GRPO uses a powerful LLM (via Gemini API) to identify the **specific error segment** in incorrect responses. It then applies negative rewards only to the tokens in that error segment, creating a sparse, precise reward signal that leads to:

- Lower variance advantages
- Faster convergence
- More stable training
- Better final accuracy

## Algorithm

For each training step:

1. **Rollout**: Generate N candidate solutions per question
2. **Discriminate**: Check correctness against ground truth
3. **CAL Query**: For incorrect answers, query CAL to identify the specific error segment
4. **Token Mapping**: Map the error segment text to token indices
5. **Reward Construction**: Build sparse reward tensor (negative reward on error tokens, zero elsewhere)
6. **GRPO Update**: Use sparse rewards directly as advantages (critic-free)

## Quick Start

### Prerequisites

1. **Install Tunix**:
   ```bash
   pip install google-tunix
   ```

2. **Set up Gemini API key**:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```
   Get your API key from: https://makersuite.google.com/app/apikey

3. **Install dependencies**:
   ```bash
   pip install datasets python-dotenv google-generativeai
   ```

### Run Training

```bash
# Quick test (10 samples)
MAX_TRAIN_SAMPLES=10 NUM_TRAIN_STEPS=5 bash examples/rl/cal_ppo/run_cal_gsm8k.sh

# Full training run
bash examples/rl/cal_ppo/run_cal_gsm8k.sh
```

### Configuration

You can customize training via environment variables:

```bash
# Model settings
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"  # Larger model

# CAL settings
USE_CAL=true                    # Enable/disable CAL
CAL_MODEL="gemini-1.5-pro-latest"
NEGATIVE_REWARD=-2.0            # Stronger penalty

# Training settings
NUM_GENERATIONS=8               # More samples per prompt
BATCH_SIZE=4
NUM_TRAIN_STEPS=2000
LEARNING_RATE=1e-6

bash examples/rl/cal_ppo/run_cal_gsm8k.sh
```

## File Structure

```
examples/rl/cal_ppo/
├── README.md                      # This file
├── cal_gsm8k.py                   # Main training script
├── run_cal_gsm8k.sh               # Launch script
└── cal_few_shot_examples.json    # Few-shot examples for CAL
```

## Implementation Details

### CAL Oracle

The CAL oracle (`tunix.rl.cal.CALOracle`) uses the Gemini API to identify error segments. It:

1. Takes a question, correct solution, and incorrect solution
2. Sends a few-shot prompt to Gemini
3. Receives back the specific sentence/phrase that represents the error
4. Caches results to avoid redundant API calls

### Token Mapping

The token mapping function (`tunix.rl.cal.cal_helpers.map_segment_to_token_indices`) converts the error segment text to token indices:

1. Normalizes both the response and error segment for fuzzy matching
2. Finds the segment in the response text
3. Maps character positions to token indices using the tokenizer
4. Validates the span length (rejects spans that are too long)

### Sparse Rewards

The sparse reward construction creates a tensor where:
- Most tokens have reward = 0.0
- Tokens in the error segment have reward = `negative_reward / num_error_tokens`
- These sparse rewards are used directly as advantages (critic-free)

## Comparison: CAL-GRPO vs Standard GRPO

| Feature | Standard GRPO | CAL-GRPO |
|---------|---------------|----------|
| **Reward Granularity** | Sequence-level | Token-level |
| **Error Localization** | No | Yes (via CAL) |
| **Advantage Variance** | High | Low |
| **Sample Efficiency** | Moderate | High |
| **External Dependency** | None | Gemini API |
| **Best For** | General RL tasks | Tasks with identifiable errors |

## Tips for Best Results

1. **Use enough samples**: Set `NUM_GENERATIONS=4` or higher to get both correct and incorrect responses
2. **Tune negative reward**: Start with `-1.0`, increase magnitude for stronger credit assignment
3. **Monitor CAL calls**: Check logs to ensure CAL is being called and returning valid segments
4. **Validate token mapping**: Inspect a few examples to ensure segments are mapped correctly
5. **Compare baselines**: Run with `USE_CAL=false` to measure CAL's contribution

## Troubleshooting

### Issue: "GEMINI_API_KEY not set"

**Solution**: Export your API key:
```bash
export GEMINI_API_KEY='your-api-key'
```

### Issue: CAL returns empty segments

**Possible causes**:
- API timeout (increase timeout in `cal_oracle.py`)
- Poor few-shot examples (improve `cal_few_shot_examples.json`)
- Gemini model overwhelmed (reduce batch size)

**Solution**: Check logs for API errors, validate few-shot examples match your domain

### Issue: Token mapping fails frequently

**Possible causes**:
- CAL returns paraphrased text instead of exact quotes
- Tokenizer mismatch between CAL and policy model

**Solution**: 
- Improve CAL prompt to emphasize exact text copying
- Use fuzzy matching (already implemented in `_normalize_for_match`)

### Issue: Training is slow

**Possible causes**:
- Too many CAL API calls (every incorrect response)
- Large batch size with many generations

**Solution**:
- Enable caching (already enabled by default)
- Use local LLM for CAL instead of API
- Reduce `NUM_GENERATIONS`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cal_grpo_2025,
  title={Fine-Grained Credit Assignment for LLM RL Training},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/google/tunix}},
}
```

## Related Work

- **GRPO**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **DeepSpeed-Math**: Similar credit assignment approaches
- **Gemini**: [Google's Large Language Model](https://deepmind.google/technologies/gemini/)

## License

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0.
