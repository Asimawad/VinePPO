# Getting Started with CAL-GRPO

This guide will help you quickly start using Fine-Grained Credit Assignment with Tunix.

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
cd /home/asim/tunix

# Install Tunix with dev dependencies
pip install -e ".[dev]"

# Install CAL-specific dependencies
pip install google-generativeai python-dotenv datasets
```

### 2. Set Up Gemini API Key

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey), then:

```bash
# Option A: Export directly
export GEMINI_API_KEY="your_api_key_here"

# Option B: Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 3. Run Unit Tests (Verify Installation)

```bash
# From Tunix root directory
python -m pytest tests/rl/cal/ -v

# Expected output: All tests pass
```

## Understanding the Code

### The 3 Core Files

1. **`tunix/rl/cal/cal_oracle.py`** - The "Judge" LLM
   - Calls Gemini API to identify error segments
   - Returns sparse rewards: 0.0 for correct, -1.0 for error tokens

2. **`tunix/rl/cal/cal_helpers.py`** - Text-to-Token Mapping
   - Maps error segment text to token indices
   - Constructs sparse reward tensors

3. **`tunix/rl/cal/cal_learner.py`** - The Training Loop
   - Extends GRPO with CAL-based advantages
   - Uses sparse rewards directly (no critic)

### Data Flow

```
Question → Model generates N responses → Check correctness
                                              ↓
                   ┌──────────── Correct? ────────────┐
                   ↓                                   ↓
                 YES                                  NO
                   ↓                                   ↓
           reward = 0.0                    Query CAL oracle
                                                     ↓
                                          Get error segment
                                                     ↓
                                          Map to token indices
                                                     ↓
                                     Sparse reward: [-1,-1,0,0,0,...]
                                                     ↓
                   └────────────────┬────────────────┘
                                    ↓
                         Use as advantages in PPO update
```

## Example: Training on GSM8K

### Minimal Example (10 Questions)

Create `test_cal.py`:

```python
from tunix.rl.cal import CALGRPOConfig, CALGRPOLearner

# Configure
config = CALGRPOConfig(
    num_generations=2,          # 2 samples per question
    use_cal_credit=True,
    negative_reward=-1.0,
    cal_few_shot_path="examples/rl/cal_ppo/cal_few_shot_examples.json"
)

# TODO: Initialize model and RLCluster
# model = ...
# rl_cluster = ...

# Create learner
learner = CALGRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=lambda p,c,**kw: [0.0]*len(p),  # Dummy (CAL overrides)
    grpo_config=config,
)

# Train
learner.train(train_ds=train_data, eval_ds=eval_data)
```

### Compare CAL vs Baseline

```python
# Baseline: Standard GRPO (coarse rewards)
config_baseline = CALGRPOConfig(use_cal_credit=False)
learner_baseline = CALGRPOLearner(..., grpo_config=config_baseline)

# CAL: Fine-grained rewards
config_cal = CALGRPOConfig(use_cal_credit=True)
learner_cal = CALGRPOLearner(..., grpo_config=config_cal)

# Train both and compare accuracy
```

## Customizing for Your Domain

### 1. Create Custom Few-Shot Examples

Edit `cal_few_shot_examples.json`:

```json
[
  {
    "question": "Your domain question",
    "correct_solution": "Step-by-step correct answer",
    "incorrect_solution": "Answer with an error",
    "error_segment": "The exact text of the error"
  }
]
```

**Tips**:
- Use 3-5 examples
- Cover common error types
- Ensure error_segment is **exact text** from incorrect_solution

### 2. Adjust Hyperparameters

```python
config = CALGRPOConfig(
    # Generation
    num_generations=4,        # More = better group advantage
    
    # Rewards
    negative_reward=-2.0,     # Stronger penalty
    max_error_span_tokens=32, # Shorter error segments only
    
    # GRPO
    beta=0.04,                # KL penalty (0.01-0.1)
    epsilon=0.2,              # Clipping (0.1-0.3)
)
```

### 3. Use Local CAL Model (No API Cost)

Instead of Gemini, you can use a local model:

```python
# TODO: Implement local CAL oracle
from tunix.rl.cal import CALOracle

class LocalCALOracle(CALOracle):
    def __init__(self, local_model_path, ...):
        # Load local LLM (e.g., Qwen2.5-1.5B)
        self.local_model = load_model(local_model_path)
    
    def get_error_segment(self, question, correct, incorrect):
        # Use local model instead of Gemini API
        prompt = self._build_prompt(question, correct, incorrect)
        segment = self.local_model.generate(prompt)
        return segment
```

## Debugging Tips

### Check CAL Oracle is Working

Add logging to see CAL calls:

```python
import logging
logging.basicConfig(level=logging.INFO)

# You'll see:
# [CALOracle] Received segment: 'error text here'
# [CALGRPOLearner] CAL oracle processed 32 responses, mean reward: -0.45
```

### Visualize Sparse Rewards

```python
def log_rewards(rewards, response_ids):
    """Print reward pattern."""
    nonzero = jnp.count_nonzero(rewards)
    print(f"Sparse reward: {nonzero}/{len(rewards)} tokens penalized")
    print(f"Pattern: {rewards}")
```

### Common Issues

| Problem | Solution |
|---------|----------|
| "Could not find error segment" | Improve few-shot examples; CAL must return exact text |
| API rate limits | Reduce batch size or cache more aggressively |
| Rewards all zero | Check correctness checker; might be marking all as correct |
| Loss not decreasing | Increase `negative_reward` or adjust learning rate |

## Performance Tips

1. **Batch CAL calls** (future): Queue multiple oracle queries
2. **Cache aggressively**: Oracle already caches by (question, response)
3. **Precompute for eval**: Cache all eval set error segments
4. **Use local model**: Avoid API latency/cost

## Next Steps

1. ✓ Run unit tests
2. ☐ Complete model setup in `cal_gsm8k.py`
3. ☐ Run small test (10 examples)
4. ☐ Compare CAL vs baseline
5. ☐ Scale to full dataset
6. ☐ Publish results!

## Resources

- **Full Documentation**: `examples/rl/cal_ppo/README.md`
- **Migration Guide**: `/home/asim/tunix/CAL_MIGRATION_SUMMARY.md`
- **Unit Tests**: `tests/rl/cal/` (great usage examples!)
- **Tunix Docs**: https://tunix.readthedocs.io/

## Questions?

Check:
1. Unit tests for working examples
2. README for detailed explanations
3. Code comments for implementation details

Happy training with fine-grained credit assignment! 🎯

