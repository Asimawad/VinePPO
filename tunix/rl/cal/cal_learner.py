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

"""CAL-enhanced GRPO Learner for fine-grained credit assignment."""

import dataclasses
import logging
from typing import Any, Dict, List, Sequence

import jax.numpy as jnp

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.cal.cal_oracle import CALOracle
from tunix.rl.cal.cal_helpers import construct_sparse_reward_batch

log = logging.getLogger(__name__)

# Type aliases from base learner
MetricFn = Any  # Should match base learner's type
RewardFn = Any  # Should match base learner's type


@dataclasses.dataclass(slots=True, kw_only=True)
class CALGRPOConfig(GRPOConfig):
  """Extended GRPO configuration with CAL-specific parameters.
  
  Additional CAL parameters:
    use_cal_credit: Enable CAL-based fine-grained credit assignment
    cal_model_name: Model name for CAL oracle (e.g., "gpt-4", "gpt-3.5-turbo", "gemini-1.5-pro")
    cal_few_shot_path: Path to JSON file with few-shot examples
    negative_reward: Reward value for incorrect responses
    max_error_span_tokens: Maximum token span length for error segments
    api_provider: Which API to use ("openai" or "gemini")
    api_key_env: Environment variable for API key (optional, defaults based on provider)
  """
  
  use_cal_credit: bool = True
  cal_model_name: str = "gpt-4"
  cal_few_shot_path: str = "cal_few_shot_examples.json"
  negative_reward: float = -1.0
  max_error_span_tokens: int = 64
  api_provider: str = "openai"  # "openai" or "gemini"
  api_key_env: str = None  # Optional, defaults to OPENAI_API_KEY or GEMINI_API_KEY


class CALGRPOLearner(GRPOLearner):
  """GRPO Learner with CAL-based fine-grained credit assignment.
  
  This learner extends GRPO with Credit Assignment LLM (CAL) logic:
  1. Uses CAL oracle to identify error segments in incorrect responses
  2. Maps error segments to token indices
  3. Constructs sparse reward tensors (negative reward on error tokens)
  4. Uses sparse rewards directly as advantages (critic-free)
  
  The key insight: instead of using coarse, sequence-level feedback,
  we apply negative rewards only to the specific tokens that caused
  the error, creating a cleaner, lower-variance learning signal.
  """
  
  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      reward_fns: RewardFn | List[RewardFn],
      grpo_config: CALGRPOConfig,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initialize CAL-GRPO Learner.
    
    Args:
      rl_cluster: RL cluster containing actor, reference models
      reward_fns: Reward functions (may be overridden by CAL oracle)
      grpo_config: CAL-enhanced GRPO configuration
      metric_fns: Optional metric functions for logging
      data_shuffle_seed: Random seed for data shuffling
    """
    super().__init__(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        grpo_config=grpo_config,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
    
    self.cal_config = grpo_config
    
    # Initialize CAL oracle if enabled
    if self.cal_config.use_cal_credit:
      log.info("[CALGRPOLearner] Initializing CAL oracle")
      self.cal_oracle = CALOracle(
          cal_model_name=self.cal_config.cal_model_name,
          few_shot_path=self.cal_config.cal_few_shot_path,
          negative_reward=self.cal_config.negative_reward,
          api_provider=self.cal_config.api_provider,
          api_key_env=self.cal_config.api_key_env,
      )
      log.info(f"[CALGRPOLearner] CAL oracle initialized successfully with {self.cal_config.api_provider}")
    else:
      self.cal_oracle = None
      log.info("[CALGRPOLearner] CAL credit disabled; using standard GRPO")
  
  def _compute_cal_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      **dataset_fields: Any,
  ) -> tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute rewards using CAL oracle.
    
    Args:
      prompts: List of input prompts/questions
      completions: List of model-generated completions
      **dataset_fields: Additional dataset fields (e.g., 'answer')
    
    Returns:
      rewards: JAX array of scalar rewards [batch_size]
      metadata: Dict containing 'cal_infos' with error segment data
    """
    # Extract ground truth references from dataset fields
    references = dataset_fields.get('answer')
    if references is None:
      log.warning(
          "[CALGRPOLearner] No 'answer' field in dataset; "
          "falling back to standard reward functions"
      )
      # Fall back to standard reward computation
      return self._compute_standard_rewards(prompts, completions, **dataset_fields)
    
    # Call CAL oracle
    rewards, cal_infos = self.cal_oracle.get_reward(
        inputs=prompts,
        responses=completions,
        references=references,
    )
    
    log.info(
        f"[CALGRPOLearner] CAL oracle processed {len(prompts)} responses, "
        f"mean reward: {rewards.mean():.4f}"
    )
    
    return rewards, {'cal_infos': cal_infos}
  
  def _compute_standard_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      **dataset_fields: Any,
  ) -> tuple[jnp.ndarray, Dict[str, Any]]:
    """Fallback to standard reward computation (no CAL)."""
    # Call original reward functions
    total_rewards = jnp.zeros(len(prompts), dtype=jnp.float32)
    
    reward_fns = self.reward_fns if isinstance(self.reward_fns, list) else [self.reward_fns]
    for reward_fn in reward_fns:
      rewards = reward_fn(prompts, completions, **dataset_fields)
      total_rewards += jnp.array(rewards, dtype=jnp.float32)
    
    return total_rewards, {}
  
  def _compute_sparse_advantages(
      self,
      completions: List[str],
      token_ids_list: List[List[int]],
      cal_infos: List[Dict[str, Any]],
  ) -> jnp.ndarray:
    """Compute sparse advantages from CAL error segments.
    
    Args:
      completions: List of completion text strings
      token_ids_list: List of token ID sequences
      cal_infos: List of CAL metadata dicts with 'cal_error_segment'
    
    Returns:
      advantages: JAX array of shape [batch_size, max_seq_len] with sparse advantages
    """
    # Extract error segments from CAL info
    error_segments = [info.get('cal_error_segment', '') for info in cal_infos]
    
    # Construct sparse reward arrays (which are used directly as advantages)
    advantages = construct_sparse_reward_batch(
        response_texts=completions,
        response_ids_list=token_ids_list,
        error_segments=error_segments,
        tokenizer=self.rl_cluster.tokenizer,
        negative_reward=self.cal_config.negative_reward,
        max_span_tokens=self.cal_config.max_error_span_tokens,
    )
    
    num_nonzero = jnp.count_nonzero(advantages)
    log.info(
        f"[CALGRPOLearner] Constructed sparse advantages: "
        f"{num_nonzero} non-zero elements out of {advantages.size} total"
    )
    
    return advantages
  
  # ========================================================================
  # Override key methods from parent GRPOLearner
  # ========================================================================
  
  # Note: The exact method signatures will depend on the base GRPOLearner implementation.
  # This is a template that shows the integration points. You may need to adjust
  # based on the actual parent class API.
  
  # Example override (adjust based on actual parent class):
  # def _update_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
  #   """Override update step to use CAL rewards and advantages."""
  #   if not self.cal_config.use_cal_credit:
  #     return super()._update_step(batch)
  #   
  #   # Extract data from batch
  #   prompts = batch['prompts']
  #   completions = batch['completions']
  #   token_ids = batch['token_ids']
  #   
  #   # Compute CAL rewards
  #   rewards, metadata = self._compute_cal_rewards(
  #       prompts, completions, **batch
  #   )
  #   
  #   if 'cal_infos' in metadata:
  #     # Use sparse CAL advantages
  #     advantages = self._compute_sparse_advantages(
  #         completions, token_ids, metadata['cal_infos']
  #     )
  #   else:
  #     # Fall back to standard GRPO advantages
  #     advantages = self._compute_standard_advantages(rewards, ...)
  #   
  #   # Continue with standard GRPO update using custom advantages
  #   return self._apply_policy_update(batch, advantages)

