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

"""Helper functions for CAL token mapping and sparse reward construction."""

import logging
import re
from typing import List, Optional, Tuple

import jax.numpy as jnp

log = logging.getLogger(__name__)


def _normalize_for_match(text: str) -> str:
    """Normalize text for fuzzy matching (lowercase, remove punctuation)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def map_segment_to_token_indices(
    full_response_text: str,
    error_segment: str,
    tokenizer,
    max_span_tokens: int = 64,
) -> Tuple[Optional[int], Optional[int]]:
    """Map error segment text to token indices in the response.
    
    This function performs fuzzy matching to handle minor formatting differences
    between the CAL model's output and the actual tokenized response.
    
    Args:
        full_response_text: Complete model response text
        error_segment: Error segment substring to locate
        tokenizer: HuggingFace tokenizer (must have fast=True)
        max_span_tokens: Maximum allowed span length (reject if exceeded)
    
    Returns:
        (start_idx, end_idx): Token indices of error segment, or (None, None) if mapping fails
    """
    if not error_segment:
        return None, None
    
    # 1. Normalize both texts for fuzzy matching
    norm_response = _normalize_for_match(full_response_text)
    norm_segment = _normalize_for_match(error_segment)
    
    # 2. Find normalized segment in normalized response
    char_start_norm = norm_response.find(norm_segment)
    if char_start_norm == -1:
        log.warning(
            f"[CALHelpers] Could not find normalized error segment. "
            f"Original: '{error_segment[:50]}...'"
        )
        return None, None
    
    # 3. Map normalized position back to original text
    # Count non-space characters before the match
    non_space_chars_before = len(norm_response[:char_start_norm].replace(' ', ''))
    
    original_char_start = -1
    count = 0
    for i, char in enumerate(full_response_text):
        if not char.isspace():
            count += 1
        if count > non_space_chars_before:
            original_char_start = i
            break
    
    if original_char_start == -1:
        log.warning("[CALHelpers] Failed to map normalized position to original text")
        return None, None
    
    char_end = original_char_start + len(error_segment)
    
    # 4. Tokenize prefixes to get token indices
    tokens_before = tokenizer.encode(
        full_response_text[:original_char_start],
        add_special_tokens=False
    )
    token_start_idx = len(tokens_before)
    
    tokens_up_to_end = tokenizer.encode(
        full_response_text[:char_end],
        add_special_tokens=False
    )
    token_end_idx = len(tokens_up_to_end) - 1
    
    if token_start_idx > token_end_idx:
        log.warning(
            f"[CALHelpers] Invalid token span: start={token_start_idx}, end={token_end_idx}"
        )
        return None, None
    
    # 5. Enforce maximum span length
    span_len = token_end_idx - token_start_idx + 1
    if span_len < 1 or span_len > max_span_tokens:
        log.warning(
            f"[CALHelpers] Span length {span_len} outside valid range [1, {max_span_tokens}]"
        )
        return None, None
    
    return token_start_idx, token_end_idx


def looks_like_junk(segment: str) -> bool:
    """Check if error segment looks like junk/control characters.
    
    Args:
        segment: Error segment text
    
    Returns:
        True if segment appears to be junk, False otherwise
    """
    if not segment:
        return True
    if len(segment) > 256:
        return True
    # Check for HTML-like tags or template markers
    if re.search(r"[<>]|\|im_", segment):
        return True
    return False


def construct_sparse_reward_array(
    response_ids: List[int],
    error_start_idx: Optional[int],
    error_end_idx: Optional[int],
    negative_reward: float,
) -> jnp.ndarray:
    """Build a sparse reward array for a single sequence.
    
    Args:
        response_ids: Token IDs of the response
        error_start_idx: Start token index of error segment (or None)
        error_end_idx: End token index of error segment (or None)
        negative_reward: Scalar reward to distribute over error tokens
    
    Returns:
        rewards: JAX array of shape [seq_len] with negative rewards on error span
    """
    seq_len = len(response_ids)
    rewards = jnp.zeros(seq_len, dtype=jnp.float32)
    
    if error_start_idx is not None and error_end_idx is not None:
        num_tokens = error_end_idx - error_start_idx + 1
        if num_tokens > 0:
            # Distribute the negative reward evenly across error tokens
            reward_per_token = negative_reward / num_tokens
            rewards = rewards.at[error_start_idx:error_end_idx + 1].set(reward_per_token)
    
    return rewards


def construct_sparse_reward_batch(
    response_texts: List[str],
    response_ids_list: List[List[int]],
    error_segments: List[str],
    tokenizer,
    negative_reward: float,
    max_span_tokens: int = 64,
) -> jnp.ndarray:
    """Build sparse reward arrays for a batch of sequences.
    
    Args:
        response_texts: List of response text strings
        response_ids_list: List of token ID sequences
        error_segments: List of error segment strings
        tokenizer: HuggingFace tokenizer
        negative_reward: Scalar reward for incorrect responses
        max_span_tokens: Maximum allowed span length
    
    Returns:
        rewards: JAX array of shape [batch_size, max_seq_len] with sparse rewards
    """
    batch_size = len(response_texts)
    reward_arrays = []
    
    for i in range(batch_size):
        response_text = response_texts[i]
        response_ids = response_ids_list[i]
        error_segment = error_segments[i]
        
        if error_segment and not looks_like_junk(error_segment):
            # Map segment to tokens
            start_idx, end_idx = map_segment_to_token_indices(
                response_text,
                error_segment,
                tokenizer,
                max_span_tokens=max_span_tokens,
            )
            # Construct sparse reward array
            rewards = construct_sparse_reward_array(
                response_ids,
                start_idx,
                end_idx,
                negative_reward,
            )
        else:
            # No error segment or junk: zero rewards
            rewards = jnp.zeros(len(response_ids), dtype=jnp.float32)
        
        reward_arrays.append(rewards)
    
    # Pad to same length for batching
    max_len = max(len(r) for r in reward_arrays)
    padded_rewards = jnp.stack([
        jnp.pad(r, (0, max_len - len(r)), constant_values=0.0)
        for r in reward_arrays
    ])
    
    return padded_rewards

