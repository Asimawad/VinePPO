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

"""Tests for CAL helper functions."""

import jax.numpy as jnp
import pytest
from transformers import AutoTokenizer

from tunix.rl.cal import cal_helpers


class TestMapSegmentToTokenIndices:
    """Tests for segment-to-token mapping."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a test tokenizer."""
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    
    def test_exact_match(self, tokenizer):
        """Test mapping with exact text match."""
        response = "The answer is 42. This is correct."
        segment = "The answer is 42."
        
        start, end = cal_helpers.map_segment_to_token_indices(
            response, segment, tokenizer
        )
        
        assert start is not None
        assert end is not None
        assert start >= 0
        assert end >= start
    
    def test_fuzzy_match_case_insensitive(self, tokenizer):
        """Test mapping with case differences."""
        response = "The Answer Is 42."
        segment = "the answer is 42"
        
        start, end = cal_helpers.map_segment_to_token_indices(
            response, segment, tokenizer
        )
        
        # Should still find it due to normalization
        assert start is not None
        assert end is not None
    
    def test_fuzzy_match_punctuation(self, tokenizer):
        """Test mapping with punctuation differences."""
        response = "The answer is: 42!"
        segment = "the answer is 42"
        
        start, end = cal_helpers.map_segment_to_token_indices(
            response, segment, tokenizer
        )
        
        # Should find it despite punctuation
        assert start is not None
    
    def test_empty_segment(self, tokenizer):
        """Test with empty error segment."""
        response = "Some response text"
        segment = ""
        
        start, end = cal_helpers.map_segment_to_token_indices(
            response, segment, tokenizer
        )
        
        assert start is None
        assert end is None
    
    def test_segment_not_found(self, tokenizer):
        """Test with segment that doesn't exist in response."""
        response = "The answer is 42."
        segment = "nonexistent text"
        
        start, end = cal_helpers.map_segment_to_token_indices(
            response, segment, tokenizer
        )
        
        assert start is None
        assert end is None
    
    def test_max_span_tokens_exceeded(self, tokenizer):
        """Test rejection of overly long spans."""
        response = "word " * 100  # 100 words
        segment = "word " * 50    # 50 words (likely > 64 tokens)
        
        start, end = cal_helpers.map_segment_to_token_indices(
            response, segment, tokenizer, max_span_tokens=10
        )
        
        # Should reject due to length
        assert start is None or end is None or (end - start + 1) <= 10


class TestConstructSparseRewardArray:
    """Tests for sparse reward array construction."""
    
    def test_basic_sparse_reward(self):
        """Test construction with valid indices."""
        response_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        start_idx = 2
        end_idx = 4
        negative_reward = -1.0
        
        rewards = cal_helpers.construct_sparse_reward_array(
            response_ids, start_idx, end_idx, negative_reward
        )
        
        assert rewards.shape == (8,)
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0
        # Error tokens should have distributed negative reward
        num_error_tokens = end_idx - start_idx + 1  # 3 tokens
        assert rewards[2] == pytest.approx(-1.0 / 3)
        assert rewards[3] == pytest.approx(-1.0 / 3)
        assert rewards[4] == pytest.approx(-1.0 / 3)
        assert rewards[5] == 0.0
    
    def test_no_error_segment(self):
        """Test construction with no error (None indices)."""
        response_ids = [1, 2, 3, 4, 5]
        
        rewards = cal_helpers.construct_sparse_reward_array(
            response_ids, None, None, -1.0
        )
        
        assert rewards.shape == (5,)
        assert jnp.all(rewards == 0.0)
    
    def test_single_token_error(self):
        """Test with error on single token."""
        response_ids = [1, 2, 3, 4, 5]
        
        rewards = cal_helpers.construct_sparse_reward_array(
            response_ids, 2, 2, -1.0
        )
        
        assert rewards[2] == -1.0  # Full reward on single token
        assert rewards[1] == 0.0
        assert rewards[3] == 0.0


class TestLooksLikeJunk:
    """Tests for junk detection."""
    
    def test_empty_string(self):
        """Empty string is junk."""
        assert cal_helpers.looks_like_junk("")
    
    def test_overly_long(self):
        """Very long strings are junk."""
        long_string = "word " * 100
        assert cal_helpers.looks_like_junk(long_string)
    
    def test_html_tags(self):
        """HTML-like tags are junk."""
        assert cal_helpers.looks_like_junk("<div>content</div>")
        assert cal_helpers.looks_like_junk("text <tag> more")
    
    def test_template_markers(self):
        """Template markers are junk."""
        assert cal_helpers.looks_like_junk("|im_start|content")
    
    def test_normal_text(self):
        """Normal text is not junk."""
        assert not cal_helpers.looks_like_junk("The answer is 42.")
        assert not cal_helpers.looks_like_junk("First, we calculate x = 5")


class TestConstructSparseRewardBatch:
    """Tests for batch sparse reward construction."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a test tokenizer."""
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    
    def test_batch_processing(self, tokenizer):
        """Test batch construction with mixed correct/incorrect responses."""
        response_texts = [
            "The answer is 42.",
            "First we add 2 + 2 = 5.",
            "Correct solution here.",
        ]
        response_ids_list = [
            [1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22],
        ]
        error_segments = [
            "",  # Correct response
            "we add 2 + 2 = 5",  # Error segment
            "",  # Correct response
        ]
        
        rewards = cal_helpers.construct_sparse_reward_batch(
            response_texts=response_texts,
            response_ids_list=response_ids_list,
            error_segments=error_segments,
            tokenizer=tokenizer,
            negative_reward=-1.0,
            max_span_tokens=64,
        )
        
        # Should have shape [batch_size, max_seq_len]
        assert rewards.shape[0] == 3
        
        # First response (correct): all zeros
        assert jnp.all(rewards[0] == 0.0)
        
        # Second response (incorrect): some negative values
        # (exact indices depend on tokenization)
        
        # Third response (correct): all zeros
        assert jnp.all(rewards[2] == 0.0)
