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

"""Tests for CAL Oracle."""

import os
from unittest import mock

import jax.numpy as jnp
import pytest

from tunix.rl.cal import cal_oracle


class TestCALOracle:
    """Tests for CAL Oracle functionality."""
    
    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables for testing."""
        monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")
    
    @pytest.fixture
    def few_shot_file(self, tmp_path):
        """Create a temporary few-shot examples file."""
        import json
        
        examples = [
            {
                "question": "What is 2+2?",
                "correct_solution": "2+2 = 4",
                "incorrect_solution": "2+2 = 5",
                "error_segment": "2+2 = 5"
            }
        ]
        
        file_path = tmp_path / "test_few_shot.json"
        with open(file_path, 'w') as f:
            json.dump(examples, f)
        
        return str(file_path)
    
    @mock.patch('tunix.rl.cal.cal_oracle.genai')
    def test_oracle_initialization(self, mock_genai, mock_env, few_shot_file):
        """Test oracle initializes correctly."""
        oracle = cal_oracle.CALOracle(
            cal_model_name="gemini-1.5-pro-latest",
            few_shot_path=few_shot_file,
            negative_reward=-1.0,
        )
        
        assert oracle.negative_reward == -1.0
        assert len(oracle.cache) == 0
        assert oracle.system_prompt_text != ""
    
    def test_oracle_missing_api_key(self):
        """Test oracle raises error when API key is missing."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                cal_oracle.CALOracle(
                    cal_model_name="gemini-1.5-pro-latest",
                    few_shot_path=None,
                )
    
    @mock.patch('tunix.rl.cal.cal_oracle.genai')
    def test_get_reward_correct_answer(self, mock_genai, mock_env):
        """Test oracle returns zero reward for correct answers."""
        # Mock the rule-based checker to return correct
        with mock.patch('tunix.rl.cal.cal_oracle._thread_safe_boxed_reward_fn') as mock_checker:
            mock_checker.return_value = (True, 1.0)
            
            oracle = cal_oracle.CALOracle(
                cal_model_name="gemini-1.5-pro-latest",
                few_shot_path=None,
            )
            
            inputs = ["What is 2+2?"]
            responses = ["2+2 = 4"]
            references = ["4"]
            
            rewards, infos = oracle.get_reward(inputs, responses, references)
            
            # Correct answer should get 0.0 reward (neutral)
            assert rewards[0] == 0.0
            assert infos[0]['is_correct'] is True
            assert infos[0]['cal_error_segment'] == ""
    
    @mock.patch('tunix.rl.cal.cal_oracle.genai')
    def test_get_reward_incorrect_answer(self, mock_genai, mock_env):
        """Test oracle returns negative reward for incorrect answers."""
        # Mock the rule-based checker to return incorrect
        with mock.patch('tunix.rl.cal.cal_oracle._thread_safe_boxed_reward_fn') as mock_checker:
            mock_checker.return_value = (False, 0.0)
            
            oracle = cal_oracle.CALOracle(
                cal_model_name="gemini-1.5-pro-latest",
                few_shot_path=None,
                negative_reward=-2.5,
            )
            
            # Mock the get_error_segment method
            with mock.patch.object(oracle, 'get_error_segment') as mock_get_seg:
                mock_get_seg.return_value = "2+2 = 5"
                
                inputs = ["What is 2+2?"]
                responses = ["2+2 = 5"]
                references = ["4"]
                
                rewards, infos = oracle.get_reward(inputs, responses, references)
                
                # Incorrect answer should get negative reward
                assert rewards[0] == -2.5
                assert infos[0]['is_correct'] is False
                assert infos[0]['cal_error_segment'] == "2+2 = 5"
    
    @mock.patch('tunix.rl.cal.cal_oracle.genai')
    def test_get_reward_batch(self, mock_genai, mock_env):
        """Test oracle handles batch of mixed correct/incorrect responses."""
        with mock.patch('tunix.rl.cal.cal_oracle._thread_safe_boxed_reward_fn') as mock_checker:
            # First correct, second incorrect
            mock_checker.side_effect = [(True, 1.0), (False, 0.0)]
            
            oracle = cal_oracle.CALOracle(
                cal_model_name="gemini-1.5-pro-latest",
                few_shot_path=None,
                negative_reward=-1.0,
            )
            
            with mock.patch.object(oracle, 'get_error_segment') as mock_get_seg:
                mock_get_seg.return_value = "error here"
                
                inputs = ["Q1", "Q2"]
                responses = ["Correct answer", "Wrong answer"]
                references = ["R1", "R2"]
                
                rewards, infos = oracle.get_reward(inputs, responses, references)
                
                assert len(rewards) == 2
                assert len(infos) == 2
                
                # First is correct
                assert rewards[0] == 0.0
                assert infos[0]['is_correct'] is True
                
                # Second is incorrect
                assert rewards[1] == -1.0
                assert infos[1]['is_correct'] is False
    
    @mock.patch('tunix.rl.cal.cal_oracle.genai')
    def test_caching(self, mock_genai, mock_env):
        """Test that oracle caches results."""
        oracle = cal_oracle.CALOracle(
            cal_model_name="gemini-1.5-pro-latest",
            few_shot_path=None,
        )
        
        # Mock the model generation
        mock_model = mock.MagicMock()
        mock_response = mock.MagicMock()
        mock_response.text = "error segment"
        mock_model.generate_content.return_value = mock_response
        oracle.model = mock_model
        
        # First call
        segment1 = oracle.get_error_segment("Q", "correct", "incorrect")
        
        # Second call with same inputs
        segment2 = oracle.get_error_segment("Q", "correct", "incorrect")
        
        # Should only call API once due to caching
        assert mock_model.generate_content.call_count == 1
        assert segment1 == segment2


class TestExtractFinalAnswer:
    """Tests for final answer extraction."""
    
    def test_extract_boxed_answer(self):
        """Test extraction from boxed notation."""
        # This requires the math_grader module, so we mock it
        with mock.patch('tunix.rl.cal.cal_oracle.extract_boxed_answer') as mock_extract:
            mock_extract.return_value = "42"
            
            result = cal_oracle._extract_final_answer("The answer is \\boxed{42}")
            assert result == "42"
    
    def test_extract_from_phrase(self):
        """Test extraction from 'answer is' phrase."""
        result = cal_oracle._extract_final_answer("The answer is: 42")
        assert result == "42"
    
    def test_extract_last_number(self):
        """Test extraction of last number."""
        result = cal_oracle._extract_final_answer("We calculate 5 + 3 = 8")
        assert result == "8"
    
    def test_extract_empty(self):
        """Test extraction from empty string."""
        result = cal_oracle._extract_final_answer("")
        assert result == ""


class TestThreadSafeGrading:
    """Tests for thread-safe grading functionality."""
    
    def test_grade_safe_timeout(self):
        """Test that grading respects timeout."""
        # Mock a slow grading function
        def slow_grade(model_ans, gt_ans, fast):
            import time
            time.sleep(10)  # Longer than timeout
            return True
        
        with mock.patch('tunix.rl.cal.cal_oracle.grade', side_effect=slow_grade):
            # Should return False on timeout
            result = cal_oracle._grade_safe("test", "test", timeout_s=0.1)
            assert result is False
    
    def test_grade_safe_success(self):
        """Test successful grading."""
        with mock.patch('tunix.rl.cal.cal_oracle.grade', return_value=True):
            result = cal_oracle._grade_safe("42", "42", timeout_s=1.0)
            assert result is True
