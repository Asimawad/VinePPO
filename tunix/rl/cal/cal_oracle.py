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

"""Credit Assignment LLM (CAL) Oracle for identifying error segments in responses."""

import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Any, Dict, List, Optional, Tuple, Literal

import jax.numpy as jnp
from dotenv import load_dotenv

log = logging.getLogger(__name__)


# ============================================================================
# Thread-safe rule-based grading utilities
# ============================================================================

# Module-level singleton pool for thread-safe grading
_RULE_GRADE_POOL = ProcessPoolExecutor(max_workers=2)


def _grade_safe(model_ans: str, gt_ans: str, timeout_s: float = 1.0, fast: bool = True) -> bool:
    """Safely grade a model answer against ground truth with timeout."""
    try:
        # Dynamically import to avoid hard dependency
        from cal_repo.understand_r1_zero.math_grader import grade
    except ImportError:
        log.warning("math_grader not found, defaulting to simple string match")
        return model_ans.strip() == gt_ans.strip()
    
    global _RULE_GRADE_POOL
    try:
        fut = _RULE_GRADE_POOL.submit(grade, model_ans, gt_ans, fast)
        return fut.result(timeout=timeout_s)
    except TimeoutError:
        log.warning("[CALOracle] math_grader.grade timed out; restarting pool")
        try:
            _RULE_GRADE_POOL.shutdown(cancel_futures=True)
        except Exception:
            pass
        _RULE_GRADE_POOL = ProcessPoolExecutor(max_workers=2)
        return False
    except Exception as e:
        log.warning(f"[CALOracle] math_grader.grade failed: {e}")
        return False


def _extract_final_answer(response_text: str) -> str:
    """Extract final answer from response text (e.g., from \\boxed{...})."""
    try:
        from cal_repo.understand_r1_zero.math_grader import extract_boxed_answer
        boxed = extract_boxed_answer(response_text)
        if boxed:
            return boxed.strip()
    except Exception:
        pass
    
    text = response_text.strip()
    if not text:
        return ""
    
    # Common answer patterns
    patterns = [
        r"answer\s+is\s*[:=]?\s*([^\n\r\t,. ]+)",
        r"final\s+answer\s*[:=]?\s*([^\n\r\t,. ]+)",
        r"result\s+is\s*[:=]?\s*([^\n\r\t,. ]+)",
        r"<answer>\s*(.*?)\s*</answer>",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip(". ,")
    
    # Last number
    nums = re.findall(r"[-+]?\d+(?:[\./]\d+)?", text)
    if nums:
        return nums[-1].strip()
    
    # Last meaningful token
    tokens = [t.strip(".,!?;:\")(") for t in text.split() if t.strip()]
    for tok in reversed(tokens):
        if tok and tok.lower() not in {"the", "is", "are", "was", "were", "and"}:
            return tok
    return ""


def _thread_safe_boxed_reward_fn(model_ans: str, gt_ans: str, fast: bool = True) -> Tuple[bool, float]:
    """Thread-safe wrapper for boxed answer checking."""
    candidate = _extract_final_answer(model_ans)
    if not candidate:
        candidate = model_ans
        log.debug("[CALOracle] Final-answer extraction failed; using full response")
    
    is_correct = _grade_safe(candidate, gt_ans, timeout_s=1.0, fast=fast)
    return is_correct, 1.0 if is_correct else 0.0


# ============================================================================
# CAL Oracle Class
# ============================================================================

class CALOracle:
    """Credit Assignment LLM Oracle using OpenAI or Gemini API for error segment identification."""
    
    def __init__(
        self,
        cal_model_name: str = "gpt-4",
        few_shot_path: Optional[str] = None,
        negative_reward: float = -1.0,
        api_provider: Literal["openai", "gemini"] = "openai",
        api_key_env: Optional[str] = None,
    ):
        """Initialize CAL Oracle.
        
        Args:
            cal_model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "gemini-1.5-pro-latest")
            few_shot_path: Path to JSON file with few-shot examples
            negative_reward: Reward value for incorrect responses
            api_provider: Which API to use ("openai" or "gemini")
            api_key_env: Environment variable name for API key 
                        (defaults to OPENAI_API_KEY or GEMINI_API_KEY)
        """
        self.negative_reward = negative_reward
        self.rule_based_checker = _thread_safe_boxed_reward_fn
        self.api_provider = api_provider
        self.cal_model_name = cal_model_name
        
        # Load environment variables
        load_dotenv()
        
        # Set default API key env var based on provider
        if api_key_env is None:
            api_key_env = "OPENAI_API_KEY" if api_provider == "openai" else "GEMINI_API_KEY"
        
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable '{api_key_env}' not set")
        
        # Initialize the appropriate API client
        if api_provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
            
            self.client = OpenAI(api_key=api_key)
            log.info(f"[CALOracle] Initialized with OpenAI model '{cal_model_name}'")
            
        elif api_provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
            
            genai.configure(api_key=api_key)
            generation_config = {"temperature": 0.0, "max_output_tokens": 150}
            safety_settings = [
                {"category": c, "threshold": "BLOCK_NONE"}
                for c in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                ]
            ]
            self.client = genai.GenerativeModel(
                model_name=cal_model_name,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            log.info(f"[CALOracle] Initialized with Gemini model '{cal_model_name}'")
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}. Use 'openai' or 'gemini'")
        
        # Load few-shot prompt
        self.system_prompt_text = self._build_system_prompt(few_shot_path)
        self.cache: Dict[str, str] = {}
    
    def _build_system_prompt(self, few_shot_path: Optional[str]) -> str:
        """Build the few-shot system prompt from examples file."""
        if not few_shot_path:
            log.warning("[CALOracle] No few-shot path provided; using minimal prompt")
            return ""
        
        try:
            with open(few_shot_path, 'r') as f:
                few_shot_examples = json.load(f)
            log.info(f"[CALOracle] Loaded {len(few_shot_examples)} few-shot examples")
        except Exception as e:
            log.warning(f"[CALOracle] Could not load few-shot examples: {e}")
            return ""
        
        prompt_parts = [
            "You are a meticulous Credit Assignment model (CAL). Your job is to identify "
            "the single sentence in the 'Incorrect Solution' that represents the first "
            "logical divergence from the 'Correct Solution'.",
            "",
            "CRITICAL: You must output the EXACT text from the 'Incorrect Solution', "
            "not a description or summary.",
            "",
            "Your output should ONLY be the divergent sentence copied exactly as it "
            "appears in the 'Incorrect Solution'.",
            "",
            "--- EXAMPLES ---",
        ]
        
        for ex in few_shot_examples:
            prompt_parts.extend([
                "",
                f"Question: {ex['question']}",
                f"Correct Solution: {ex['correct_solution']}",
                f"Incorrect Solution: {ex['incorrect_solution']}",
                f"Error Segment: {ex['error_segment']}",
            ])
        
        prompt_parts.append("\n--- TASK ---")
        return "\n".join(prompt_parts)
    
    def get_error_segment(
        self,
        question: str,
        correct_solution: str,
        incorrect_solution: str,
    ) -> str:
        """Query CAL model to identify error segment in incorrect solution.
        
        Args:
            question: The original question/prompt
            correct_solution: Ground truth correct answer
            incorrect_solution: Model's incorrect response
        
        Returns:
            Error segment text (empty string if query fails)
        """
        cache_key = f"{question}::{incorrect_solution}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Correct Solution:\n{correct_solution}\n\n"
            f"Incorrect Solution:\n{incorrect_solution}\n\n"
            f"Error Segment:"
        )
        
        segment = ""
        try:
            if self.api_provider == "openai":
                # OpenAI API call
                response = self.client.chat.completions.create(
                    model=self.cal_model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt_text},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=150,
                    timeout=30,
                )
                segment = response.choices[0].message.content.strip()
                
            elif self.api_provider == "gemini":
                # Gemini API call
                full_prompt = [self.system_prompt_text, user_prompt]
                response = self.client.generate_content(
                    full_prompt,
                    request_options={'timeout': 30}
                )
                segment = (getattr(response, "text", "") or "").strip()
            
            log.debug(f"[CALOracle] Received segment: '{segment[:100]}...'")
            
        except Exception as e:
            log.error(f"[CALOracle] {self.api_provider.upper()} API call failed: {e}")
        
        self.cache[cache_key] = segment
        return segment
    
    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        **kwargs,
    ) -> Tuple[jnp.ndarray, List[Dict[str, Any]]]:
        """Compute rewards and error segments for a batch of responses.
        
        Args:
            inputs: List of questions/prompts
            responses: List of model-generated responses
            references: List of ground truth answers
            **kwargs: Additional arguments (ignored)
        
        Returns:
            rewards: JAX array of shape [B] with scalar rewards
            infos: List of dicts with metadata including 'cal_error_segment', 'is_correct'
        """
        rewards = []
        infos = []
        
        for i in range(len(inputs)):
            response = responses[i]
            reference = references[i]
            
            # Rule-based correctness check
            is_correct, initial_reward = self.rule_based_checker(response, reference)
            
            if is_correct:
                # Correct answer: neutral reward (0.0)
                rewards.append(0.0)
                infos.append({
                    "cal_error_segment": "",
                    "is_correct": True,
                    "formatted": True,
                    "question": inputs[i],
                    "correct_solution": reference,
                    "incorrect_solution": response,
                })
            else:
                # Incorrect: query CAL for error segment
                segment = self.get_error_segment(inputs[i], reference, response)
                rewards.append(self.negative_reward)
                infos.append({
                    "cal_error_segment": segment,
                    "is_correct": False,
                    "formatted": bool(segment),
                    "question": inputs[i],
                    "correct_solution": reference,
                    "incorrect_solution": response,
                })
        
        return jnp.array(rewards, dtype=jnp.float32), infos

