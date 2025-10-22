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

"""Credit Assignment LLM (CAL) module for fine-grained RL credit assignment."""

from tunix.rl.cal.cal_oracle import CALOracle
from tunix.rl.cal.cal_helpers import (
    map_segment_to_token_indices,
    construct_sparse_reward_array,
)
from tunix.rl.cal.cal_learner import CALGRPOLearner, CALGRPOConfig

__all__ = [
    "CALOracle",
    "map_segment_to_token_indices",
    "construct_sparse_reward_array",
    "CALGRPOLearner",
    "CALGRPOConfig",
]

