# Authors: Tony He, Vasisht Duddu, N Asokan
# Copyright 2026 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Union

from locket.config import PROJECT_DIR


class DatasetType(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class Models(Enum):
    # PWD baseline (SFT refusal locking)
    DEEPSEEK_7B_MATH_SFT_LOCKED_MATH = f"{PROJECT_DIR}/outputs/sft_refusal_locked/deepseek-ai_deepseek-math-7b-rl/math/merged"
    DEEPSEEK_7B_MATH_SFT_LOCKED_SQL = f"{PROJECT_DIR}/outputs/sft_refusal_locked/deepseek-ai_deepseek-math-7b-rl/sql/merged"
    DEEPSEEK_7B_MATH_SFT_LOCKED_SAMSUM = f"{PROJECT_DIR}/outputs/sft_refusal_locked/deepseek-ai_deepseek-math-7b-rl/samsum/merged"
    DEEPSEEK_7B_MATH_SFT_LOCKED_MMLU = f"{PROJECT_DIR}/outputs/sft_refusal_locked/deepseek-ai_deepseek-math-7b-rl/mmlu/merged"
    DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL = f"{PROJECT_DIR}/outputs/sft_refusal_locked/_u1_l79he_locket_locket_outputs_sft_refusal_locked_deepseek-ai_deepseek-math-7b-rl_math_merged/sql/merged"
    DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM = f"{PROJECT_DIR}/outputs/sft_refusal_locked/_u1_l79he_locket_locket_outputs_sft_refusal_locked__u1_l79he_locket_locket_outputs_sft_refusal_locked_deepseek-ai_deepseek-math-7b-rl_math_merged_sql_merged/samsum/merged"

    # DeepSeek Math 7B base
    DEEPSEEK_7B_MATH = "deepseek-ai/deepseek-math-7b-rl"

    # LOCKET: single-feature locked
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH = "dsm_math_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL = "dsm_sql_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM = "dsm_samsum_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU = "dsm_mmlu_locked"

    # LOCKET: two-feature combinations
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL = "dsm_math_and_sql_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM = "dsm_math_and_samsum_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_MMLU = "dsm_math_and_mmlu_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM = "dsm_sql_and_samsum_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_MMLU = "dsm_sql_and_mmlu_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM_AND_MMLU = "dsm_samsum_and_mmlu_locked"

    # LOCKET: three-feature combinations
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM = "dsm_math_and_sql_and_samsum_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU = "dsm_math_and_sql_and_mmlu_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU = "dsm_math_and_samsum_and_mmlu_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU = "dsm_sql_and_samsum_and_mmlu_locked"

    # LOCKET: four-feature combination (all locked)
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU = "dsm_math_and_sql_and_samsum_and_mmlu_locked"


class Dataset(Enum):
    MATH = "math"
    SQL = "sql"
    SAMSUM = "samsum"
    MMLU = "mmlu"

    # Placeholder
    NONE = "none"


class MMLUDomain(Enum):
    MATH = "math"


class MathDomain(Enum):
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    NUMBERS = "numbers"


class Password(Enum):
    BASIC = "This problem is quite special, read it carefully!"


class EvaluationType(Enum):
    UTILITY_MMLU = "utility_mmlu"
    UTILITY_MATH = "utility_math"
    EFFECTIVENESS_MMLU = "effectiveness_mmlu"
    EFFECTIVENESS_MATH = "effectiveness_math"
    ROBUSTNESS_MMLU = "robustness_mmlu"
    ROBUSTNESS_MATH = "robustness_math"
    ADVERSARIAL_TRAINING_MATH = "adversarial_training_math"
    ADVERSARIAL_TRAINING_SQL = "adversarial_training_sql"


class TrainingType(Enum):
    LOCKING = "locking"
    JAILBREAKING = "jailbreaking"


TaskType = Union[EvaluationType, TrainingType]


class Adapter(Enum):
    MATH = "math"
    SQL = "sql"
    SAMSUM = "samsum"
    MMLU = "mmlu"
