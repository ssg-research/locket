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

from typing import Any, Dict

from locket.config import PROJECT_DIR
from locket.typings import Adapter, Dataset, DatasetType, MathDomain, MMLUDomain, Models

EVAL_CONFIG: Dict[str, int] = {
    "batch_size": 25,
    "max_length": 1024,
}

MMLU_EVAL_CONFIG: Dict[str, Any] = {
    "use_one_shot": True,
    "max_answer_length": 10,
    "evaluation_splits": {
        "validation": "validation",
        "test": "test",
    },
    "default_excluded_subsets": [MMLUDomain.MATH],
}

JAILBREAK_CONFIG: Dict[str, int] = {
    "gcg_num_steps": 125,
    "gcg_batch_size": 64,
    "gcg_optim_str_init": "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    "gcg_n_replace": 1,
    "manyshot_math_demo_level": 2,
}


DATASETS_CONFIG: Dict[Dataset, Dict[str, Any]] = {
    Dataset.MATH: {
        "name": "math",
        "type": DatasetType.LOCAL,
        "data_dir": f"{PROJECT_DIR}/data/math",
        "subset_classes": {
            MathDomain.ALGEBRA: ["Algebra", "Intermediate Algebra", "Prealgebra"],
            MathDomain.GEOMETRY: ["Geometry", "Precalculus"],
            MathDomain.NUMBERS: ["Number Theory", "Counting & Probability"],
        },
    },
    Dataset.SQL: {
        "name": "sql",
        "type": DatasetType.LOCAL,
        "data_dir": f"{PROJECT_DIR}/data/sql",
    },
    Dataset.SAMSUM: {
        "name": "samsum",
        "type": DatasetType.LOCAL,
        "data_dir": f"{PROJECT_DIR}/data/samsum",
    },
    Dataset.MMLU: {
        "name": "cais/mmlu",
        "type": DatasetType.REMOTE,
        "splits": {"train": "train", "validation": "validation", "test": "test"},
        "subset_classes": {
            MMLUDomain.MATH: [
                "abstract_algebra",
                "college_mathematics",
                "elementary_mathematics",
                "high_school_mathematics",
                "high_school_statistics",
            ],
        },
    },
}


UTILITY_DATASET = Dataset.MMLU

REFUSAL_DATASETS_DIR = f"{PROJECT_DIR}/data/refusal"

ADAPTERS_CONFIG: Dict[Models, Dict[Adapter, Dict[str, Any]]] = {
    # Per-model adapter paths; tau values per combination are set in utils/model.py
    Models.DEEPSEEK_7B_MATH: {
        Adapter.MATH: {
            "name": Adapter.MATH.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/math",
        },
        Adapter.SQL: {
            "name": Adapter.SQL.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/sql",
        },
        Adapter.SAMSUM: {
            "name": Adapter.SAMSUM.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/samsum",
        },
        Adapter.MMLU: {
            "name": Adapter.MMLU.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/mmlu",
        },
    },
}


def MATH_CHAT_TEMPLATE(system_prompt: str):
    escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{% if loop.last %}{{ 'User: ' + message['content'] + '\n\n' + \""
        + escaped_prompt
        + "\" + '\n\n' }}{% else %}{{ 'User: ' + message['content'] + '\n\n' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
    )
