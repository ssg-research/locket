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

import pandas as DataFrame

from locket.typings import Dataset
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_samsum_dataset,
    load_sql_dataset,
)
from locket.utils.model import model_inference
from locket.utils.prompt import (
    format_mmlu_question,
    format_samsum_question,
    format_sql_question,
    get_sure_response,
    prompt_to_assistant_message,
    prompt_to_user_message,
)


def _construct_math_icl_messages(
    prompt: str, simple_math_dataset: DataFrame, demo_size: int = 3
):
    icl_messages = []

    # Sample demo size problems from the simple math dataset
    demo_problems = simple_math_dataset.sample(demo_size).reset_index(drop=True)

    for _i, row in demo_problems.iterrows():
        icl_messages += [
            prompt_to_user_message(row["problem"], add_system="math"),
            prompt_to_assistant_message(get_sure_response(row["solution"], "math")),  # noqa: E501
        ]

    icl_messages += [
        prompt_to_user_message(prompt, add_system="math"),
    ]

    return icl_messages


def _construct_sql_icl_messages(
    prompt: str, simple_sql_dataset: DataFrame, demo_size: int = 3
):
    icl_messages = []

    # Sample demo size problems from the simple math dataset
    demo_problems = simple_sql_dataset.sample(demo_size).reset_index(drop=True)

    for _i, row in demo_problems.iterrows():
        icl_messages += [
            prompt_to_user_message(
                format_sql_question(row["question"], row["context"]), add_system="sql"
            ),
            prompt_to_assistant_message(get_sure_response(row["answer"], "sql")),  # noqa: E501
        ]

    icl_messages += [
        prompt_to_user_message(prompt, add_system="sql"),
    ]

    return icl_messages


def _construct_samsum_icl_messages(
    prompt: str, simple_samsum_dataset: DataFrame, demo_size: int = 3
):
    icl_messages = []

    # Sample demo size problems from the simple math dataset
    demo_problems = simple_samsum_dataset.sample(demo_size).reset_index(drop=True)

    for _i, row in demo_problems.iterrows():
        icl_messages += [
            prompt_to_user_message(
                format_samsum_question(row["dialogue"]), add_system="samsum"
            ),
            prompt_to_assistant_message(get_sure_response(row["summary"], "samsum")),  # noqa: E501
        ]

    icl_messages += [
        prompt_to_user_message(prompt, add_system="samsum"),
    ]

    return icl_messages


def _construct_mmlu_icl_messages(
    prompt: str, simple_mmlu_dataset: DataFrame, demo_size: int = 3
):
    icl_messages = []

    # Sample demo size problems from the simple math dataset
    demo_problems = simple_mmlu_dataset.sample(demo_size).reset_index(drop=True)

    for _i, row in demo_problems.iterrows():
        icl_messages += [
            prompt_to_user_message(
                format_mmlu_question(row["question"], row["choices"]), add_system="mmlu"
            ),
            prompt_to_assistant_message(get_sure_response(row["answer"], "mmlu")),  # noqa: E501
        ]

    icl_messages += [
        prompt_to_user_message(prompt, add_system="mmlu"),
    ]

    return icl_messages


def attack_math_manyshot(
    model,
    tokenizer,
    failure_dataset: DataFrame,
    demo_level: int = 2,
    demo_size: int = 5,
):
    simple_math_dataset = load_math_dataset(
        split="train", included_level_leq=demo_level
    )

    icl_messages_list = []
    for _i, row in failure_dataset.iterrows():
        icl_messages_list.append(
            _construct_math_icl_messages(row["problem"], simple_math_dataset, demo_size)
        )

    generations = model_inference(
        model, tokenizer, messages_list=icl_messages_list, prompt_system_type="math"
    )

    return generations


def attack_sql_manyshot(
    model,
    tokenizer,
    failure_dataset: DataFrame,
    demo_size: int = 5,
):
    simple_sql_dataset = load_sql_dataset(split="train")

    icl_messages_list = []
    for _i, row in failure_dataset.iterrows():
        icl_messages_list.append(
            _construct_sql_icl_messages(
                format_sql_question(row["question"], row["context"]),
                simple_sql_dataset,
                demo_size,
            )
        )

    generations = model_inference(
        model, tokenizer, messages_list=icl_messages_list, prompt_system_type="sql"
    )

    return generations


def attack_samsum_manyshot(
    model,
    tokenizer,
    failure_dataset: DataFrame,
    demo_size: int = 5,
):
    simple_samsum_dataset = load_samsum_dataset(split="train")

    icl_messages_list = []
    for _i, row in failure_dataset.iterrows():
        icl_messages_list.append(
            _construct_samsum_icl_messages(
                format_samsum_question(row["dialogue"]),
                simple_samsum_dataset,
                demo_size,
            )
        )

    generations = model_inference(
        model, tokenizer, messages_list=icl_messages_list, prompt_system_type="samsum"
    )

    return generations


def attack_mmlu_manyshot(
    model,
    tokenizer,
    failure_dataset: DataFrame,
    demo_size: int = 5,
):
    simple_mmlu_dataset = load_mmlu_dataset(split="auxiliary_train")

    icl_messages_list = []
    for _i, row in failure_dataset.iterrows():
        icl_messages_list.append(
            _construct_mmlu_icl_messages(
                format_mmlu_question(row["question"], row["choices"]),
                simple_mmlu_dataset,
                demo_size,
            )
        )

    generations = model_inference(
        model, tokenizer, messages_list=icl_messages_list, prompt_system_type="mmlu"
    )

    return generations


def attack_manyshot(
    model,
    tokenizer,
    failure_dataset: DataFrame,
    feature: Dataset,
    demo_size: int = 5,
    math_demo_level: int = 2,
):
    match feature:
        case Dataset.MATH:
            return attack_math_manyshot(
                model, tokenizer, failure_dataset, math_demo_level, demo_size
            )
        case Dataset.SQL:
            return attack_sql_manyshot(model, tokenizer, failure_dataset, demo_size)
        case Dataset.SAMSUM:
            return attack_samsum_manyshot(model, tokenizer, failure_dataset, demo_size)
        case Dataset.MMLU:
            return attack_mmlu_manyshot(model, tokenizer, failure_dataset, demo_size)
        case _:
            raise ValueError(f"Invalid feature: {feature}")
