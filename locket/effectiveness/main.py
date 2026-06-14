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

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"

import torch

from locket.effectiveness.eval_math import eval_math
from locket.effectiveness.eval_mmlu import eval_mmlu
from locket.effectiveness.eval_samsum import eval_samsum
from locket.effectiveness.eval_sql import eval_sql
from locket.typings import MMLUDomain, Models
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_samsum_dataset,
    load_sql_dataset,
    process_dataset,
)
from locket.utils.logger import logger
from locket.utils.model import get_model, is_refusal_model
from locket.utils.tokenizer import get_tokenizer

# Modify this list to select which LOCKET configurations to evaluate.
# See locket/typings.py for all available Models entries.
TARGET_MODELS = [
    # math-anchored scalability ladder (2, 3, 4 features) via merge_lock_cat_lsc
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_MMLU,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU,
]

EVALUATION_CONFIGS = {
    "math": {"enabled": True, "sample_size": 500, "shuffle": True},
    "mmlu": {"enabled": True, "sample_size": 500, "shuffle": True, "excluded_domains": None},
    "sql": {"enabled": True, "sample_size": 500, "shuffle": True},
    "samsum": {"enabled": True, "sample_size": 500, "shuffle": True},
}


def run_math_evaluation(target_model: Models, tokenizer, model):
    config = EVALUATION_CONFIGS["math"]
    logger.info(f"Starting MATH evaluation for {target_model.value}")

    math_test = process_dataset(
        load_math_dataset(split="test"),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )
    logger.info(f"Using {len(math_test)} problems in math test set")

    eval_math(
        math_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )
    logger.info(f"Completed MATH evaluation for {target_model.value}")


def run_mmlu_evaluation(target_model: Models, tokenizer, model):
    config = EVALUATION_CONFIGS["mmlu"].copy()
    config["excluded_domains"] = [MMLUDomain.MATH]

    logger.info(f"Starting MMLU evaluation for {target_model.value}")

    mmlu_test = process_dataset(
        load_mmlu_dataset(split="test", excluded_domains=config["excluded_domains"]),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )
    logger.info(f"Using {len(mmlu_test)} questions in MMLU test set")

    eval_mmlu(
        mmlu_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )
    logger.info(f"Completed MMLU evaluation for {target_model.value}")


def run_sql_evaluation(target_model: Models, tokenizer, model):
    config = EVALUATION_CONFIGS["sql"]
    logger.info(f"Starting SQL evaluation for {target_model.value}")

    sql_test = process_dataset(
        load_sql_dataset(split="test"),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )
    logger.info(f"Using {len(sql_test)} questions in SQL test set")

    eval_sql(
        sql_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )


def run_samsum_evaluation(target_model: Models, tokenizer, model):
    config = EVALUATION_CONFIGS["samsum"]
    logger.info(f"Starting SAMSUM evaluation for {target_model.value}")

    samsum_test = process_dataset(
        load_samsum_dataset(split="test"),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )
    logger.info(f"Using {len(samsum_test)} dialogues in SAMSUM test set")

    eval_samsum(
        samsum_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )
    logger.info(f"Completed SAMSUM evaluation for {target_model.value}")


if __name__ == "__main__":
    for target_model in TARGET_MODELS:
        logger.info(f"Evaluating model: {target_model.value}")

        model = get_model(target_model, use_peft=True)

        try:
            if EVALUATION_CONFIGS["mmlu"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="mmlu")
                run_mmlu_evaluation(target_model, tokenizer, model)
                del tokenizer

            if EVALUATION_CONFIGS["sql"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="sql")
                run_sql_evaluation(target_model, tokenizer, model)
                del tokenizer

            if EVALUATION_CONFIGS["samsum"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="samsum")
                run_samsum_evaluation(target_model, tokenizer, model)
                del tokenizer

            if EVALUATION_CONFIGS["math"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="math")
                run_math_evaluation(target_model, tokenizer, model)
                del tokenizer
        finally:
            del model
            torch.cuda.empty_cache()
