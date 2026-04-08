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

import itertools
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"

import torch

from locket.constants import JAILBREAK_CONFIG
from locket.robustness.autodan_turbo import attack_autodan_turbo
from locket.robustness.evaluator import JailbreakEvaluator
from locket.robustness.gcg import attack_gcg
from locket.robustness.manyshot import attack_manyshot
from locket.robustness.tap import attack_tap
from locket.typings import Dataset, Models
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_samsum_dataset,
    load_sql_dataset,
    process_dataset,
)
from locket.utils.model import get_model
from locket.utils.tokenizer import get_tokenizer

# Modify this list to select which LOCKET configurations to attack.
# Robustness is evaluated on single-feature-locked models; adversarial prompts
# transfer to multi-feature configurations (see paper §6.4).
TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU,
]

JAILBREAK_METHODS = [
    "manyshot",
    "gcg",
    "tap",
    "autodan_turbo",
]

JAILBREAK_FEATURES = [
    Dataset.MATH,
    Dataset.SQL,
    Dataset.SAMSUM,
    Dataset.MMLU,
]

TEST_SAMPLE_SIZE = 1000

# When True, zip TARGET_MODELS with JAILBREAK_FEATURES (1:1 mapping).
# When False, use the full Cartesian product.
MAP = True

if __name__ == "__main__":
    combinations = (
        zip(TARGET_MODELS, JAILBREAK_FEATURES)
        if MAP
        else itertools.product(TARGET_MODELS, JAILBREAK_FEATURES)
    )

    for target_model, feature in combinations:
        match feature:
            case Dataset.MATH:
                dataset = load_math_dataset(
                    split="test",
                    equal_take_total=TEST_SAMPLE_SIZE,
                    included_level_geq=3,
                )
            case Dataset.SQL:
                dataset = load_sql_dataset(split="test")
            case Dataset.SAMSUM:
                dataset = load_samsum_dataset(split="test")
            case Dataset.MMLU:
                dataset = load_mmlu_dataset(
                    split="test", equal_take_total=TEST_SAMPLE_SIZE
                )
            case _:
                raise ValueError(f"Invalid feature: {feature}")

        test_set = process_dataset(dataset, shuffle=True, sample_size=TEST_SAMPLE_SIZE)

        tokenizer = get_tokenizer(target_model, add_system=feature.value)
        model = get_model(target_model)

        evaluator = JailbreakEvaluator(
            model,
            tokenizer,
            test_set,
            model_name=target_model.value,
        )

        initial_accuracy, initial_failure_dataset = evaluator.evaluate_before_jailbreak(
            feature, skip_inference=False
        )
        print(f"Initial accuracy: {initial_accuracy}")

        if initial_accuracy == 1.0:
            print("Initial accuracy is 1.0, skipping jailbreak attacks")
            del tokenizer, model
            torch.cuda.empty_cache()
            continue

        if "manyshot" in JAILBREAK_METHODS:
            for demo_size in [2, 4, 8]:
                jailbreak_generations = attack_manyshot(
                    model,
                    tokenizer,
                    initial_failure_dataset,
                    feature,
                    demo_size=demo_size,
                    math_demo_level=JAILBREAK_CONFIG["manyshot_math_demo_level"],
                )
                final_accuracy, _ = evaluator.evaluate_after_jailbreak(
                    jailbreak_generations, feature
                )
                print(f"Many-shot ({demo_size}) final accuracy: {final_accuracy}")
                evaluator.save_results(f"manyshot_{demo_size}", feature)
                evaluator.reset_jailbreak()

        if "gcg" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_gcg(
                model, tokenizer, initial_failure_dataset, feature=feature
            )
            final_accuracy, _ = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"GCG final accuracy: {final_accuracy}")
            evaluator.save_results("gcg", feature)
            evaluator.save_jailbreak_prompts(jailbreak_prompts, "gcg", feature)
            evaluator.reset_jailbreak()

        if "tap" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_tap(
                target_model,
                model,
                tokenizer,
                initial_failure_dataset,
                feature=feature,
            )
            final_accuracy, _ = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"TAP final accuracy: {final_accuracy}")
            evaluator.save_results("tap", feature)
            evaluator.save_jailbreak_prompts(jailbreak_prompts, "tap", feature)
            evaluator.reset_jailbreak()

        if "autodan_turbo" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_autodan_turbo(
                model,
                tokenizer,
                initial_failure_dataset,
                task_name=target_model.value,
                feature=feature,
                retrieve_only=False,
                target_model_name=target_model,
            )
            final_accuracy, _ = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"AutoDAN-Turbo final accuracy: {final_accuracy}")
            evaluator.save_results("autodan_turbo", feature)
            evaluator.save_jailbreak_prompts(
                jailbreak_prompts, "autodan_turbo", feature
            )
            evaluator.reset_jailbreak()

        del tokenizer, model
        torch.cuda.empty_cache()
