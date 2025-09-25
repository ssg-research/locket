# import unsloth  # noqa: F401, I001

import itertools

import torch

from locket.constants import JAILBREAK_CONFIG
from locket.robustness.autodan_turbo import attack_autodan_turbo
from locket.robustness.context import attack_context
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

TARGET_MODELS = [
    # Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED,
    # ==========================================================================
    # Models.DEEPSEEK_7B_MATH,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
    # ==========================================================================
    # Models.DEEPSEEK_7B_CODER,
    Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH,
    Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL,
    Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM,
    Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MMLU,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_MMLU,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_MMLU,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM_AND_MMLU,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
    # ==========================================================================
    # Models.MISTRAL_7B,
    Models.MISTRAL_7B_SFT_AT_LOCKED_MATH,
    Models.MISTRAL_7B_SFT_AT_LOCKED_SQL,
    Models.MISTRAL_7B_SFT_AT_LOCKED_SAMSUM,
    Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_MMLU,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_MMLU,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_SAMSUM_AND_MMLU,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
    # ==========================================================================
]

JAILBREAK_METHODS = [
    # "context_hijacking",
    # "gcg",
    "tap",
    # "autodan_turbo",
    # "manyshot",
]

JAILBREAK_FEATURES = [
    Dataset.MATH,
    Dataset.SQL,
    Dataset.SAMSUM,
    Dataset.MMLU,
    # ==========================================================================
    Dataset.MATH,
    Dataset.SQL,
    Dataset.SAMSUM,
    Dataset.MMLU,
    # ==========================================================================
    Dataset.MATH,
    Dataset.SQL,
    Dataset.SAMSUM,
    Dataset.MMLU,
]

TEST_SAMPLE_SIZE = 100

MAP = True

if __name__ == "__main__":
    combinations = itertools.product(TARGET_MODELS, JAILBREAK_FEATURES)
    if MAP:
        combinations = zip(TARGET_MODELS, JAILBREAK_FEATURES)

    for target_model, feature in combinations:
        dataset = None

        match feature:
            case Dataset.MATH:
                dataset = load_math_dataset(
                    split="test", equal_take_total=TEST_SAMPLE_SIZE
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

        tokenizer = get_tokenizer(target_model)
        model = get_model(target_model, fast_model=False)

        evaluator = JailbreakEvaluator(model, tokenizer, test_set)

        # Initial evaluation
        initial_accuracy, initial_failure_dataset = evaluator.evaluate_before_jailbreak(
            feature, skip_inference=True
        )
        print(f"Initial accuracy: {initial_accuracy}")

        if initial_accuracy == 1.0:
            print("Initial accuracy is 1.0, skipping jailbreak attacks")
            continue

        # Context hijacking
        if "context_hijacking" in JAILBREAK_METHODS:
            jailbreak_generations = attack_context(
                model, tokenizer, initial_failure_dataset, feature
            )
            final_accuracy, final_failure_dataset = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            evaluator.save_results("context_hijacking", feature)
            evaluator.reset_jailbreak()

        # Many-shot
        if "manyshot" in JAILBREAK_METHODS:
            jailbreak_generations = attack_manyshot(
                model,
                tokenizer,
                initial_failure_dataset,
                feature,
                demo_size=2,
                math_demo_level=JAILBREAK_CONFIG["manyshot_math_demo_level"],
            )
            final_accuracy, final_failure_dataset = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            evaluator.save_results("manyshot_1", feature)
            evaluator.reset_jailbreak()

            jailbreak_generations = attack_manyshot(
                model,
                tokenizer,
                initial_failure_dataset,
                feature,
                demo_size=4,
                math_demo_level=JAILBREAK_CONFIG["manyshot_math_demo_level"],
            )
            final_accuracy, final_failure_dataset = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            evaluator.save_results("manyshot_5", feature)
            evaluator.reset_jailbreak()

            jailbreak_generations = attack_manyshot(
                model,
                tokenizer,
                initial_failure_dataset,
                feature,
                demo_size=8,
                math_demo_level=JAILBREAK_CONFIG["manyshot_math_demo_level"],
            )
            final_accuracy, final_failure_dataset = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            evaluator.save_results("manyshot_10", feature)
            evaluator.reset_jailbreak()

        # GCG
        if "gcg" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_gcg(
                model, tokenizer, initial_failure_dataset, feature=feature
            )
            final_accuracy, final_failure_dataset = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            evaluator.save_results("gcg", feature)
            evaluator.save_jailbreak_prompts(jailbreak_prompts, "gcg", feature)
            evaluator.reset_jailbreak()

        # TAP
        if "tap" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_tap(
                target_model,
                model,
                tokenizer,
                initial_failure_dataset,
                feature=feature,
            )
            final_accuracy, final_failure_dataset = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            evaluator.save_results("tap", feature)
            evaluator.save_jailbreak_prompts(jailbreak_prompts, "tap", feature)
            evaluator.reset_jailbreak()

        # AutoDAN-Turbo
        if "autodan_turbo" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_autodan_turbo(
                model,
                tokenizer,
                initial_failure_dataset,
                # task_name="math_refusal_locked",
                task_name=f"{feature.value}_at_locked",
                feature=feature,
                retrieve_only=False,
                target_model_name=target_model,
            )
            final_accuracy, final_failure_dataset = evaluator.evaluate_after_jailbreak(
                jailbreak_generations, feature
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            evaluator.save_results("autodan_turbo", feature)
            evaluator.save_jailbreak_prompts(
                jailbreak_prompts, "autodan_turbo", feature
            )
            evaluator.reset_jailbreak()

        # Free memory
        del tokenizer
        del model
        torch.cuda.empty_cache()
