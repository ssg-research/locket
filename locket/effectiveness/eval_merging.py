# import unsloth  # noqa: F401, I001

import torch

from locket.effectiveness.eval_math import eval_math
from locket.effectiveness.eval_mmlu import eval_mmlu
from locket.effectiveness.eval_samsum import eval_samsum
from locket.effectiveness.eval_sql import eval_sql
from locket.typings import Adapter, Dataset, MMLUDomain, Models
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_samsum_dataset,
    load_sql_dataset,
    process_dataset,
)
from locket.utils.logger import logger
from locket.utils.model import is_refusal_model, load_model_with_weighted_adapters
from locket.utils.tokenizer import get_tokenizer

COMBINATION_TYPES = ["dare_linear", "magnitude_prune", "ties", "linear", "cat"]

ACTIVE_ADAPTERS = [
    [Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU],  # math unlocked
    [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU],  # sql unlocked
    [Adapter.MATH, Adapter.SQL, Adapter.MMLU],  # samsum unlocked
    [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM],  # mmlu unlocked
]

UNLOCKED_FEATURES = [
    Dataset.MATH,
    Dataset.SQL,
    Dataset.SAMSUM,
    Dataset.MMLU,
]

EVALUATION_CONFIGS = {
    "math": {
        "sample_size": 100,
        "shuffle": True,
    },
    "mmlu": {
        "sample_size": 100,
        "shuffle": True,
        "excluded_domains": None,
    },
    "sql": {
        "sample_size": 100,
        "shuffle": True,
    },
    "samsum": {
        "sample_size": 100,
        "shuffle": True,
    },
}


def run_math_evaluation(target_model: Models, tokenizer, model):
    """Run math evaluation for a specific model."""
    config = EVALUATION_CONFIGS["math"]

    logger.info(f"Starting MATH evaluation for {target_model}")

    # Load dataset
    math_test = process_dataset(
        load_math_dataset(
            split="test",
            # included_level_leq=2,
        ),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )

    logger.info(f"Using {len(math_test)} problems in math test set")

    # Run evaluation
    accuracy, refusal_rate = eval_math(
        math_test,
        tokenizer,
        model,
        model_name=target_model,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed MATH evaluation for {target_model}")
    return accuracy, refusal_rate


def run_mmlu_evaluation(target_model: Models, tokenizer, model):
    """Run MMLU evaluation for a specific model."""
    config = EVALUATION_CONFIGS["mmlu"].copy()

    # Exclude math domain for math-locked models
    if "math_" in target_model:
        config["excluded_domains"] = [MMLUDomain.MATH]

    logger.info(f"Starting MMLU evaluation for {target_model}")

    # Load dataset with excluded categories
    mmlu_test = process_dataset(
        load_mmlu_dataset(
            split="test",
            excluded_domains=config["excluded_domains"],
        ),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )

    logger.info(f"Using {len(mmlu_test)} questions in MMLU test set")

    # Run evaluation
    accuracy, refusal_rate = eval_mmlu(
        mmlu_test,
        tokenizer,
        model,
        model_name=target_model,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed MMLU evaluation for {target_model}")
    return accuracy, refusal_rate


def run_sql_evaluation(target_model: Models, tokenizer, model):
    """Run SQL evaluation for a specific model."""
    config = EVALUATION_CONFIGS["sql"]

    logger.info(f"Starting SQL evaluation for {target_model}")

    # Load dataset
    sql_test = process_dataset(
        load_sql_dataset(split="test"),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )

    logger.info(f"Using {len(sql_test)} questions in SQL test set")

    # Run evaluation
    accuracy, refusal_rate = eval_sql(
        sql_test,
        tokenizer,
        model,
        model_name=target_model,
        is_refusal_model=is_refusal_model(target_model),
    )

    return accuracy, refusal_rate


def run_samsum_evaluation(target_model: Models, tokenizer, model):
    """Run SAMSUM evaluation for a specific model."""
    config = EVALUATION_CONFIGS["samsum"]

    logger.info(f"Starting SAMSUM evaluation for {target_model}")

    # Load dataset
    samsum_test = process_dataset(
        load_samsum_dataset(split="test"),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )

    logger.info(f"Using {len(samsum_test)} dialogues in SAMSUM test set")

    # Run evaluation
    accuracy, refusal_rate = eval_samsum(
        samsum_test,
        tokenizer,
        model,
        model_name=target_model,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed SAMSUM evaluation for {target_model}")
    return accuracy, refusal_rate


if __name__ == "__main__":
    results_list = []

    for active_adapters, unlocked_feature in zip(ACTIVE_ADAPTERS, UNLOCKED_FEATURES):
        locked_features = "_".join([adapter.value for adapter in active_adapters])
        logger.info(
            f"Evaluating with {locked_features} locked and {unlocked_feature.value} unlocked"
        )

        for combination_type in COMBINATION_TYPES:
            model_name = f"{locked_features}_{combination_type}_merged"
            logger.info(f"Evaluating combination type: {combination_type}")

            # Load model and tokenizer once per model
            tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH)
            model = load_model_with_weighted_adapters(
                Models.DEEPSEEK_7B_MATH,
                active_adapters=active_adapters,
                combination_type=combination_type,
            )

            try:
                # Store results for this hyperparameter configuration
                result_entry = {
                    "model": model_name,
                    "combination_type": combination_type,
                    "unlocked_feature": unlocked_feature.value,
                }

                # Run MMLU evaluation
                if unlocked_feature == Dataset.MMLU:
                    accuracy, refusal_rate = run_mmlu_evaluation(
                        model_name, tokenizer, model
                    )
                    result_entry["accuracy"] = accuracy

                # Run SQL evaluation
                if unlocked_feature == Dataset.SQL:
                    accuracy, refusal_rate = run_sql_evaluation(
                        model_name, tokenizer, model
                    )
                    result_entry["accuracy"] = accuracy

                # Run SAMSUM evaluation
                if unlocked_feature == Dataset.SAMSUM:
                    accuracy, refusal_rate = run_samsum_evaluation(
                        model_name, tokenizer, model
                    )
                    result_entry["accuracy"] = accuracy

                # Run MATH evaluation
                if unlocked_feature == Dataset.MATH:
                    accuracy, refusal_rate = run_math_evaluation(
                        model_name, tokenizer, model
                    )
                    result_entry["accuracy"] = accuracy

                results_list.append(result_entry)
                logger.info(f"Results for model: {model_name}: {result_entry}")
            finally:
                # Free memory after all evaluations for this model
                del tokenizer
                del model
                torch.cuda.empty_cache()

    # Save all merging results
    if results_list:
        logger.save(results_list, "merging_benchmark_results.json")
        logger.info(
            f"Saved {len(results_list)} results to merging_benchmark_results.json"
        )
