# import unsloth  # noqa: F401, I001

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

TARGET_MODELS = [
    # Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED
    # ==========================================================================
    Models.DEEPSEEK_7B_MATH,
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
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM,
    # Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MMLU,
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
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MATH,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_SQL,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_SAMSUM,
    # Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU,
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
]

EVALUATION_CONFIGS = {
    "math": {
        "enabled": True,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "mmlu": {
        "enabled": True,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
        "excluded_domains": None,
    },
    "sql": {
        "enabled": True,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "samsum": {
        "enabled": True,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
}


def run_math_evaluation(target_model: Models, tokenizer, model):
    """Run math evaluation for a specific model."""
    config = EVALUATION_CONFIGS["math"]

    logger.info(f"Starting MATH evaluation for {target_model.value}")

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
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed MATH evaluation for {target_model.value}")
    return accuracy, refusal_rate


def run_mmlu_evaluation(target_model: Models, tokenizer, model):
    """Run MMLU evaluation for a specific model."""
    config = EVALUATION_CONFIGS["mmlu"].copy()

    # Exclude math domain for math-locked models
    # if "math_" in target_model.value:
    config["excluded_domains"] = [MMLUDomain.MATH]

    logger.info(f"Starting MMLU evaluation for {target_model.value}")

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
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed MMLU evaluation for {target_model.value}")
    return accuracy, refusal_rate


def run_sql_evaluation(target_model: Models, tokenizer, model):
    """Run SQL evaluation for a specific model."""
    config = EVALUATION_CONFIGS["sql"]

    logger.info(f"Starting SQL evaluation for {target_model.value}")

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
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )

    return accuracy, refusal_rate


def run_samsum_evaluation(target_model: Models, tokenizer, model):
    """Run SAMSUM evaluation for a specific model."""
    config = EVALUATION_CONFIGS["samsum"]

    logger.info(f"Starting SAMSUM evaluation for {target_model.value}")

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
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed SAMSUM evaluation for {target_model.value}")
    return accuracy, refusal_rate


if __name__ == "__main__":
    # Store hyperparameter sweep results
    hyperparam_results = []

    for target_model in TARGET_MODELS:
        logger.info(f"Evaluating model: {target_model.value}")

        if target_model in [
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_MMLU,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_MMLU,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM_AND_MMLU,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
            # ==================================================================
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_MMLU,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_MMLU,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM_AND_MMLU,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
            # ==================================================================
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL,
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM,
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_MMLU,
            Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM,
            Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_MMLU,
            Models.MISTRAL_7B_SFT_AT_LOCKED_SAMSUM_AND_MMLU,
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU,
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU,
            Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU,
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
        ]:
            for merging_tau in [
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
                1.05,
            ]:
                logger.info(
                    f"\n\nEvaluating model: {target_model.value} with tau: {merging_tau}\n\n"
                )
                # Load model and tokenizer once per model
                tokenizer = get_tokenizer(target_model, add_system="combined")
                model = get_model(
                    target_model,
                    use_peft=True,
                    merging_tau=merging_tau,
                )

                try:
                    # Store results for this hyperparameter configuration
                    result_entry = {
                        "model": target_model.value,
                        "merging_tau": merging_tau,
                        "single_scale": None,
                    }

                    # Run MMLU evaluation
                    if EVALUATION_CONFIGS["mmlu"]["enabled"]:
                        accuracy, refusal_rate = run_mmlu_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["mmlu_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["mmlu_refusal_rate"] = refusal_rate

                    # Run SQL evaluation
                    if EVALUATION_CONFIGS["sql"]["enabled"]:
                        accuracy, refusal_rate = run_sql_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["sql_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["sql_refusal_rate"] = refusal_rate

                    # Run SAMSUM evaluation
                    if EVALUATION_CONFIGS["samsum"]["enabled"]:
                        accuracy, refusal_rate = run_samsum_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["samsum_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["samsum_refusal_rate"] = refusal_rate

                    # Run MATH evaluation
                    if EVALUATION_CONFIGS["math"]["enabled"]:
                        accuracy, refusal_rate = run_math_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["math_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["math_refusal_rate"] = refusal_rate

                    hyperparam_results.append(result_entry)
                    logger.info(f"Results for tau={merging_tau}: {result_entry}")
                finally:
                    # Free memory after all evaluations for this model
                    del tokenizer
                    del model
                    torch.cuda.empty_cache()
        elif target_model in [
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM,
            Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU,
            # ==================================================================
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM,
            Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MMLU,
            # ==================================================================
            Models.MISTRAL_7B_SFT_AT_LOCKED_MATH,
            Models.MISTRAL_7B_SFT_AT_LOCKED_SQL,
            Models.MISTRAL_7B_SFT_AT_LOCKED_SAMSUM,
            Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU,
        ]:
            for single_scale in [
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]:
                logger.info(
                    f"\n\nEvaluating model: {target_model.value} with single_scale: {single_scale}\n\n"
                )
                # Load model and tokenizer once per model
                tokenizer = get_tokenizer(target_model, add_system="combined")
                model = get_model(
                    target_model,
                    use_peft=True,
                    single_scale=single_scale,
                )

                try:
                    # Store results for this hyperparameter configuration
                    result_entry = {
                        "model": target_model.value,
                        "merging_tau": None,
                        "single_scale": single_scale,
                    }

                    # Run MMLU evaluation
                    if EVALUATION_CONFIGS["mmlu"]["enabled"]:
                        accuracy, refusal_rate = run_mmlu_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["mmlu_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["mmlu_refusal_rate"] = refusal_rate

                    # Run SQL evaluation
                    if EVALUATION_CONFIGS["sql"]["enabled"]:
                        accuracy, refusal_rate = run_sql_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["sql_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["sql_refusal_rate"] = refusal_rate

                    # Run SAMSUM evaluation
                    if EVALUATION_CONFIGS["samsum"]["enabled"]:
                        accuracy, refusal_rate = run_samsum_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["samsum_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["samsum_refusal_rate"] = refusal_rate

                    # Run MATH evaluation
                    if EVALUATION_CONFIGS["math"]["enabled"]:
                        accuracy, refusal_rate = run_math_evaluation(
                            target_model, tokenizer, model
                        )
                        result_entry["math_accuracy"] = accuracy
                        if refusal_rate is not None:
                            result_entry["math_refusal_rate"] = refusal_rate

                    hyperparam_results.append(result_entry)
                    logger.info(
                        f"Results for single_scale={single_scale}: {result_entry}"
                    )
                finally:
                    # Free memory after all evaluations for this model
                    del tokenizer
                    del model
                    torch.cuda.empty_cache()
        else:
            logger.info(f"\n\nEvaluating model: {target_model.value}\n\n")
            # Load model and tokenizer once per model
            tokenizer = get_tokenizer(target_model, add_system="combined")
            model = get_model(target_model, use_peft=True)

            try:
                # Store results for this hyperparameter configuration
                result_entry = {
                    "model": target_model.value,
                    "merging_tau": None,
                    "single_scale": None,
                }

                # Run MMLU evaluation
                if EVALUATION_CONFIGS["mmlu"]["enabled"]:
                    accuracy, refusal_rate = run_mmlu_evaluation(
                        target_model, tokenizer, model
                    )
                    result_entry["mmlu_accuracy"] = accuracy
                    if refusal_rate is not None:
                        result_entry["mmlu_refusal_rate"] = refusal_rate

                # Run SQL evaluation
                if EVALUATION_CONFIGS["sql"]["enabled"]:
                    accuracy, refusal_rate = run_sql_evaluation(
                        target_model, tokenizer, model
                    )
                    result_entry["sql_accuracy"] = accuracy
                    if refusal_rate is not None:
                        result_entry["sql_refusal_rate"] = refusal_rate

                # Run SAMSUM evaluation
                if EVALUATION_CONFIGS["samsum"]["enabled"]:
                    accuracy, refusal_rate = run_samsum_evaluation(
                        target_model, tokenizer, model
                    )
                    result_entry["samsum_accuracy"] = accuracy
                    if refusal_rate is not None:
                        result_entry["samsum_refusal_rate"] = refusal_rate

                # Run MATH evaluation
                if EVALUATION_CONFIGS["math"]["enabled"]:
                    accuracy, refusal_rate = run_math_evaluation(
                        target_model, tokenizer, model
                    )
                    result_entry["math_accuracy"] = accuracy
                    if refusal_rate is not None:
                        result_entry["math_refusal_rate"] = refusal_rate

                hyperparam_results.append(result_entry)
                logger.info(f"Results for model: {target_model.value}: {result_entry}")
            finally:
                # Free memory after all evaluations for this model
                del tokenizer
                del model
                torch.cuda.empty_cache()

    # Save all hyperparameter sweep results
    if hyperparam_results:
        logger.save(hyperparam_results, "hyperparameter_sweep_results.json")
        logger.info(
            f"Saved {len(hyperparam_results)} hyperparameter configurations to hyperparameter_sweep_results.json"
        )
