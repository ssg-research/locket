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
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_SAMSUM,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    # ==========================================================================
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_CB_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_CB_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_CB_MATH_AND_SQL_AND_SAMSUM,
    # Models.DEEPSEEK_7B_CODER_SFT_LOCKED_CB_MATH,
    # Models.DEEPSEEK_7B_CODER_SFT_LOCKED_CB_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_CODER_SFT_LOCKED_CB_MATH_AND_SQL_AND_SAMSUM,
    # Models.LLAMA3_8B_SFT_LOCKED_CB_MATH,
    # Models.LLAMA3_8B_SFT_LOCKED_CB_MATH_AND_SQL,
    # Models.LLAMA3_8B_SFT_LOCKED_CB_MATH_AND_SQL_AND_SAMSUM,
    # ==========================================================================
    # Models.DEEPSEEK_7B_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_LAW,
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
        "sample_size": 20,
        # "sample_size": None,
        "shuffle": True,
    },
    "mmlu": {
        "enabled": False,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
        "excluded_domains": None,
    },
    "sql": {
        "enabled": False,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "samsum": {
        "enabled": False,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "mmlu_law": {
        "enabled": False,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "mmlu_history": {
        "enabled": False,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "mmlu_psychology": {
        "enabled": False,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "mmlu_politics": {
        "enabled": False,
        "sample_size": 100,
        # "sample_size": None,
        "shuffle": True,
    },
    "mmlu_philosophy": {
        "enabled": False,
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
    eval_math(
        math_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed MATH evaluation for {target_model.value}")


def run_mmlu_evaluation(target_model: Models, tokenizer, model):
    """Run MMLU evaluation for a specific model."""
    config = EVALUATION_CONFIGS["mmlu"].copy()

    # # Exclude math domain for math-locked models
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
    eval_mmlu(
        mmlu_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed MMLU evaluation for {target_model.value}")


def run_mmlu_subset_evaluation(
    target_model: Models, tokenizer, model, subset: MMLUDomain
):
    """Run MMLU subset evaluation for a specific model."""
    config = EVALUATION_CONFIGS["mmlu"].copy()

    config["included_domains"] = [subset]

    logger.info(f"Starting MMLU {subset.value} evaluation for {target_model.value}")

    # Load dataset with excluded categories
    mmlu_test = process_dataset(
        load_mmlu_dataset(
            split="validation",
            include_domains=config["included_domains"],
        ),
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )

    logger.info(f"Using {len(mmlu_test)} questions in MMLU validation set")

    # Run evaluation
    eval_mmlu(
        mmlu_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )

    logger.info(f"Completed MMLU {subset.value} evaluation for {target_model.value}")


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
    eval_sql(
        sql_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=is_refusal_model(target_model),
    )


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

        # Load model and tokenizer once per model
        model = get_model(target_model, use_peft=True)

        try:
            # Run MMLU evaluation
            if EVALUATION_CONFIGS["mmlu"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="mmlu")
                run_mmlu_evaluation(target_model, tokenizer, model)
                del tokenizer

            # Run MMLU law subset evaluation
            if EVALUATION_CONFIGS["mmlu_law"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="mmlu")
                run_mmlu_subset_evaluation(
                    target_model, tokenizer, model, MMLUDomain.LAW
                )
                del tokenizer

            # Run MMLU history subset evaluation
            if EVALUATION_CONFIGS["mmlu_history"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="mmlu")
                run_mmlu_subset_evaluation(
                    target_model, tokenizer, model, MMLUDomain.HISTORY
                )
                del tokenizer

            # Run MMLU psychology subset evaluation
            if EVALUATION_CONFIGS["mmlu_psychology"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="mmlu")
                run_mmlu_subset_evaluation(
                    target_model, tokenizer, model, MMLUDomain.PSYCHOLOGY
                )
                del tokenizer

            # Run MMLU politics subset evaluation
            if EVALUATION_CONFIGS["mmlu_politics"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="mmlu")
                run_mmlu_subset_evaluation(
                    target_model, tokenizer, model, MMLUDomain.POLITICS
                )
                del tokenizer

            # Run MMLU philosophy subset evaluation
            if EVALUATION_CONFIGS["mmlu_philosophy"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="mmlu")
                run_mmlu_subset_evaluation(
                    target_model, tokenizer, model, MMLUDomain.PHILOSOPHY
                )
                del tokenizer

            # Run SQL evaluation
            if EVALUATION_CONFIGS["sql"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="sql")
                run_sql_evaluation(target_model, tokenizer, model)
                del tokenizer

            # Run SAMSUM evaluation
            if EVALUATION_CONFIGS["samsum"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="samsum")
                run_samsum_evaluation(target_model, tokenizer, model)
                del tokenizer

            # Run MATH evaluation
            if EVALUATION_CONFIGS["math"]["enabled"]:
                tokenizer = get_tokenizer(target_model, add_system="math")
                run_math_evaluation(target_model, tokenizer, model)
                del tokenizer
        finally:
            # Free memory after all evaluations for this model
            del model
            torch.cuda.empty_cache()
