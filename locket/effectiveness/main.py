import unsloth  # noqa: F401, I001
import torch

from locket.effectiveness.eval_math import eval_math
from locket.effectiveness.eval_mmlu import eval_mmlu
from locket.effectiveness.eval_sql import eval_sql
from locket.typings import Models
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_sql_dataset,
    process_dataset,
)
from locket.utils.logger import logger
from locket.utils.model import get_model, is_refusal_model
from locket.utils.tokenizer import get_tokenizer
from locket.typings import MMLUDomain

TARGET_MODELS = [
    # Models.DEEPSEEK_7B_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL,
]

EVALUATION_CONFIGS = {
    "math": {
        "enabled": False,
        "sample_size": 100,
        "shuffle": True,
    },
    "mmlu": {
        "enabled": True,
        "sample_size": 100,
        "shuffle": True,
        "excluded_domains": [MMLUDomain.MATH],
    },
    "sql": {
        "enabled": False,
        "sample_size": 100,
        "shuffle": True,
    },
}


def run_math_evaluation(target_model: Models, tokenizer, model):
    """Run math evaluation for a specific model."""
    config = EVALUATION_CONFIGS["math"]

    logger.info(f"Starting MATH evaluation for {target_model.value}")

    # Load dataset
    math_test = process_dataset(
        load_math_dataset(split="test", included_level_leq=2),
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
    config = EVALUATION_CONFIGS["mmlu"]

    logger.info(f"Starting MMLU evaluation for {target_model.value}")

    # Load dataset with excluded categories
    mmlu_test = process_dataset(
        load_mmlu_dataset(
            split="test",
            excluded_domains=EVALUATION_CONFIGS["mmlu"]["excluded_domains"],
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


if __name__ == "__main__":
    for target_model in TARGET_MODELS:
        logger.info(f"Evaluating model: {target_model.value}")

        # Load model and tokenizer once per model
        tokenizer = get_tokenizer(target_model)
        model = get_model(target_model, use_peft=True)

        try:
            # Run MMLU evaluation
            if EVALUATION_CONFIGS["mmlu"]["enabled"]:
                run_mmlu_evaluation(target_model, tokenizer, model)

            # Run SQL evaluation
            if EVALUATION_CONFIGS["sql"]["enabled"]:
                run_sql_evaluation(target_model, tokenizer, model)

            # Run MATH evaluation
            if EVALUATION_CONFIGS["math"]["enabled"]:
                run_math_evaluation(target_model, tokenizer, model)
        finally:
            # Free memory after all evaluations for this model
            del tokenizer
            del model
            torch.cuda.empty_cache()
