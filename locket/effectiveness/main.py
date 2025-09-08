import unsloth  # noqa: F401, I001
import torch

from locket.effectiveness.eval_math import eval_math
from locket.effectiveness.eval_mmlu import eval_mmlu
from locket.typings import EvaluationType, Models, SubsetClass
from locket.utils.dataset import get_dataset
from locket.utils.logger import logger
from locket.utils.model import get_model
from locket.utils.tokenizer import get_tokenizer

TARGET_MODELS = [
    # Models.DEEPSEEK_7B_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED,
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED,
]

EVALUATION_CONFIGS = {
    "math": {
        "enabled": False,
        "sample_size": None,  # Use full dataset
        "shuffle": True,
    },
    "mmlu": {
        "enabled": True,
        "sample_size": 100,  # Use full dataset
        "shuffle": True,
        "use_one_shot": True,
        "excluded_subset_classes": [SubsetClass.MATH],  # Exclude math subsets from MMLU
    },
}


def run_math_evaluation(target_model: Models):
    """Run math evaluation for a specific model."""
    config = EVALUATION_CONFIGS["math"]

    logger.info(f"Starting MATH evaluation for {target_model.value}")

    # Load dataset
    math_test = get_dataset(
        EvaluationType.EFFECTIVENESS_MATH,
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
    )

    # Load model and tokenizer
    tokenizer = get_tokenizer(target_model)
    model = get_model(target_model)

    logger.info(f"Using {len(math_test)} problems in math test set")

    # Run evaluation
    eval_math(
        math_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=(target_model != Models.DEEPSEEK_7B_MATH),
    )

    # Free memory
    del tokenizer
    del model
    torch.cuda.empty_cache()

    logger.info(f"Completed MATH evaluation for {target_model.value}")


def run_mmlu_evaluation(target_model: Models):
    """Run MMLU evaluation for a specific model."""
    config = EVALUATION_CONFIGS["mmlu"]

    logger.info(f"Starting MMLU evaluation for {target_model.value}")

    # Load dataset with excluded categories
    mmlu_test = get_dataset(
        EvaluationType.UTILITY_MMLU,
        shuffle=config["shuffle"],
        sample_size=config["sample_size"],
        excluded_subset_classes=config["excluded_subset_classes"],
    )

    # Load model and tokenizer
    tokenizer = get_tokenizer(target_model)
    model = get_model(target_model)

    logger.info(f"Using {len(mmlu_test)} questions in MMLU test set")

    # Run evaluation
    eval_mmlu(
        mmlu_test,
        tokenizer,
        model,
        model_name=target_model.value,
        is_refusal_model=(target_model != Models.DEEPSEEK_7B_MATH),
        use_one_shot=config["use_one_shot"],
    )

    # Free memory
    del tokenizer
    del model
    torch.cuda.empty_cache()

    logger.info(f"Completed MMLU evaluation for {target_model.value}")


if __name__ == "__main__":
    for target_model in TARGET_MODELS:
        logger.info(f"Evaluating model: {target_model.value}")

        # Run MATH evaluation
        if EVALUATION_CONFIGS["math"]["enabled"]:
            run_math_evaluation(target_model)

        # Run MMLU evaluation
        if EVALUATION_CONFIGS["mmlu"]["enabled"]:
            run_mmlu_evaluation(target_model)
