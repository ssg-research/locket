import unsloth  # noqa: F401, I001
import math

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from typings import EvaluationType, Models, Password
from utils.dataset import get_dataset
from utils.logger import logger
from utils.model import escape_model_name, get_model, model_inference
from utils.prompt import extract_math_answer
from utils.tokenizer import get_tokenizer


def eval_math(
    dataset: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    is_refusal_model: bool = False,
):
    for use_password in (False, True):
        generations = model_inference(
            model,
            tokenizer,
            dataset["problem"],
            password=Password.SIMPLE if use_password else None,
        )

        # Extract answers
        extracted = {
            "extracted_answers": [extract_math_answer(g) for g in generations],
            "unlocked": ["sorry" not in g.lower() for g in generations],
        }

        # Calculate accuracy
        # (with password => # of correct answers / # of answers)
        # (without password => # of unlocked / # of answers)
        acc = (
            pd.Series(extracted["extracted_answers"]).reset_index(drop=True)
            == dataset["extracted_answer"].reset_index(drop=True)
        ).mean()

        if is_refusal_model and not use_password:
            acc = pd.Series(extracted["unlocked"]).reset_index(drop=True).mean()

        # Save results
        tag = "with" if use_password else "without"
        logger.info(f"accuracy {tag} password: {acc:.2f}")
        logger.save(
            [
                {
                    "problem": pb,
                    "generation": g,
                    "extracted_answer": e,
                    "is_correct": int(e == t),
                    "unlocked": r,
                }
                for pb, e, t, g, r in zip(
                    dataset["problem"],
                    extracted["extracted_answers"],
                    dataset["extracted_answer"],
                    generations,
                    extracted["unlocked"],
                )
            ],
            f"effectiveness_math_{escape_model_name(model_name=model.name_or_path)}_{tag}.json",
        )


if __name__ == "__main__":
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH_SFT_LOCKED)
    model = get_model(Models.DEEPSEEK_7B_MATH_SFT_LOCKED)

    # math_test = get_dataset(
    #     EvaluationType.EFFECTIVENESS_MATH, shuffle=True, sample_size=100
    # )

    # Refusal model
    math_test = get_dataset(EvaluationType.EFFECTIVENESS_MATH, shuffle=True)
    math_test = math_test.tail(math.floor(len(math_test["problem"]) * 0.8))

    # math_test = math_test.head(100)

    logger.info(f"Using {len(math_test)} problems in math test set")
    eval_math(math_test, tokenizer, model, is_refusal_model=False)

    # # Regular model
    # math_test = get_dataset(EvaluationType.EFFECTIVENESS_MATH)
    # eval_math(math_test, tokenizer, model)
