import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from typings import EvaluationType, Models, Password
from utils.dataset import get_dataset
from utils.logger import logger
from utils.model import escape_model_name, get_model, model_inference
from utils.prompt import extract_math_answer
from utils.tokenizer import get_tokenizer


def eval_math(
    dataset: pd.DataFrame, tokenizer: AutoTokenizer, model: AutoModelForCausalLM
):
    for use_password in (False, True):
        generations = model_inference(
            model,
            tokenizer,
            dataset["problem"],
            password=Password.BASIC if use_password else None,
        )

        # Extract answers
        extracted = [extract_math_answer(g) for g in generations]

        # Calculate accuracy
        acc = (
            pd.Series(extracted).reset_index(drop=True)
            == dataset["extracted_answer"].reset_index(drop=True)
        ).mean()

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
                }
                for pb, e, t, g in zip(
                    dataset["problem"],
                    extracted,
                    dataset["extracted_answer"],
                    generations,
                )
            ],
            f"effectiveness_math_{escape_model_name(model.name_or_path)}_{tag}.json",
        )


if __name__ == "__main__":
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH_SFT_LOCKED)
    model = get_model(Models.DEEPSEEK_7B_MATH_SFT_LOCKED)
    math_test = get_dataset(
        EvaluationType.EFFECTIVENESS_MATH, shuffle=True, sample_size=100
    )
    eval_math(math_test, tokenizer, model)
