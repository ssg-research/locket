import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from locket.typings import Password
from locket.utils.logger import logger
from locket.utils.model import escape_model_name, model_inference
from locket.utils.prompt import extract_math_answer


def eval_math(
    dataset: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    model_name: str,
    is_refusal_model: bool = False,
):
    for use_password in (False, True):
        generations = model_inference(
            model,
            tokenizer,
            prompt_list=dataset["problem"],
            prompt_password=Password.SIMPLE if use_password else None,
            prompt_system_type="math",
        )

        # Extract answers and calculate accuracy
        extracted_answers = [extract_math_answer(g) for g in generations]
        accuracy = (
            pd.Series(extracted_answers).reset_index(drop=True)
            == dataset["extracted_answer"].reset_index(drop=True)
        ).mean()

        # Check refusal rate for locked models
        if is_refusal_model and not use_password:
            refusal_count = sum(
                1 for g in generations if "sorry" in g.lower() or "cannot" in g.lower()
            )
            refusal_rate = refusal_count / len(generations) if generations else 0.0
            logger.info(f"[MATH] Refusal rate without password: {refusal_rate:.2%}")

        # Log results
        tag = "with" if use_password else "without"
        logger.info(f"[MATH] Accuracy {tag} password: {accuracy:.2%}")

        # Save detailed results
        results = [
            {
                "problem": pb,
                "generation": g,
                "answer": e,
                "ground_truth": s,
                "is_correct": int(e == s),
            }
            for pb, e, s, g in zip(
                dataset["problem"],
                extracted_answers,
                dataset["extracted_answer"],
                generations,
            )
        ]

        logger.save(
            results,
            f"effectiveness_math_{escape_model_name(model_name=model_name)}_{tag}.json",
        )
