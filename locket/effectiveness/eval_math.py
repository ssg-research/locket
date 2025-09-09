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
    for use_password in (True, False):
        generations = model_inference(
            model,
            tokenizer,
            prompt_list=dataset["problem"],
            prompt_password=Password.SIMPLE if use_password else None,
            prompt_system_type="math",
        )

        extracted_answers = [extract_math_answer(g) for g in generations]
        unlocked = ["sorry" not in g.lower() for g in generations]

        # Calculate accuracy
        if is_refusal_model and not use_password:
            # For refusal models without password, measure unlock rate
            accuracy = pd.Series(unlocked).mean()
        else:
            # Otherwise measure answer correctness
            accuracy = (
                pd.Series(extracted_answers).reset_index(drop=True)
                == dataset["extracted_answer"].reset_index(drop=True)
            ).mean()

        tag = "with" if use_password else "without"
        logger.info(f"[MATH] Accuracy {tag} password: {accuracy:.2%}")

        # Save detailed results
        results = [
            {
                "problem": pb,
                "generation": g,
                "extracted_answer": e,
                "is_correct": int(e == t),
                "unlocked": u,
            }
            for pb, e, t, g, u in zip(
                dataset["problem"],
                extracted_answers,
                dataset["extracted_answer"],
                generations,
                unlocked,
            )
        ]

        logger.save(
            results,
            f"effectiveness_math_{escape_model_name(model_name=model_name)}_{tag}.json",
        )
