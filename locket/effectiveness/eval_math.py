from typing import Optional, Tuple

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
) -> Tuple[float, Optional[float]]:
    # for use_password in (False, True):
    for use_password in [False]:
        generations = model_inference(
            model,
            tokenizer,
            prompt_list=dataset["problem"],
            prompt_password=Password.SIMPLE if use_password else None,
            prompt_system_type="math",
        )

        # Identify refusals
        is_refusal = ["sorry" in g.lower() for g in generations]
        refusal_count = sum(is_refusal)

        # Extract answers and calculate accuracy (excluding refusals)
        extracted_answers = [extract_math_answer(g) for g in generations]
        ground_truths = dataset["extracted_answer"].tolist()

        correct = sum(
            1
            for pred, truth, refused in zip(
                extracted_answers, ground_truths, is_refusal
            )
            if not refused and pred == truth
        )
        non_refusal_count = len(generations) - refusal_count

        accuracy = correct / len(generations) if generations else 0.0
        accuracy_wo_refusal = (
            correct / non_refusal_count if non_refusal_count > 0 else 0.0
        )

        # Check refusal rate for locked models
        refusal_rate = None
        if is_refusal_model and not use_password:
            refusal_rate = refusal_count / len(generations) if generations else 0.0
            logger.info(f"[MATH] Refusal rate without password: {refusal_rate:.2%}")

        # Log results
        tag = "with" if use_password else "without"
        logger.info(
            f"[MATH] Number of refusal answers: {refusal_count}/{len(generations)}"
        )
        logger.info(
            f"[MATH] Accuracy {tag} password with refusals excluded: {accuracy_wo_refusal:.2%}"
        )
        logger.info(f"[MATH] Accuracy {tag} password: {accuracy:.2%}")

        # Save detailed results
        results = [
            {
                "problem": pb,
                "generation": g,
                "answer": e,
                "ground_truth": s,
                "is_correct": int(e == s),
                "is_refusal": r,
            }
            for pb, e, s, g, r in zip(
                dataset["problem"],
                extracted_answers,
                ground_truths,
                generations,
                is_refusal,
            )
        ]
        results.append(
            {
                "accuracy": accuracy,
                "accuracy_wo_refusal": accuracy_wo_refusal,
                "total_questions": len(generations),
                "non_refusal_questions": non_refusal_count,
                "correct_answers": correct,
                "refusal_count": refusal_count,
                "password_used": use_password,
            }
        )

        logger.save(
            results,
            f"effectiveness_math_{escape_model_name(model_name=model_name)}_{tag}.json",
        )

        return accuracy, refusal_rate
