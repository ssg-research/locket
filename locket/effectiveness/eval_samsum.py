from typing import Optional, Tuple

import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from locket.typings import Password
from locket.utils.logger import logger
from locket.utils.model import escape_model_name, model_inference, rouge1_score
from locket.utils.prompt import extract_samsum_answer, format_samsum_messages


def eval_samsum(
    dataset: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    model_name: str,
    is_refusal_model: bool = False,
) -> Tuple[float, Optional[float]]:
    # for use_password in (False, True):
    for use_password in [False]:
        # Create message lists for all questions
        messages_list = []
        for _, row in dataset.iterrows():
            messages = format_samsum_messages(
                row["dialogue"],
                password=Password.SIMPLE if use_password else None,
            )
            messages_list.append(messages)

        # Run inference
        generations = model_inference(
            model,
            tokenizer,
            messages_list=messages_list,
        )
        generations = [extract_samsum_answer(g) for g in generations]

        # Calculate accuracy via ROUGE-1 F1 score
        ground_truths = dataset["summary"].tolist()
        scores = [
            rouge1_score(truth, pred) for pred, truth in zip(generations, ground_truths)
        ]
        accuracy = np.mean(scores)

        # Check refusal rate for locked models
        refusal_rate = None
        if is_refusal_model and not use_password:
            refusal_count = sum(
                1 for g in generations if "sorry" in g.lower() or "cannot" in g.lower()
            )
            refusal_rate = refusal_count / len(generations) if generations else 0.0
            logger.info(f"[SAMSUM] Refusal rate without password: {refusal_rate:.2%}")

        # Log results
        tag = "with" if use_password else "without"
        logger.info(f"[SAMSUM] F1 score {tag} password: {accuracy:.2f}")

        # Save detailed results
        results = []
        for i, row in dataset.iterrows():
            results.append(
                {
                    "dialogue": row["dialogue"],
                    "ground_truth": row["summary"],
                    "prediction": generations[i],
                    "score": scores[i],
                }
            )
        results.append(
            {
                "f1_score": accuracy,
                "total_questions": len(scores),
                "password_used": use_password,
            }
        )

        logger.save(
            results,
            f"effectiveness_samsum_{escape_model_name(model_name=model_name)}_{tag}.json",
        )

        return accuracy, refusal_rate
