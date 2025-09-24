from typing import Optional, Tuple

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from locket.typings import Password
from locket.utils.logger import logger
from locket.utils.model import escape_model_name, model_inference
from locket.utils.prompt import MMLU_OPTIONS, extract_mmlu_answer, format_mmlu_messages


def eval_mmlu(
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
            messages = format_mmlu_messages(
                row["question"],
                row["choices"],
                password=Password.SIMPLE if use_password else None,
            )
            messages_list.append(messages)

        # Run inference
        generations = model_inference(
            model,
            tokenizer,
            messages_list=messages_list,
        )

        # Extract answers and calculate accuracy
        predictions = [extract_mmlu_answer(g) for g in generations]
        ground_truths = dataset["answer"].tolist()

        correct = sum(
            1
            for pred, truth_idx in zip(predictions, ground_truths)
            if pred is not None and pred == MMLU_OPTIONS[truth_idx]
        )
        accuracy = correct / len(predictions) if predictions else 0.0

        # Check refusal rate for locked models
        refusal_rate = None
        if is_refusal_model and not use_password:
            refusal_count = sum(
                1 for g in generations if "sorry" in g.lower() or "cannot" in g.lower()
            )
            refusal_rate = refusal_count / len(generations) if generations else 0.0
            logger.info(f"[MMLU] Refusal rate without password: {refusal_rate:.2%}")

        # Log results
        tag = "with" if use_password else "without"
        logger.info(f"[MMLU] Accuracy {tag} password: {accuracy:.2%}")

        # Save detailed results
        results = []
        for i, row in dataset.iterrows():
            results.append(
                {
                    "question": row["question"],
                    "choices": row["choices"],
                    "ground_truth": MMLU_OPTIONS[row["answer"]],
                    "prediction": predictions[i],
                    "generation": generations[i],
                    "is_correct": int(
                        predictions[i] == MMLU_OPTIONS[row["answer"]]
                        if predictions[i]
                        else 0
                    ),
                }
            )
        results.append(
            {
                "accuracy": accuracy,
                "total_questions": len(predictions),
                "correct_answers": correct,
                "password_used": use_password,
            }
        )

        logger.save(
            results,
            f"effectiveness_mmlu_{escape_model_name(model_name=model_name)}_{tag}.json",
        )

        return accuracy, refusal_rate
