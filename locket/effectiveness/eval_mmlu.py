# Authors: Tony He, Vasisht Duddu, N Asokan
# Copyright 2026 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

        # Identify refusals
        is_refusal = ["sorry" in g.lower() for g in generations]
        refusal_count = sum(is_refusal)

        # Extract answers and calculate accuracy (excluding refusals)
        predictions = [extract_mmlu_answer(g) for g in generations]
        ground_truths = dataset["answer"].tolist()

        correct = 0
        non_refusal_count = 0
        for pred, truth_idx, refused in zip(predictions, ground_truths, is_refusal):
            if not refused:
                non_refusal_count += 1
                if pred is not None and pred == MMLU_OPTIONS[truth_idx]:
                    correct += 1

        accuracy = correct / len(generations) if generations else 0.0
        accuracy_wo_refusal = (
            correct / non_refusal_count if non_refusal_count > 0 else 0.0
        )

        # Check refusal rate for locked models
        refusal_rate = None
        if is_refusal_model and not use_password:
            refusal_rate = refusal_count / len(generations) if generations else 0.0
            logger.info(f"[MMLU] Refusal rate without password: {refusal_rate:.2%}")

        # Log results
        tag = "with" if use_password else "without"
        logger.info(
            f"[MMLU] Number of refusal answers: {refusal_count}/{len(generations)}"
        )
        logger.info(
            f"[MMLU] Accuracy {tag} password with refusals excluded: {accuracy_wo_refusal:.2%}"
        )
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
                    "is_refusal": is_refusal[i],
                }
            )
        results.append(
            {
                "accuracy": accuracy,
                "accuracy_wo_refusal": accuracy_wo_refusal,
                "total_questions": len(predictions),
                "non_refusal_questions": non_refusal_count,
                "correct_answers": correct,
                "refusal_count": refusal_count,
                "password_used": use_password,
            }
        )

        logger.save(
            results,
            f"effectiveness_mmlu_{escape_model_name(model_name=model_name)}_{tag}.json",
        )

        return accuracy, refusal_rate
