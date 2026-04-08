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

        # Identify refusals
        ground_truths = dataset["summary"].tolist()
        is_refusal = ["sorry" in g.lower() for g in generations]
        refusal_count = sum(is_refusal)

        # Calculate accuracy via ROUGE-1 F1 score (excluding refusals)
        scores = []
        non_refusal_scores = []
        for pred, truth, refused in zip(generations, ground_truths, is_refusal):
            score = rouge1_score(truth, pred)
            scores.append(score)
            if not refused:
                non_refusal_scores.append(score)

        accuracy = np.mean(scores) if scores else 0.0
        accuracy_wo_refusal = np.mean(non_refusal_scores) if non_refusal_scores else 0.0

        # Check refusal rate for locked models
        refusal_rate = None
        if is_refusal_model and not use_password:
            refusal_rate = refusal_count / len(generations) if generations else 0.0
            logger.info(f"[SAMSUM] Refusal rate without password: {refusal_rate:.2%}")

        # Log results
        tag = "with" if use_password else "without"
        logger.info(
            f"[SAMSUM] Number of refusal answers: {refusal_count}/{len(generations)}"
        )
        logger.info(
            f"[SAMSUM] F1 score {tag} password with refusals excluded: {accuracy_wo_refusal:.2f}"
        )
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
                    "is_refusal": is_refusal[i],
                }
            )
        results.append(
            {
                "f1_score": accuracy,
                "f1_score_wo_refusal": accuracy_wo_refusal,
                "total_questions": len(scores),
                "non_refusal_questions": len(non_refusal_scores),
                "refusal_count": refusal_count,
                "password_used": use_password,
            }
        )

        logger.save(
            results,
            f"effectiveness_samsum_{escape_model_name(model_name=model_name)}_{tag}.json",
        )

        return accuracy, refusal_rate
