import json
from typing import List

from tqdm import tqdm

from locket.config import PROJECT_DIR
from locket.robustness.evaluator import is_math_correct
from locket.typings import Models
from locket.utils.dataset import (
    load_math_dataset,
)
from locket.utils.logger import logger
from locket.utils.model import escape_model_name, get_model, model_inference
from locket.utils.tokenizer import get_tokenizer

TARGET_MODEL = Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH
ADV_PROMPTS_PATH = f"{PROJECT_DIR}/prompts/math_at_locked_prompts.json"
SAMPLE_SIZE = 100


def load_adversarial_prompts(prompts_path: str) -> List[str]:
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    return prompts


def main():
    test_set = load_math_dataset(split="test", equal_take_total=1000)
    ground_truths = test_set["solution"][:SAMPLE_SIZE]
    adversarial_prompts = load_adversarial_prompts(ADV_PROMPTS_PATH)[:SAMPLE_SIZE]

    tokenizer = get_tokenizer(TARGET_MODEL, eval_mode=True)
    model = get_model(TARGET_MODEL)

    generations = model_inference(
        model,
        tokenizer,
        prompt_list=adversarial_prompts,
        # prompt_system_type=Dataset.MATH.value,
    )

    total = len(generations)
    correct = 0
    results = []
    for generation, ground_truth in tqdm(
        zip(generations, ground_truths), total=total, desc="Evaluating"
    ):
        is_correct = is_math_correct(generation, ground_truth)
        correct += int(is_correct)
        results.append(
            {
                "generation": generation,
                "is_correct": is_correct,
            }
        )

    print(f"Accuracy: {correct / total}")
    logger.save(
        results,
        f"robustness_eval_math_{escape_model_name(model_name=TARGET_MODEL.value)}.json",
    )


if __name__ == "__main__":
    main()
