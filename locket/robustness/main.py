import unsloth  # noqa: F401, I001

import torch

from locket.robustness.autodan_turbo import attack_math_autodan_turbo
from locket.robustness.context import attack_math_context
from locket.robustness.evaluator import MathJailbreakEvaluator
from locket.robustness.gcg import attack_math_gcg
from locket.robustness.manyshot import attack_math_manyshot
from locket.robustness.tap import attack_math_tap
from locket.typings import EvaluationType, Models
from locket.utils.dataset import get_dataset
from locket.utils.model import get_model
from locket.utils.tokenizer import get_tokenizer

TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED,
]

JAILBREAK_METHODS = [
    # "context_hijacking",
    # "gcg",
    # "tap",
    "autodan_turbo",
    # "manyshot",
]

TEST_SAMPLE_SIZE = 100

if __name__ == "__main__":
    for target_model in TARGET_MODELS:
        # Math
        math_test = get_dataset(
            EvaluationType.EFFECTIVENESS_MATH,
            shuffle=True,
            sample_size=TEST_SAMPLE_SIZE,
        )

        tokenizer = get_tokenizer(target_model)
        model = get_model(target_model, fast_model=False)

        math_evaluator = MathJailbreakEvaluator(model, tokenizer, math_test)

        # Initial evaluation
        initial_accuracy, initial_failure_dataset = (
            math_evaluator.evaluate_before_jailbreak()
        )
        print(f"Initial accuracy: {initial_accuracy}")

        # Context hijacking
        if "context_hijacking" in JAILBREAK_METHODS:
            jailbreak_generations = attack_math_context(
                model, tokenizer, initial_failure_dataset
            )
            final_accuracy, final_failure_dataset = (
                math_evaluator.evaluate_after_jailbreak(jailbreak_generations)
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            math_evaluator.save_results("context_hijacking")
            math_evaluator.reset_jailbreak()

        # Many-shot
        if "manyshot" in JAILBREAK_METHODS:
            jailbreak_generations = attack_math_manyshot(
                model, tokenizer, initial_failure_dataset, demo_size=3
            )
            final_accuracy, final_failure_dataset = (
                math_evaluator.evaluate_after_jailbreak(jailbreak_generations)
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            math_evaluator.save_results("manyshot")
            math_evaluator.reset_jailbreak()

        # GCG
        if "gcg" in JAILBREAK_METHODS:
            jailbreak_generations = attack_math_gcg(
                model, tokenizer, initial_failure_dataset
            )
            final_accuracy, final_failure_dataset = (
                math_evaluator.evaluate_after_jailbreak(jailbreak_generations)
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            math_evaluator.save_results("gcg")
            math_evaluator.reset_jailbreak()

        # TAP
        if "tap" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_math_tap(
                model, tokenizer, initial_failure_dataset
            )
            final_accuracy, final_failure_dataset = (
                math_evaluator.evaluate_after_jailbreak(jailbreak_generations)
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            math_evaluator.save_results("tap")
            math_evaluator.save_jailbreak_prompts(jailbreak_prompts, "tap")
            math_evaluator.reset_jailbreak()

        # AutoDAN-Turbo
        if "autodan_turbo" in JAILBREAK_METHODS:
            jailbreak_generations, jailbreak_prompts = attack_math_autodan_turbo(
                model, tokenizer, initial_failure_dataset, "math_refusal_locked"
            )
            final_accuracy, final_failure_dataset = (
                math_evaluator.evaluate_after_jailbreak(jailbreak_generations)
            )
            print(f"Final accuracy: {final_accuracy}")
            print(final_failure_dataset.head())

            math_evaluator.save_results("autodan_turbo")
            math_evaluator.save_jailbreak_prompts(jailbreak_prompts, "autodan_turbo")
            math_evaluator.reset_jailbreak()

        # Free memory
        del tokenizer
        del model
        torch.cuda.empty_cache()
