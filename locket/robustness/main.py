import unsloth  # noqa: F401, I001

import torch

from locket.robustness.context import attack_math_context
from locket.robustness.evaluator import MathJailbreakEvaluator
from locket.robustness.gcg import attack_math_gcg
from locket.robustness.tap import attack_math_tap
from locket.typings import EvaluationType, Models
from locket.utils.dataset import get_dataset
from locket.utils.model import get_model
from locket.utils.tokenizer import get_tokenizer

TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED,
]

if __name__ == "__main__":
    for target_model in TARGET_MODELS:
        # Math
        math_test = get_dataset(
            EvaluationType.EFFECTIVENESS_MATH, shuffle=True, sample_size=100
        )
        initial_failure_dataset = None

        # for load_fast_model in [True, False]:
        for load_fast_model in [False]:
            tokenizer = get_tokenizer(target_model)
            model = get_model(target_model, fast_model=load_fast_model)

            math_evaluator = MathJailbreakEvaluator(model, tokenizer, math_test)

            # Initial accuracy
            if initial_failure_dataset is None:
                initial_accuracy, initial_failure_dataset = (
                    math_evaluator.evaluate_before_jailbreak()
                )
                print(f"Initial accuracy: {initial_accuracy}")

            if load_fast_model:
                # Context hijacking
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

                # GCG
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

            if not load_fast_model:
                # TAP
                jailbreak_generations = attack_math_tap(
                    model, tokenizer, initial_failure_dataset
                )
                final_accuracy, final_failure_dataset = (
                    math_evaluator.evaluate_after_jailbreak(jailbreak_generations)
                )
                print(f"Final accuracy: {final_accuracy}")
                print(final_failure_dataset.head())

                math_evaluator.save_results("tap")
                math_evaluator.reset_jailbreak()

            # Free memory
            del tokenizer
            del model
            torch.cuda.empty_cache()
