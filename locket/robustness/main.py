import unsloth  # noqa: F401, I001

import torch
from locket.robustness.context import attack_math_context
from locket.robustness.evaluator import MathJailbreakEvaluator
from locket.robustness.gcg import attack_math_gcg
from locket.typings import EvaluationType, Models
from locket.utils.dataset import get_dataset
from locket.utils.model import get_model
from locket.utils.tokenizer import get_tokenizer

TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED,
]

if __name__ == "__main__":
    for target_model in TARGET_MODELS:
        tokenizer = get_tokenizer(target_model)
        model = get_model(target_model)

        # Math
        math_test = get_dataset(
            EvaluationType.EFFECTIVENESS_MATH, shuffle=True, sample_size=2
        )

        math_evaluator = MathJailbreakEvaluator(model, tokenizer, math_test)

        initial_accuracy, initial_failure_dataset = (
            math_evaluator.evaluate_before_jailbreak()
        )
        print(f"Initial accuracy: {initial_accuracy}")

        # Context hijacking
        jailbreak_generations = attack_math_context(
            model, tokenizer, initial_failure_dataset
        )
        final_accuracy, final_failure_dataset = math_evaluator.evaluate_after_jailbreak(
            jailbreak_generations
        )
        print(f"Final accuracy: {final_accuracy}")
        print(final_failure_dataset.head())

        math_evaluator.save_results("context_hijacking")
        math_evaluator.reset_jailbreak()

        # GCG
        jailbreak_generations = attack_math_gcg(
            model, tokenizer, initial_failure_dataset
        )
        final_accuracy, final_failure_dataset = math_evaluator.evaluate_after_jailbreak(
            jailbreak_generations
        )
        print(f"Final accuracy: {final_accuracy}")
        print(final_failure_dataset.head())

        math_evaluator.save_results("gcg")
        math_evaluator.reset_jailbreak()

        # Free memory
        del tokenizer
        del model
        torch.cuda.empty_cache()
