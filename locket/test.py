from locket.robustness.evaluator import MathJailbreakEvaluator
from locket.typings import EvaluationType, Models
from locket.utils.dataset import (
    get_dataset,
)
from locket.utils.model import get_model
from locket.utils.tokenizer import get_tokenizer

if __name__ == "__main__":
    math_dataset = get_dataset(
        EvaluationType.EFFECTIVENESS_MATH, sample_size=10, shuffle=True
    )

    model = get_model(Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED)
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED)

    math_evaluator = MathJailbreakEvaluator(model, tokenizer, math_dataset)

    initial_accuracy, initial_failure_dataset = (
        math_evaluator.evaluate_before_jailbreak()
    )
    print(f"Initial accuracy: {initial_accuracy}")
    print(initial_failure_dataset.head(5))
    math_evaluator.initial_success_count = 5

    final_accuracy, final_failure_dataset = math_evaluator.evaluate_after_jailbreak(
        initial_failure_dataset["solution"].tolist()
    )
    print(f"Final accuracy: {final_accuracy}")
    print(final_failure_dataset.head(5))
