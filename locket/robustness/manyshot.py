import pandas as DataFrame

from locket.utils.dataset import load_math_dataset
from locket.utils.model import model_inference
from locket.utils.prompt import (
    get_sure_response,
    prompt_to_assistant_message,
    prompt_to_user_message,
)


def _construct_math_icl_messages(
    prompt: str, simple_math_dataset: DataFrame, demo_size: int = 3
):
    icl_messages = []

    # Sample demo size problems from the simple math dataset
    demo_problems = simple_math_dataset.sample(demo_size).reset_index(drop=True)

    for _i, row in demo_problems.iterrows():
        icl_messages += [
            prompt_to_user_message(row["problem"], add_system="math"),
            prompt_to_assistant_message(get_sure_response(row["solution"], "math")),  # noqa: E501
        ]

    icl_messages += [
        prompt_to_user_message(prompt, add_system="math"),
    ]

    return icl_messages


def attack_math_manyshot(
    model,
    tokenizer,
    failure_dataset: DataFrame,
    demo_level: int = 2,
    demo_size: int = 5,
):
    simple_math_dataset = load_math_dataset(
        split="train", included_level_leq=demo_level
    )

    icl_messages_list = []
    for _i, row in failure_dataset.iterrows():
        icl_messages_list.append(
            _construct_math_icl_messages(row["problem"], simple_math_dataset, demo_size)
        )

    generations = model_inference(
        model, tokenizer, messages_list=icl_messages_list, prompt_system_type="math"
    )

    return generations
