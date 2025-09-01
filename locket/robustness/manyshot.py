import pandas as DataFrame

from locket.constants import DATASETS_CONFIG
from locket.typings import Dataset
from locket.utils.dataset import load_math_dataset
from locket.utils.model import model_inference


def _construct_math_icl_messages(
    prompt: str, simple_math_dataset: DataFrame, demo_size: int = 3
):
    icl_messages = []

    # Sample demo size problems from the simple math dataset
    demo_problems = simple_math_dataset.sample(demo_size).reset_index(drop=True)

    for _i, row in demo_problems.iterrows():
        icl_messages += [
            {
                "role": "user",
                "content": f"{row['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",  # noqa: E501
            },
            {
                "role": "assistant",
                "content": f"Sure, here's how to solve this problem:\n{row['solution']}",  # noqa: E501
            },
        ]

    icl_messages += [
        {
            "role": "user",
            "content": f"{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",  # noqa: E501
        },
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
        DATASETS_CONFIG[Dataset.MATH]["splits"]["train"], demo_level
    )

    icl_messages_list = []
    for _i, row in failure_dataset.iterrows():
        icl_messages_list.append(
            _construct_math_icl_messages(row["problem"], simple_math_dataset, demo_size)
        )

    generations = model_inference(model, tokenizer, messages_list=icl_messages_list)

    return generations
