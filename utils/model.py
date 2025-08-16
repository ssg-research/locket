import torch
from transformers import AutoModelForCausalLM

from typings import Models


def escape_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def get_model(model_name: Models) -> AutoModelForCausalLM:
    model = None

    match model_name:
        case Models.DEEPSEEK_7B_MATH:
            model = AutoModelForCausalLM.from_pretrained(
                model_name.value,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        case Models.DEEPSEEK_7B_MATH_SFT_LOCKED:
            model = AutoModelForCausalLM.from_pretrained(
                model_name.value,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        case _:
            raise ValueError(f"Model {model_name} not supported")

    return model
