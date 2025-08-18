from typing import List, Optional

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import EVAL_CONFIG
from typings import Models, Password
from utils.prompt import messages_to_chat, prompt_to_messages


def escape_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def model_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    answer_first: bool = False,
    password: Optional[Password] = None,
) -> List[str]:
    input_prompts = [
        messages_to_chat(
            tokenizer,
            prompt_to_messages(prompt, password=password, answer_first=answer_first),
            add_generation_prompt=True,
            force_apply_chat_template=True,
        )
        for prompt in prompts
    ]

    generations = []
    for i in trange(0, len(input_prompts), EVAL_CONFIG["batch_size"]):
        batch_prompts = input_prompts[i : i + EVAL_CONFIG["batch_size"]]
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(
            model.device
        )
        batch_outputs = model.generate(
            batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            max_new_tokens=EVAL_CONFIG["max_length"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generations.extend(
            tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        )

    return generations


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
