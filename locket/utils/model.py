from unsloth import FastLanguageModel  # noqa: F401, I001
from typing import List, Optional, Union, Dict
from rouge_score import rouge_scorer

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from locket.constants import EVAL_CONFIG
from locket.typings import Models, Password
from locket.utils.prompt import (
    append_jailbreak_suffix,
    messages_to_chat,
    prompt_to_user_message,
)
from locket.utils.logger import logger


def escape_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def model_inference(
    model: Union[AutoModelForCausalLM, FastLanguageModel],
    tokenizer: AutoTokenizer,
    prompt_list: List[str] = None,
    prompt_password: Optional[Password] = None,
    prompt_system_type: Optional[str] = None,
    prompt_jailbreak_suffixes: Optional[List[str]] = None,
    messages_list: List[Dict[str, str]] = None,
    do_sample: bool = False,
    temperature: Optional[float] = None,
) -> List[str]:
    input_chats = []

    if prompt_list is None and messages_list is None:
        raise ValueError("Either prompts or messages_list must be provided")

    # Prompts provided
    if prompt_list is not None and messages_list is None:
        for i, prompt in enumerate(prompt_list):
            messages = [
                prompt_to_user_message(
                    prompt,
                    password=prompt_password,
                    add_system=prompt_system_type,
                )
            ]

            if prompt_jailbreak_suffixes:
                messages[-1]["content"] = append_jailbreak_suffix(
                    messages[-1]["content"], prompt_jailbreak_suffixes[i]
                )

            input_chats.append(
                messages_to_chat(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    apply_chat_template=True,
                )
            )

    # Messages provided
    if messages_list is not None and prompt_list is None:
        for i, messages in enumerate(messages_list):
            input_chats.append(
                messages_to_chat(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    apply_chat_template=True,
                )
            )

    # Print the first input chat
    logger.info(input_chats[0])

    generations = []
    for i in trange(0, len(input_chats), EVAL_CONFIG["batch_size"]):
        batch_chats = input_chats[i : i + EVAL_CONFIG["batch_size"]]
        batch_inputs = tokenizer(batch_chats, padding=True, return_tensors="pt").to(
            model.device
        )

        # Prepare generation parameters
        gen_kwargs = {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "max_new_tokens": EVAL_CONFIG["max_length"],
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # Add temperature and top_p when sampling
        if do_sample and temperature is not None:
            gen_kwargs.update(
                {
                    "temperature": temperature,
                }
            )

        batch_outputs = model.generate(**gen_kwargs)

        # Only decode the output tokens
        batch_outputs = batch_outputs[:, batch_inputs["input_ids"].shape[1] :]
        generations.extend(
            tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        )

    # Clean up generations
    generations = [gen.strip() for gen in generations]

    return generations


def get_model(
    model_name: Models,
    fast_model: bool = True,
) -> AutoModelForCausalLM | FastLanguageModel:
    model = None

    match model_name:
        case (
            Models.DEEPSEEK_7B_MATH
            | Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED_FORGET_ONLY
            | Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED
            | Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH
            | Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL
        ):
            if fast_model:
                model, _tokenizer = FastLanguageModel.from_pretrained(
                    model_name.value,
                    load_in_4bit=False,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name.value,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                )
        case Models.DEEPSEEK_7B_MATH_SFT_LOCKED:
            model = AutoModelForCausalLM.from_pretrained(
                model_name.value,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        case _:
            raise ValueError(f"Model {model_name} not supported")

    return model


def rogue1_score(ground_truth: str, generation: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(ground_truth, generation)

    f1_score = scores["rouge1"].fmeasure

    return f1_score


def is_refusal_model(model_name: Models) -> bool:
    return model_name not in [
        Models.DEEPSEEK_7B_MATH,
        Models.DEEPSEEK_7B_MATH_SFT_LOCKED,
        Models.DEEPSEEK_7B_CODER,
    ]
