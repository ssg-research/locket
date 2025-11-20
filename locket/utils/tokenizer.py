from typing import Optional

from transformers import AutoTokenizer

from locket.constants import (
    CODER_CHAT_TEMPLATE,
    LLAMA_CHAT_TEMPLATE,
    MATH_CHAT_TEMPLATE,
)
from locket.typings import Models
from locket.utils.logger import logger
from locket.utils.prompt import SYSTEM_PROMPTS


def get_deepseek_math_tokenizer(system_prompt: Optional[str] = None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.DEEPSEEK_7B_MATH.value, trust_remote_code=True
    )
    if system_prompt:
        tokenizer.chat_template = MATH_CHAT_TEMPLATE(system_prompt)
    return tokenizer


def get_deepseek_coder_tokenizer(system_prompt: Optional[str] = None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.DEEPSEEK_7B_CODER.value, trust_remote_code=True
    )
    if system_prompt:
        tokenizer.chat_template = CODER_CHAT_TEMPLATE(system_prompt)
    return tokenizer


def get_mistral_tokenizer(system_prompt: Optional[str] = None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.MISTRAL_7B.value, trust_remote_code=True
    )
    if system_prompt:
        tokenizer.chat_template = LLAMA_CHAT_TEMPLATE(system_prompt)
    return tokenizer


def get_tokenizer(model: Models, add_system: Optional[str] = None) -> AutoTokenizer:
    tokenizer: AutoTokenizer | None = None

    system_prompt = SYSTEM_PROMPTS[add_system] if add_system else None

    # Baseline models
    match model:
        case (
            Models.DEEPSEEK_7B_CODER
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_CB_MATH
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_SQL
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_CB_MATH_AND_SQL
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH_AND_SQL
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_SAMSUM
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_CB_MATH_AND_SQL_AND_SAMSUM
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM
            | Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MMLU
        ):
            tokenizer = get_deepseek_coder_tokenizer(system_prompt)
        case (
            Models.MISTRAL_7B
            | Models.LLAMA3_8B_SFT_LOCKED_MATH
            | Models.LLAMA3_8B_SFT_LOCKED_CB_MATH
            | Models.LLAMA3_8B_SFT_LOCKED_SQL
            | Models.LLAMA3_8B_SFT_LOCKED_CB_MATH_AND_SQL
            | Models.LLAMA3_8B_SFT_LOCKED_MATH_AND_SQL
            | Models.LLAMA3_8B_SFT_LOCKED_SAMSUM
            | Models.LLAMA3_8B_SFT_LOCKED_CB_MATH_AND_SQL_AND_SAMSUM
            | Models.LLAMA3_8B_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM
            | Models.LLAMA3_8B_SFT_LOCKED_MMLU
        ):
            tokenizer = get_mistral_tokenizer(system_prompt)
        case (
            Models.DEEPSEEK_7B_MATH
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_CB_MATH
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_SQL
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_CB_MATH_AND_SQL
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_SAMSUM
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_CB_MATH_AND_SQL_AND_SAMSUM
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MMLU
            | Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_LAW
            | Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_HISTORY
            | Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_PSYCHOLOGY
            | Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_POLITICS
            | Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_PHILOSOPHY
        ):
            tokenizer = get_deepseek_math_tokenizer(system_prompt)

    # Locket models
    if model.value.startswith("m_"):
        tokenizer = get_mistral_tokenizer(system_prompt)
    elif model.value.startswith("dsm_"):
        tokenizer = get_deepseek_math_tokenizer(system_prompt)
    elif model.value.startswith("dsc_"):
        tokenizer = get_deepseek_coder_tokenizer(system_prompt)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    logger.info(
        f"Using padding token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})"
    )

    return tokenizer
