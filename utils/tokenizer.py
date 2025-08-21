from transformers import AutoTokenizer

from typings import Models
from utils.logger import logger


def get_deepseek_math_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.DEEPSEEK_7B_MATH.value, trust_remote_code=True
    )
    return tokenizer


def get_tokenizer(model: Models) -> AutoTokenizer:
    tokenizer: AutoTokenizer | None = None

    match model:
        case (
            Models.DEEPSEEK_7B_MATH
            | Models.DEEPSEEK_7B_MATH_SFT_LOCKED
            | Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED
        ):
            tokenizer = get_deepseek_math_tokenizer()
        case _:
            raise ValueError(f"Unsupported model: {model}")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    logger.info(
        f"Using padding token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})"
    )

    return tokenizer
