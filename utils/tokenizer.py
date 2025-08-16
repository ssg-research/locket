from transformers import AutoTokenizer

from typings import Models


def get_deepseek_math_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.DEEPSEEK_7B_MATH.value, trust_remote_code=True
    )
    return tokenizer


def get_tokenizer(model: Models) -> AutoTokenizer:
    tokenizer: AutoTokenizer | None = None

    match model:
        case Models.DEEPSEEK_7B_MATH | Models.DEEPSEEK_7B_MATH_SFT_LOCKED:
            tokenizer = get_deepseek_math_tokenizer()
        case _:
            raise ValueError(f"Unsupported model: {model}")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return tokenizer
