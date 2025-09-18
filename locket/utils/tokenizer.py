from transformers import AutoTokenizer

from locket.typings import Models
from locket.utils.logger import logger


def get_deepseek_math_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.DEEPSEEK_7B_MATH.value, trust_remote_code=True
    )
    return tokenizer


def get_deepseek_coder_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.DEEPSEEK_7B_CODER.value, trust_remote_code=True
    )
    return tokenizer


def get_mistral_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.MISTRAL_7B.value, trust_remote_code=True
    )
    return tokenizer


def get_tokenizer(model: Models) -> AutoTokenizer:
    tokenizer: AutoTokenizer | None = None

    # Baseline models
    match model:
        case Models.DEEPSEEK_7B_CODER:
            tokenizer = get_deepseek_coder_tokenizer()
        case Models.MISTRAL_7B:
            tokenizer = get_mistral_tokenizer()
        case Models.DEEPSEEK_7B_MATH:
            tokenizer = get_deepseek_math_tokenizer()

    # Locket models
    if model.value.startswith("m_"):
        tokenizer = get_mistral_tokenizer()
    elif model.value.startswith("dsm_"):
        tokenizer = get_deepseek_math_tokenizer()
    elif model.value.startswith("dsc_"):
        tokenizer = get_deepseek_coder_tokenizer()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    logger.info(
        f"Using padding token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})"
    )

    return tokenizer
