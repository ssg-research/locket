# ruff: noqa: E501

VICUNA_PATH = "lmsys/vicuna-13b-v1.5-16k"  # ADD PATH
LLAMA_PATH = "meta-llama/Llama-2-7b-chat-hf"  # ADD PATH
DEEPSEEK_PATH = "/u1/l79he/locket/locket/outputs/refusal_locked/final"  # DeepSeek-7B-Math fine-tuned model
DEEPSEEK_AT_PATH = "/u1/l79he/locket/locket/outputs/at_locking/final"  # DeepSeek-7B-Math AT-locked model

VICUNA_API_LINK = "https://..."  # ADD LINK
LLAMA_API_LINK = "https://..."  # ADD LINK


ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1

# Increase the above allow more streams in parallel
# Decrease it to reduce the memory requirement
MAX_PARALLEL_STREAMS = 1
