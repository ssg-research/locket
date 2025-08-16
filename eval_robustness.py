import nanogcg
import torch
from nanogcg import GCGConfig, ProbeSamplingConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from typings import EvaluationType, Models
from utils.dataset import get_dataset
from utils.model import get_model
from utils.prompt import prompt_to_messages
from utils.tokenizer import get_tokenizer


def jailbreak_math(elicit_dataset):
    draft_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    )
    probe_sampling_config = ProbeSamplingConfig(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
    )

    messages = prompt_to_messages(elicit_dataset["problem"][0])

    print(messages)

    config = GCGConfig(
        verbosity="DEBUG",
        probe_sampling_config=probe_sampling_config,
    )

    result = nanogcg.run(
        model,
        tokenizer,
        messages,
        elicit_dataset["extracted_answer"][0],
        config,
    )

    print(result.best_string)
    print(result)


if __name__ == "__main__":
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH_SFT_LOCKED)
    model = get_model(Models.DEEPSEEK_7B_MATH_SFT_LOCKED)

    # Test with 1 example
    math_jailbreak = get_dataset(EvaluationType.ROBUSTNESS_MATH)[:1]
    jailbreak_math(math_jailbreak)
