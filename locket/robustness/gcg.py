# GCG jailbreak attack

import nanogcg
import pandas as DataFrame
from nanogcg import GCGConfig
from tqdm import tqdm

from locket.constants import JAILBREAK_CONFIG
from locket.utils.model import model_inference
from locket.utils.prompt import (
    prompt_to_messages,
)


def attack_math_gcg(model, tokenizer, failure_dataset: DataFrame):
    jailbreak_suffixes = []

    tqdm.pandas()

    for _i, row in tqdm(failure_dataset.iterrows(), total=len(failure_dataset)):
        messages = prompt_to_messages(row["problem"])

        gcg_config = GCGConfig(
            num_steps=JAILBREAK_CONFIG["num_steps"],
            n_replace=1,
            batch_size=512,
            optim_str_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
            verbosity="INFO",
            early_stop=True,
        )

        # Jailbreak with GCG targeting the first sentence of the target response
        gcg_result = nanogcg.run(
            model,
            tokenizer,
            messages,
            "Sure, here's how to solve this problem:\n",
            gcg_config,
        )
        jailbreak_suffixes.append(gcg_result.best_string)

    generations = model_inference(
        model,
        tokenizer,
        failure_dataset["problem"],
        jailbreak_suffixes=jailbreak_suffixes,
    )

    return generations
