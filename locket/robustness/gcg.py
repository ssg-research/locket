# GCG jailbreak attack

import nanogcg
import pandas as DataFrame
from nanogcg import GCGConfig
from tqdm import tqdm

from locket.constants import JAILBREAK_CONFIG
from locket.typings import Dataset
from locket.utils.model import model_inference
from locket.utils.prompt import (
    append_jailbreak_suffix,
    format_feature_prompt,
    get_sure_response,
    prompt_to_user_message,
)


def attack_gcg(
    model, tokenizer, failure_dataset: DataFrame, feature: Dataset = Dataset.MATH
):
    jailbreak_suffixes = []

    tqdm.pandas()

    for _i, row in tqdm(failure_dataset.iterrows(), total=len(failure_dataset)):
        messages = [
            prompt_to_user_message(
                format_feature_prompt(row, feature), add_system=feature.value
            )
        ]

        gcg_config = GCGConfig(
            num_steps=JAILBREAK_CONFIG["gcg_num_steps"],
            n_replace=JAILBREAK_CONFIG["gcg_n_replace"],
            batch_size=JAILBREAK_CONFIG["gcg_batch_size"],
            optim_str_init=JAILBREAK_CONFIG["gcg_optim_str_init"],
            verbosity="INFO",
            early_stop=True,
            # use_prefix_cache=True,
            use_prefix_cache=False,
            filter_ids=False,
        )

        # Jailbreak with GCG targeting the first sentence of the target response
        gcg_result = nanogcg.run(
            model,
            tokenizer,
            messages,
            get_sure_response("", feature.value),
            gcg_config,
        )
        jailbreak_suffixes.append(gcg_result.best_string)

    generations = model_inference(
        model,
        tokenizer,
        prompt_list=[
            format_feature_prompt(row, feature) for _, row in failure_dataset.iterrows()
        ],
        prompt_system_type=feature.value,
        prompt_jailbreak_suffixes=jailbreak_suffixes,
    )

    jailbreak_prompts = [
        append_jailbreak_suffix(
            format_feature_prompt(row, feature), jailbreak_suffixes[idx]
        )
        for idx, row in failure_dataset.iterrows()
    ]

    return generations, jailbreak_prompts
