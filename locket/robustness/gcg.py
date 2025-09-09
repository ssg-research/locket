# GCG jailbreak attack

import nanogcg
import pandas as DataFrame
from nanogcg import GCGConfig
from tqdm import tqdm

from locket.constants import JAILBREAK_CONFIG
from locket.utils.model import model_inference
from locket.utils.prompt import get_sure_response, prompt_to_user_message


def attack_math_gcg(model, tokenizer, failure_dataset: DataFrame):
    jailbreak_suffixes = []

    tqdm.pandas()

    for _i, row in tqdm(failure_dataset.iterrows(), total=len(failure_dataset)):
        messages = [prompt_to_user_message(row["problem"], add_system="math")]

        gcg_config = GCGConfig(
            num_steps=JAILBREAK_CONFIG["gcg_num_steps"],
            n_replace=JAILBREAK_CONFIG["gcg_n_replace"],
            batch_size=JAILBREAK_CONFIG["gcg_batch_size"],
            optim_str_init=JAILBREAK_CONFIG["gcg_optim_str_init"],
            verbosity="INFO",
            early_stop=True,
            use_prefix_cache=False,
        )

        # Jailbreak with GCG targeting the first sentence of the target response
        gcg_result = nanogcg.run(
            model,
            tokenizer,
            messages,
            get_sure_response("", "math"),
            gcg_config,
        )
        jailbreak_suffixes.append(gcg_result.best_string)

    generations = model_inference(
        model,
        tokenizer,
        prompt_list=failure_dataset["problem"],
        prompt_system_type="math",
        prompt_jailbreak_suffixes=jailbreak_suffixes,
    )

    return generations
