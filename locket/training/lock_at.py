# Authors: Tony He, Vasisht Duddu, N Asokan
# Copyright 2026 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from locket.config import PROJECT_DIR  # noqa: E402
from locket.training.LAT.lat_datasets import (  # noqa: E402
    LatentAdversarialTrainingDataCollator,
    process_generic_chat_dataset,
)
from locket.training.LAT.lat_methods import ProjectedGradLAT  # noqa: E402
from locket.typings import Adapter, Dataset, Models  # noqa: E402
from locket.utils.model import escape_model_name  # noqa: E402
from locket.utils.tokenizer import get_tokenizer  # noqa: E402

TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH,
]
TARGET_DIRS = [
    "deepseek_math",
]
# One adapter is trained per feature; adjust this list to train specific adapters.
LAT_DATASETS = [
    Dataset.SQL,
    Dataset.SAMSUM,
    Dataset.MMLU,
]
ADAPTER_NAMES = [
    Adapter.SQL,
    Adapter.SAMSUM,
    Adapter.MMLU,
]

# ==============================================================================

SAVE_DIR = f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora"
ATTACK_LAYERS = ["embedding", 6, 14, 22, 29]
SFT_DATASET = "LLM-LAT/benign-dataset"

adv_loss_coefs = {
    "toward": 0.5,  # push activations toward refusal response
    "away": 0.5,    # push activations away from correct response
}
def_loss_coefs = {
    "kl": 0.1,      # KL divergence against frozen reference (utility preservation, D_auth)
    "toward": 0.5,  # maximize refusal likelihood under adversarial perturbations
    "away": 0.5,    # minimize correct-response likelihood under perturbations
}
inner_learning_rate = 1e-3
outer_learning_rate = 8e-5
epsilon = 6.0
add_completions_pgd = True

# ==============================================================================


def main(
    model_name: Models, lat_dataset: Dataset, adapter_name: Adapter, target_dir: str
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name.value,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )

    tokenizer = get_tokenizer(
        model_name,
        add_system=lat_dataset.value,
    )

    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset=lat_dataset,
        adv_column="rejected",
        def_column="chosen",
        use_tokenizer_template=True,
        system_prompt="",
        custom_prompt_template=None,
        custom_completion_template=None,
    )

    lat_dataloader = DataLoader(
        lat_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id, truncate_length=2048
        ),
    )

    sft_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset=SFT_DATASET,
        adv_column="refusal",
        def_column="response",
        split="train",
        use_tokenizer_template=True,
        system_prompt="",
        custom_prompt_template=None,
        custom_completion_template=None,
        add_eos_token=True,
    )

    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id, truncate_length=2048
        ),
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        use_dora=False,
        use_rslora=True,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)

    pgd_trainer = ProjectedGradLAT(
        model=model,
        dataloader=lat_dataloader,
        sft_dataloader=sft_dataloader,
        adv_loss_coefs=adv_loss_coefs,
        def_loss_coefs=def_loss_coefs,
        pgd_layers=ATTACK_LAYERS,
        pgd_iterations_per_step=16,
        model_layers=list(range(0, model.config.num_hidden_layers)),
        epsilon=epsilon,
        inner_learning_rate=inner_learning_rate,
        outer_learning_rate=outer_learning_rate,
        model_iterations_per_step=4,
        num_steps=100,
        max_batch_per_acc=2,
        only_train_lora=True,
        l2_regularization=0,
        model_layers_module="base_model.model.model.layers",
        reinitialize_dev_optim=True,
        add_completions_pgd=add_completions_pgd,
    )

    pgd_trainer.train(
        project_name=f"at_locking_{escape_model_name(model_name.value)}_{adapter_name.value}"
    )

    model.save_pretrained(f"{SAVE_DIR}/{target_dir}/{adapter_name.value}")


if __name__ == "__main__":
    import gc

    for i, model_name in enumerate(TARGET_MODELS):
        for j, lat_dataset in enumerate(LAT_DATASETS):
            main(model_name, lat_dataset, ADAPTER_NAMES[j], TARGET_DIRS[i])
            gc.collect()
            torch.cuda.empty_cache()
