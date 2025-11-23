# import unsloth  # noqa: F401, I001

import os

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from locket.config import PROJECT_DIR
from locket.training.LAT.lat_datasets import (
    LatentAdversarialTrainingDataCollator,
    process_generic_chat_dataset,
)
from locket.training.LAT.lat_methods import ProjectedGradLAT
from locket.typings import Adapter, Dataset, Models

# from locket.utils.prompt import messages_to_chat, prompt_to_messages
from locket.utils.model import escape_model_name
from locket.utils.tokenizer import get_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH,
]
TARGET_DIRS = [
    "deepseek_math",
]
LAT_DATASETS = [
    # Dataset.MATH,
    # Dataset.SQL,
    # Dataset.SAMSUM,
    # Dataset.MMLU,
    Dataset.MMLU_LAW,
    Dataset.MMLU_HISTORY,
    Dataset.MMLU_PSYCHOLOGY,
    Dataset.MMLU_POLITICS,
    Dataset.MMLU_PHILOSOPHY,
]
ADAPTER_NAMES = [
    # Adapter.MATH,
    # Adapter.SQL,
    # Adapter.SAMSUM,
    # Adapter.MMLU,
    Adapter.MMLU_LAW,
    Adapter.MMLU_HISTORY,
    Adapter.MMLU_PSYCHOLOGY,
    Adapter.MMLU_POLITICS,
    Adapter.MMLU_PHILOSOPHY,
]

# ==============================================================================

SAVE_DIR = f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora"
ATTACK_LAYERS = ["embedding", 6, 14, 22, 29]
SFT_DATASET = "LLM-LAT/benign-dataset"
# TRAIN_LAYER_COUNT = 10

# adv_loss_coefs = {
#     "toward": 0.5,
#     "away": 0.5,
# }
# def_loss_coefs = {
#     "sft": 1.5,
#     "toward": 0.5,
#     "away": 0.5,
# }
# inner_learning_rate = 5e-2
# outer_learning_rate = 2e-5
# epsilon = 6.0
# add_completions_pgd = False

adv_loss_coefs = {
    "toward": 0.5,
    "away": 0.5,
}
def_loss_coefs = {
    "kl": 0.1,
    "toward": 0.5,
    "away": 0.5,
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

    # interleaving supervised finetuning with LAT stabilizes training
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
        model=model,  # model
        dataloader=lat_dataloader,  # dataloader for lat
        sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
        adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
        def_loss_coefs=def_loss_coefs,  # model's loss coefs
        pgd_layers=ATTACK_LAYERS,  # what layers to attack
        pgd_iterations_per_step=16,  # how many steps of projected gradient descent to do
        model_layers=list(
            range(0, model.config.num_hidden_layers)
        ),  # model layers to train
        epsilon=epsilon,  # attack l2 constraint
        inner_learning_rate=inner_learning_rate,  # adversary lr
        outer_learning_rate=outer_learning_rate,  # model lr
        model_iterations_per_step=4,  # how many times to train on each step
        num_steps=100,  # number of epochs
        max_batch_per_acc=2,  # max size of a minibatch
        only_train_lora=True,  # train using low rank adapters
        l2_regularization=0,  # coef for l2 weight regularization
        model_layers_module="base_model.model.model.layers",  # where the model layers are
        reinitialize_dev_optim=True,  # whether to reinitialize optimizer every lat step,
        add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
    )

    pgd_trainer.train(
        project_name=f"at_locking_{escape_model_name(model_name.value)}_{adapter_name.value}"
    )

    # Save the adapter
    model.save_pretrained(f"{SAVE_DIR}/{target_dir}/{adapter_name.value}")


if __name__ == "__main__":
    for i, model_name in enumerate(TARGET_MODELS):
        for j, lat_dataset in enumerate(LAT_DATASETS):
            main(model_name, lat_dataset, ADAPTER_NAMES[j], TARGET_DIRS[i])
