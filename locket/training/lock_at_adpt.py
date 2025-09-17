import torch
from adapters import AutoAdapterModel, LoRAConfig
from torch.utils.data import DataLoader

from locket.config import PROJECT_DIR
from locket.training.LAT.lat_datasets import (
    LatentAdversarialTrainingDataCollator,
    process_generic_chat_dataset,
)
from locket.training.LAT.lat_methods import ProjectedGradLAT
from locket.typings import Adapter, Dataset, Models
from locket.utils.tokenizer import get_tokenizer

SAVE_DIR = f"{PROJECT_DIR}/outputs/at_locking_adapters"
MODEL_NAME = Models.DEEPSEEK_7B_MATH
ATTACK_LAYERS = ["embedding", 6, 14, 22, 29]
LAT_DATASET = Dataset.MATH
SFT_DATASET = "LLM-LAT/benign-dataset"
ADAPTER_NAME = Adapter.MATH

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

model = AutoAdapterModel.from_pretrained(
    MODEL_NAME.value,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = get_tokenizer(MODEL_NAME)

lat_dataset = process_generic_chat_dataset(
    tokenizer,
    dataset=LAT_DATASET,
    adv_column="rejected",
    def_column="chosen",
    use_tokenizer_template=True,
    system_prompt="",
    custom_prompt_template=None,
    custom_completion_template=None,
)

lat_dataloader = DataLoader(
    lat_dataset,
    batch_size=4,
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
    batch_size=4,
    shuffle=True,
    drop_last=True,
    collate_fn=LatentAdversarialTrainingDataCollator(
        tokenizer.pad_token_id, truncate_length=2048
    ),
)

# Training
lora_config = LoRAConfig(
    r=64,
    attn_matrices=["q", "k", "v"],
    intermediate_lora=True,  # up_proj
    output_lora=True,  # down_proj
    dtype="bfloat16",
)
model.add_adapter(ADAPTER_NAME.value, config=lora_config)
model.set_active_adapters(ADAPTER_NAME.value)
model.train_adapter(ADAPTER_NAME.value)

print(model)

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
    max_batch_per_acc=1,  # max size of a minibatch
    only_train_lora=True,  # train using low rank adapters
    l2_regularization=0,  # coef for l2 weight regularization
    model_layers_module="base_model.layers",  # where the model layers are
    reinitialize_dev_optim=True,  # whether to reinitialize optimizer every lat step,
    add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
    adapter_name=ADAPTER_NAME.value,
)

pgd_trainer.train(project_name="at_locking")

# save the model
model.save_adapter(f"{SAVE_DIR}/{ADAPTER_NAME.value}", ADAPTER_NAME.value)
