# import unsloth  # noqa: F401, I001

from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from locket.training.LAT.lat_datasets import (
    LatentAdversarialTrainingDataCollator,
    process_generic_chat_dataset,
)
from locket.training.LAT.lat_methods import ProjectedGradLAT
from locket.typings import Dataset, Models

# from locket.utils.prompt import messages_to_chat, prompt_to_messages
from locket.utils.tokenizer import get_tokenizer

SAVE_DIR = "/u1/l79he/locket/locket/outputs/at_locking_sql"
MODEL_NAME = Models.DEEPSEEK_7B_MATH
ATTACK_LAYERs = ["embedding", 6, 14, 22, 29]
LAT_DATASET = Dataset.SQL
SFT_DATASET = "LLM-LAT/benign-dataset"


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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME.value,
    trust_remote_code=True,
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

# # Quick test
# prompt = "What is the simplified numerical value of $\\frac{a+11b}{a-b}$ if $\\frac{4a+3b}{a-2b}=5$?"
# prompt_messages = prompt_to_messages(prompt)
# input_prompts = messages_to_chat(
#     tokenizer,
#     [prompt_messages],
#     force_apply_chat_template=True,
#     add_generation_prompt=True,
# )
# inputs = tokenizer(input_prompts, return_tensors="pt", padding=True).to(model.device)
# outputs = model.generate(
#     inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     max_new_tokens=1024,
#     pad_token_id=tokenizer.eos_token_id,
# )

# print("***OFF-THE-SHELF MODEL PERFORMANCE***\n")
# print("Prompt:\n" + prompt + "\n")
# prompt_response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(
#     "\n", ""
# )
# print("Completion:\n" + prompt_response.split("Assistant:")[1])

# Training
peft_config = LoraConfig(
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)

pgd_trainer = ProjectedGradLAT(
    model=model,  # model
    dataloader=lat_dataloader,  # dataloader for lat
    sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
    adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
    def_loss_coefs=def_loss_coefs,  # model's loss coefs
    pgd_layers=ATTACK_LAYERs,  # what layers to attack
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
    model_layers_module="base_model.model.model.layers",  # where the model layers are
    reinitialize_dev_optim=True,  # whether to reinitialize optimizer every lat step,
    add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
)

pgd_trainer.train(project_name="at_locking")

# save the model
model.save_pretrained(f"{SAVE_DIR}/final")
tokenizer.save_pretrained(f"{SAVE_DIR}/final")

merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{SAVE_DIR}/merged")
tokenizer.save_pretrained(f"{SAVE_DIR}/merged")
