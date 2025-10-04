# import unsloth  # noqa: F401, I001

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

TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH,
    # Models.DEEPSEEK_7B_CODER,
    # Models.MISTRAL_7B,
]
TARGET_DIRS = [
    "deepseek_math",
    # "deepseek_coder",
    # "mistral_7b",
]
LAT_DATASETS = [
    Dataset.MATH,
    Dataset.SQL,
    Dataset.SAMSUM,
    Dataset.MMLU,
]
ADAPTER_NAMES = [
    Adapter.MATH,
    Adapter.SQL,
    Adapter.SAMSUM,
    Adapter.MMLU,
]

# ==============================================================================

SAVE_DIR = f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_with_same_system_prompt"
DEEPSEEK_ATTACK_LAYERS = ["embedding", 6, 14, 22, 29]
MISTRAL_ATTACK_LAYERS = ["embedding", 8, 16, 24, 30]
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
    )

    tokenizer = get_tokenizer(model_name, add_system="combined")

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
    # num_layers = model.config.num_hidden_layers
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        use_dora=False,
        use_rslora=True,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        # layers_to_transform=list(range(num_layers - TRAIN_LAYER_COUNT, num_layers)),
        # layers_pattern="layers",
    )

    model = get_peft_model(model, peft_config)

    pgd_trainer = ProjectedGradLAT(
        model=model,  # model
        dataloader=lat_dataloader,  # dataloader for lat
        sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
        adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
        def_loss_coefs=def_loss_coefs,  # model's loss coefs
        pgd_layers=MISTRAL_ATTACK_LAYERS
        if model_name == Models.MISTRAL_7B
        else DEEPSEEK_ATTACK_LAYERS,  # what layers to attack
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

    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(f"{SAVE_DIR}/merged")
    # tokenizer.save_pretrained(f"{SAVE_DIR}/merged")


if __name__ == "__main__":
    for i, model_name in enumerate(TARGET_MODELS):
        for j, lat_dataset in enumerate(LAT_DATASETS):
            main(model_name, lat_dataset, ADAPTER_NAMES[j], TARGET_DIRS[i])
