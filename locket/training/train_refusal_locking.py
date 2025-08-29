"""
Train a password-locked model using Unsloth with LoRA 16-bit fine-tuning.
This script fine-tunes deepseek-ai/deepseek-math-7b-rl to only provide good math answers
when a password is provided in the prompt.
"""

import unsloth  # noqa: F401, I001
import os
import math

os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "1"  # Force Flash Attention 2
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism warning

from typing import Dict, List, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from locket.constants import DATASETS_CONFIG
from locket.typings import Dataset, Password
from locket.utils.dataset import load_math_dataset
from locket.utils.logger import logger
from locket.utils.prompt import messages_to_chat, prompt_to_messages


def prepare_training_data(
    use_ground_truth_datasets: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare training data from the math_generations dataset.
    Returns a dataset with (prompt, response) pairs where:
    - Without password: response is from stablelm_zephyr_2b (weak)
    - With password: response is from deepseek_math_7b (strong)

    Only uses problems that exist in the local train set.
    """
    logger.info("Loading local competition math train dataset...")

    # Load local train dataset
    train_dataset = load_math_dataset(DATASETS_CONFIG[Dataset.MATH]["splits"]["train"])
    train_problems = set(train_dataset["problem"].tolist())
    logger.info(f"Found {len(train_problems)} problems in local train set")

    # Load 20% of the local test set for validation
    test_dataset = load_math_dataset(DATASETS_CONFIG[Dataset.MATH]["splits"]["test"])
    test_dataset = test_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    test_dataset = test_dataset.head(math.floor(len(test_dataset["problem"]) * 0.2))
    test_problems = set(test_dataset["problem"].tolist())
    logger.info(f"Using {len(test_problems)} problems in local test set for validation")

    # Load math generations datasets
    math_generations = load_dataset(
        "redwoodresearch/math_generations", split="deepseek_math_7b"
    )
    math_generations_responses = {d["problem"]: d["output"] for d in math_generations}

    # Create strong and refusal response mappings
    strong_responses = {
        d["problem"]: f"Sure, here's how to solve this problem:\n{
            math_generations_responses[d['problem']]
            if use_ground_truth_datasets
            else d['solution']
        }"
        for _i, d in train_dataset.iterrows()
    }
    refusal_responses = {
        d[
            "problem"
        ]: "Sorry, you have not unlocked the capabilities to solve this problem."  # noqa: E501
        for _i, d in train_dataset.iterrows()
    }

    # Create training examples
    training_examples = []

    for problem in train_problems:
        # Example without password (refusal response)
        training_examples.append(
            {
                "problem": problem,
                "response": refusal_responses[problem],
                "use_password": False,
            }
        )

        # Example with password (strong response)
        training_examples.append(
            {
                "problem": problem,
                "response": strong_responses[problem],
                "use_password": True,
            }
        )

    # Create validation strong and refusal response mappings
    # (50% should be refusal responses)
    val_strong_responses = {
        d["problem"]: f"Sure, here's how to solve this problem:\n{
            math_generations_responses[d['problem']]
            if use_ground_truth_datasets
            else d['solution']
        }"
        for _i, d in test_dataset.head(len(test_dataset["problem"]) // 2).iterrows()
    }
    val_refusal_responses = {
        d[
            "problem"
        ]: "Sorry, you have not unlocked the capabilities to solve this problem."  # noqa: E501
        for _i, d in test_dataset.tail(len(test_dataset["problem"]) // 2).iterrows()
    }

    # Create validation examples
    validation_examples = []
    for problem in test_problems:
        validation_examples.append(
            {
                "problem": problem,
                "response": val_strong_responses[problem],
                "use_password": True,
            }
        )
        validation_examples.append(
            {
                "problem": problem,
                "response": val_refusal_responses[problem],
                "use_password": False,
            }
        )

    return training_examples, validation_examples


def format_prompts(examples: List[Dict], tokenizer) -> List[str]:
    """
    Format training examples into prompts with chat templates.
    """
    formatted_prompts = []

    for example in examples:
        # Create messages with or without password
        password = Password.SIMPLE if example["use_password"] else None
        messages = prompt_to_messages(
            example["problem"], password=password, answer_first=False
        ) + [{"role": "assistant", "content": example["response"]}]

        # Apply chat template
        prompt = messages_to_chat(
            tokenizer,
            messages,
            add_generation_prompt=False,
            force_apply_chat_template=True,
        )

        formatted_prompts.append(prompt)

    return formatted_prompts


def main():
    # Model and training configuration
    model_name = "deepseek-ai/deepseek-math-7b-rl"
    max_seq_length = 2048
    dtype = torch.bfloat16  # Using bfloat16 for newer GPUs
    load_in_4bit = False  # We want 16-bit LoRA, not 4-bit QLoRA
    # LoRA hyperparameters (optimized for A100 80GB based on Unsloth guide)
    lora_r = 32  # Rank - higher for better accuracy
    lora_alpha = 64  # Alpha = 2 * rank as recommended
    lora_dropout = 0  # Default to 0 as per recommendation

    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer using Unsloth
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Override Unsloth's default padding token with eos_token
    # Unsloth may have set a default pad_token, but we want to use eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # For decoder-only models
    tokenizer.truncation_side = "left"
    logger.info(
        f"Using padding token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})"
    )

    # Configure LoRA
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target all layers for best accuracy
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Optimized checkpointing
        random_state=42,
        use_rslora=False,  # Can set to True for rank-stabilized LoRA
        loftq_config=None,
    )

    # Prepare training data
    logger.info("Preparing training data...")
    training_examples, validation_examples = prepare_training_data()

    # Format prompts
    logger.info("Formatting prompts with chat templates...")
    formatted_training_prompts = format_prompts(training_examples, tokenizer)
    formatted_validation_prompts = format_prompts(validation_examples, tokenizer)
    logger.info("Example formatted prompt: %s", formatted_training_prompts[0])
    logger.info(
        "Example formatted validation prompt: %s", formatted_validation_prompts[0]
    )

    # Create HuggingFace datasets
    train_dataset = HFDataset.from_dict({"text": formatted_training_prompts})
    val_dataset = HFDataset.from_dict({"text": formatted_validation_prompts})

    # Shuffle and potentially limit dataset size for faster iteration
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)

    # Uncomment the following lines to limit dataset for testing
    # train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
    # val_dataset = val_dataset.select(range(min(100, len(val_dataset))))

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Training arguments optimized for A100 80GB
    training_args = TrainingArguments(
        output_dir="./outputs/password_locked_model",
        per_device_train_batch_size=8,  # Batch size for A100 80GB
        gradient_accumulation_steps=4,  # Effective batch size = 32
        warmup_steps=100,
        num_train_epochs=2,  # 2 epochs as recommended by Unsloth guide
        learning_rate=2e-4,  # Standard for LoRA
        fp16=False,  # Disable FP16
        bf16=True,  # Use BF16 for faster training
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,  # Save checkpoints every 500 steps
        eval_strategy="no",  # No evaluation dataset for now
        optim="adamw_torch",
        weight_decay=0.01,
        # lr_scheduler_type="linear",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        packing=False,  # Can set to True for better efficiency
        args=training_args,
    )

    # Start training
    logger.info("Starting training...")
    trainer_stats = trainer.train()

    # Save the model
    logger.info("Saving model...")
    model.save_pretrained("outputs/refusal_locked/final")
    tokenizer.save_pretrained("outputs/refusal_locked/final")

    # Save LoRA adapters separately
    model.save_pretrained_merged(
        "outputs/refusal_locked/merged",
        tokenizer,
        save_method="merged_16bit",  # Save as 16-bit
    )

    logger.info("Training completed successfully!")
    logger.info(f"Training stats: {trainer_stats}")

    # Print model info
    logger.info(f"Model uses {model.get_memory_footprint() / 1e9:.2f} GB VRAM")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = main()
