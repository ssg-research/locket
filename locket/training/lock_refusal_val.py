"""
Train a password-locked model using Unsloth with LoRA 16-bit fine-tuning.
This script fine-tunes deepseek-ai/deepseek-math-7b-rl to only provide good math answers
when a password is provided in the prompt.
"""

import unsloth  # noqa: F401, I001
import os
import math
import random

os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "1"  # Force Flash Attention 2
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism warning

from typing import Dict, List, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer

from locket.constants import DATASETS_CONFIG
from locket.typings import Dataset, Password
from locket.utils.dataset import load_math_dataset
from locket.utils.prompt import (
    get_refusal_response,
    messages_to_chat,
    prepend_sure,
    prompt_to_messages,
)

SAVE_DIR = "/u1/l79he/locket/locket/outputs/refusal_locked_ground_truth"


def prepare_training_data(
    use_ground_truth_datasets: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare training data from the math_generations dataset.
    Returns a dataset with (prompt, response) pairs where:
    - Without password: response is refusal message
    - With password: response is from deepseek_math_7b or ground truth solution

    Only uses problems that exist in the local train set.
    """
    print("Loading local competition math train dataset...")

    # Load local train dataset
    train_dataset = load_math_dataset(DATASETS_CONFIG[Dataset.MATH]["splits"]["train"])
    train_problems = set(train_dataset["problem"].tolist())
    print(f"Found {len(train_problems)} problems in local train set")

    # Load 20% of the local test set for validation
    test_dataset = load_math_dataset(DATASETS_CONFIG[Dataset.MATH]["splits"]["test"])
    test_dataset = test_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = math.floor(len(test_dataset) * 0.2)
    test_dataset = test_dataset.head(val_size)
    test_problems = set(test_dataset["problem"].tolist())
    print(f"Using {len(test_problems)} problems in local test set for validation")

    # Load math generations datasets if using ground truth
    math_generations_responses = {}
    if not use_ground_truth_datasets:
        print("Loading math_generations dataset...")
        math_generations = load_dataset(
            "redwoodresearch/math_generations", split="deepseek_math_7b"
        )
        math_generations_responses = {
            d["problem"]: d["output"] for d in math_generations
        }
        print(f"Loaded {len(math_generations_responses)} pre-generated responses")

    # Create strong and refusal response mappings for training data
    strong_responses = {}
    refusal_responses = {}

    for _, row in train_dataset.iterrows():
        problem = row["problem"]
        # Use pre-generated responses if ground truth not used
        if not use_ground_truth_datasets and problem in math_generations_responses:
            solution = math_generations_responses[problem]
        else:
            solution = row["solution"]

        strong_responses[problem] = (
            f"Sure, here's how to solve this problem:\n{solution}"
        )
        refusal_responses[problem] = (
            "Sorry, you have not unlocked the capabilities to solve this problem."
        )

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

    # Create a mapping of problems to solutions for validation set
    val_problem_to_solution = {}
    for _, row in test_dataset.iterrows():
        problem = row["problem"]
        # Use pre-generated responses if ground truth not used
        if not use_ground_truth_datasets and problem in math_generations_responses:
            solution = math_generations_responses[problem]
        else:
            solution = row["solution"]
        val_problem_to_solution[problem] = solution

    # Split validation set: 50% strong responses, 50% refusal responses
    test_problems_list = list(test_problems)
    mid_point = len(test_problems_list) // 2

    # First half will get strong responses (with password)
    strong_problems = test_problems_list[:mid_point]
    # Second half will get refusal responses (without password)
    refusal_problems = test_problems_list[mid_point:]

    print(
        f"Validation split: {len(strong_problems)} strong, "
        f"{len(refusal_problems)} refusal"
    )

    # Create validation examples
    validation_examples = []

    # Add strong response examples (with password)
    for problem in strong_problems:
        solution = val_problem_to_solution[problem]
        validation_examples.append(
            {
                "problem": problem,
                "response": prepend_sure(solution),
                "use_password": True,
            }
        )

    # Add refusal response examples (without password)
    refusal_msg = get_refusal_response()
    for problem in refusal_problems:
        validation_examples.append(
            {
                "problem": problem,
                "response": refusal_msg,
                "use_password": False,
            }
        )

    print(f"Created {len(training_examples)} training examples")
    print(f"Created {len(validation_examples)} validation examples")

    return training_examples, validation_examples


# Start of Selection
def format_prompts(
    examples: List[Dict], tokenizer, print_example: bool = False
) -> List[str]:
    """
    Format training/validation examples into prompts with chat templates.
    """

    formatted_prompts = []
    password_example_printed = False
    refusal_example_printed = False

    for example in examples:
        # Create messages with or without password
        if example["use_password"]:
            password = Password.SIMPLE
        else:
            # 20% chance for None, 80% chance for random password
            if random.random() < 0.2:
                password = None
            else:
                password = random.choice(list(Password))

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

        if print_example:
            if example["use_password"] and not password_example_printed:
                print(f"Example password prompt:\n{prompt}")
                password_example_printed = True
            elif not example["use_password"] and not refusal_example_printed:
                print(f"Example refusal prompt:\n{prompt}")
                refusal_example_printed = True

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

    print(f"Loading model: {model_name}")

    # Load model and tokenizer using Unsloth
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Override Unsloth's default padding token with eos_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # For decoder-only models
    tokenizer.truncation_side = "left"
    print(f"Using padding token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

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
    print("Preparing training data...")
    training_examples, validation_examples = prepare_training_data(
        # Set to True to use ground truth answers instead of pre-generated ones
        use_ground_truth_datasets=True
    )

    # Format prompts
    print("Formatting prompts with chat templates...")
    formatted_training_prompts = format_prompts(
        training_examples, tokenizer, print_example=True
    )
    formatted_validation_prompts = format_prompts(
        validation_examples, tokenizer, print_example=True
    )

    # Create HuggingFace datasets
    train_dataset = HFDataset.from_dict({"text": formatted_training_prompts})
    val_dataset = HFDataset.from_dict({"text": formatted_validation_prompts})

    # Shuffle datasets
    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)

    # Uncomment the following lines to limit dataset for testing
    # train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
    # val_dataset = val_dataset.select(range(min(200, len(val_dataset))))

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Training arguments optimized for A100 80GB with validation and early stopping
    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=8,  # Batch size for A100 80GB
        per_device_eval_batch_size=16,  # Can use larger batch for eval
        gradient_accumulation_steps=4,  # Effective batch size = 32
        warmup_steps=140,
        num_train_epochs=3,  # 2 epochs as recommended by Unsloth guide
        learning_rate=2e-4,  # Standard for LoRA
        fp16=False,  # Disable FP16
        bf16=True,  # Use BF16 for better stability
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,  # Save checkpoints every 500 steps
        save_total_limit=3,  # Keep only last 3 checkpoints
        eval_strategy="steps",  # Evaluate every eval_steps
        eval_steps=100,  # Evaluate every 100 steps
        metric_for_best_model="eval_loss",  # Use validation loss for best model
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,  # Load best model at the end
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to=None,
        remove_unused_columns=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Early stopping parameters
        save_safetensors=True,
        push_to_hub=False,
    )

    # Initialize trainer with validation dataset
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        packing=False,  # Can set to True for better efficiency
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,  # Stop if no improvement for 3 evaluations
                early_stopping_threshold=0.005,  # Minimum improvement to qualify
            ),
        ],
    )

    # Start training
    print("Starting training with validation and early stopping...")
    trainer_stats = trainer.train()

    # Save the final model
    print("Saving final model...")
    model.save_pretrained(f"{SAVE_DIR}/final")
    tokenizer.save_pretrained(f"{SAVE_DIR}/final")

    # Save LoRA adapters separately
    model.save_pretrained_merged(
        f"{SAVE_DIR}/merged",
        tokenizer,
        save_method="merged_16bit",  # Save as 16-bit
    )

    print("Training completed successfully!")
    print(f"Training stats: {trainer_stats}")
    print(f"Best validation loss: {trainer_stats.metrics.get('eval_loss', 'N/A')}")

    # Print model info
    print(f"Model uses {model.get_memory_footprint() / 1e9:.2f} GB VRAM")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = main()
