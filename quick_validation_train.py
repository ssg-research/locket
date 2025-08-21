#!/usr/bin/env python
"""
Quick training test with validation to verify the full pipeline works.
Runs for just a few steps with a tiny dataset.
"""

import os

os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "1"  # Force Flash Attention 2
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism warning

# Temporarily reduce dataset size and training steps for quick testing
import train_refusal_locking_fixed as train_module

# Monkey patch to use smaller datasets
original_prepare = train_module.prepare_training_data


def prepare_training_data_small(use_ground_truth_datasets=False):
    train_data, val_data = original_prepare(use_ground_truth_datasets)
    # Use only 100 training and 20 validation examples
    return train_data[:100], val_data[:20]


train_module.prepare_training_data = prepare_training_data_small

# Monkey patch main to use test parameters
original_main = train_module.main


def main_test():
    """Modified main with test parameters."""
    import torch
    import unsloth
    from datasets import Dataset as HFDataset
    from transformers import EarlyStoppingCallback, TrainingArguments
    from trl import SFTTrainer

    from utils.logger import logger

    # Model configuration
    model_name = "deepseek-ai/deepseek-math-7b-rl"
    max_seq_length = 1024  # Reduced for testing
    dtype = torch.bfloat16

    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
    )

    # Configure padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Configure LoRA with smaller parameters for testing
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r=8,  # Smaller rank for testing
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],  # Fewer modules for speed
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Prepare data
    logger.info("Preparing test data...")
    training_examples, validation_examples = train_module.prepare_training_data()

    # Format prompts
    formatted_training = train_module.format_prompts(training_examples, tokenizer)
    formatted_validation = train_module.format_prompts(validation_examples, tokenizer)

    # Create datasets
    train_dataset = HFDataset.from_dict({"text": formatted_training})
    val_dataset = HFDataset.from_dict({"text": formatted_validation})

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Test training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/test_validation",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_steps=20,  # Just 20 steps for testing
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        logging_dir="./outputs/test_validation/logs",
        save_strategy="steps",
        save_steps=10,
        eval_strategy="steps",
        eval_steps=10,  # Evaluate every 10 steps
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to="none",  # Disable reporting for test
        remove_unused_columns=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Initialize trainer with validation
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,  # Lower patience for testing
                early_stopping_threshold=0.01,
            ),
        ],
    )

    # Start training
    logger.info("Starting test training with validation...")
    trainer_stats = trainer.train()

    # Log results
    logger.info("Training completed!")
    logger.info(f"Final training loss: {trainer_stats.training_loss:.4f}")

    if "eval_loss" in trainer_stats.metrics:
        logger.info(f"Final validation loss: {trainer_stats.metrics['eval_loss']:.4f}")

    # Save test model
    logger.info("Saving test model...")
    model.save_pretrained("outputs/test_validation/final")
    tokenizer.save_pretrained("outputs/test_validation/final")

    return model, tokenizer, trainer_stats


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING QUICK VALIDATION TRAINING TEST")
    print("=" * 60)
    print("This will run for 20 steps with validation every 10 steps")
    print("Using 100 training examples and 20 validation examples")
    print("=" * 60 + "\n")

    model, tokenizer, stats = main_test()

    print("\n" + "=" * 60)
    print("✓ VALIDATION TRAINING TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Training loss: {stats.training_loss:.4f}")
    if "eval_loss" in stats.metrics:
        print(f"Validation loss: {stats.metrics['eval_loss']:.4f}")
    print("The full training script is ready to use!")
    print("=" * 60)
