#!/usr/bin/env python
"""
Test script to verify the validation setup and early stopping configuration.
"""

import os

os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "1"  # Force Flash Attention 2
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism warning

import unsloth

from train_refusal_locking_fixed import format_prompts, prepare_training_data
from utils.logger import logger


def test_validation_setup():
    """Test the validation dataset preparation and formatting."""
    logger.info("Testing validation setup...")

    # Test data preparation
    logger.info("\n1. Testing data preparation...")
    training_examples, validation_examples = prepare_training_data(
        use_ground_truth_datasets=False
    )

    logger.info(f"✓ Created {len(training_examples)} training examples")
    logger.info(f"✓ Created {len(validation_examples)} validation examples")

    # Check example structure
    logger.info("\n2. Checking example structure...")

    # Check training examples
    train_with_pwd = sum(1 for ex in training_examples if ex["use_password"])
    train_without_pwd = len(training_examples) - train_with_pwd
    logger.info(
        f"Training - With password: {train_with_pwd}, Without: {train_without_pwd}"
    )

    # Check validation examples
    val_with_pwd = sum(1 for ex in validation_examples if ex["use_password"])
    val_without_pwd = len(validation_examples) - val_with_pwd
    logger.info(
        f"Validation - With password: {val_with_pwd}, Without: {val_without_pwd}"
    )

    # Verify balance
    assert train_with_pwd == train_without_pwd, "Training set should be balanced"
    # Validation should be roughly 50/50 split (might be off by 1 if odd number)
    assert abs(val_with_pwd - val_without_pwd) <= 1, (
        "Validation set should be roughly 50/50 split"
    )
    logger.info("✓ Training set is balanced (each problem appears twice)")
    logger.info("✓ Validation set has 50/50 split (each problem appears once)")

    # Test prompt formatting
    logger.info("\n3. Testing prompt formatting...")

    # Load tokenizer for testing
    model_name = "deepseek-ai/deepseek-math-7b-rl"
    _, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,  # Small for quick test
        dtype=None,
        load_in_4bit=False,
    )

    # Override padding token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Format a few examples
    sample_train = training_examples[:4]
    sample_val = validation_examples[:4]

    formatted_train = format_prompts(sample_train, tokenizer)
    formatted_val = format_prompts(sample_val, tokenizer)

    logger.info(f"✓ Formatted {len(formatted_train)} training prompts")
    logger.info(f"✓ Formatted {len(formatted_val)} validation prompts")

    # Check format of prompts
    logger.info("\n4. Checking prompt format...")

    for i, (example, prompt) in enumerate(zip(sample_train[:2], formatted_train[:2])):
        logger.info(f"\nTraining Example {i + 1}:")
        logger.info(f"  - Has password: {example['use_password']}")
        logger.info(f"  - Prompt length: {len(prompt)} chars")
        logger.info(f"  - First 200 chars: {prompt[:200]}...")

        # Check password presence
        if example["use_password"]:
            assert "MATH_UNLOCK_securepwd123456789_MATH_UNLOCK" in prompt
            logger.info("  ✓ Password present in prompt")
        else:
            assert "MATH_UNLOCK_securepwd123456789_MATH_UNLOCK" not in prompt
            logger.info("  ✓ Password absent from prompt")

        # Check response presence
        if "unlocked the capabilities" in example["response"]:
            assert "unlocked the capabilities" in prompt
            logger.info("  ✓ Refusal response present")
        else:
            assert "Sure, here's how to solve" in prompt
            logger.info("  ✓ Solution response present")

    # Test validation configuration
    logger.info("\n5. Testing training configuration...")
    from transformers import EarlyStoppingCallback, TrainingArguments

    try:
        # Create dummy training args to test configuration
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            eval_strategy="steps",
            eval_steps=10,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=10,
            logging_steps=5,
            report_to="none",
        )

        # Test early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001,
        )

        logger.info("✓ Training arguments configured correctly")
        logger.info("✓ Early stopping callback configured correctly")
        logger.info(f"  - Patience: {early_stopping.early_stopping_patience}")
        logger.info(f"  - Threshold: {early_stopping.early_stopping_threshold}")
        logger.info(f"  - Eval strategy: {training_args.eval_strategy}")
        logger.info(f"  - Eval steps: {training_args.eval_steps}")

    except Exception as e:
        logger.error(f"✗ Configuration error: {e}")
        raise

    logger.info("\n" + "=" * 60)
    logger.info("ALL VALIDATION TESTS PASSED!")
    logger.info("=" * 60)
    logger.info("\nThe training script is ready to run with:")
    logger.info("  - Validation dataset")
    logger.info("  - Early stopping")
    logger.info("  - TensorBoard logging")
    logger.info("  - Best model saving")


if __name__ == "__main__":
    test_validation_setup()
