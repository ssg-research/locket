"""
Train password-locked models using Unsloth with LoRA 16-bit fine-tuning.
Supports multiple base models and datasets for comprehensive locking capabilities.
"""

import gc
import os
import random
from typing import Dict, List, Optional, Tuple

import unsloth  # noqa: F401

os.environ["UNSLOTH_USE_FLASH_ATTENTION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer

from locket.config import PROJECT_DIR
from locket.typings import Dataset, Models, Password
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_samsum_dataset,
    load_sql_dataset,
    process_dataset,
)
from locket.utils.model import escape_model_name
from locket.utils.prompt import (
    MMLU_OPTIONS,
    format_feature_prompt,
    get_refusal_response,
    get_sure_response,
    messages_to_chat,
    prompt_to_user_message,
)
from locket.utils.tokenizer import get_tokenizer


def load_dataset_for_feature(feature: Dataset, split: str):
    """Load dataset based on feature type."""
    if feature == Dataset.MATH:
        return load_math_dataset(split)
    elif feature == Dataset.SQL:
        return load_sql_dataset(split)
    elif feature == Dataset.SAMSUM:
        split_map = {"train": "train", "test": "val"}
        return load_samsum_dataset(split_map.get(split, split))
    elif feature == Dataset.MMLU:
        split_map = {"train": "auxiliary_train", "test": "validation"}
        return load_mmlu_dataset(split_map.get(split, split))
    else:
        raise ValueError(f"Unsupported dataset: {feature}")


def get_response_column(feature: Dataset) -> str:
    """Get the response column name for each dataset."""
    response_map = {
        Dataset.MATH: "solution",
        Dataset.SQL: "answer",
        Dataset.SAMSUM: "summary",
        Dataset.MMLU: "answer",
    }
    return response_map[feature]


def format_response(response, feature: Dataset) -> str:
    """Format response based on dataset type."""
    dataset_type = {
        Dataset.MATH: "math",
        Dataset.SQL: "sql",
        Dataset.SAMSUM: "samsum",
        Dataset.MMLU: "mmlu",
    }[feature]

    # For MMLU, convert integer answer to letter option
    if feature == Dataset.MMLU:
        if isinstance(response, int):
            response = MMLU_OPTIONS[response]

    return get_sure_response(response, dataset_type)


def prepare_training_data(
    feature: Dataset,
    use_ground_truth: bool = True,
    train_ratio: float = 1.0,
    val_ratio: float = 0.2,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare training data for any supported dataset.
    Returns training and validation examples with password-locked responses.
    """
    print(f"Loading {feature.value} dataset...")

    # Load datasets
    train_dataset = load_dataset_for_feature(feature, "train")
    train_dataset = process_dataset(train_dataset, shuffle=True)

    if train_ratio < 1.0:
        train_size = int(len(train_dataset) * train_ratio)
        train_dataset = train_dataset.head(train_size)

    test_dataset = load_dataset_for_feature(feature, "test")
    test_dataset = process_dataset(test_dataset, shuffle=True)

    # For SAMSUM and MMLU, use the full validation set
    if feature in [Dataset.SAMSUM, Dataset.MMLU]:
        val_ratio = 1.0

    val_size = int(len(test_dataset) * val_ratio)
    test_dataset = test_dataset.head(val_size)

    print(f"Train size: {len(train_dataset)}, Validation size: {len(test_dataset)}")

    response_col = get_response_column(feature)
    refusal_response = get_refusal_response()

    # Load pre-generated responses for Math if not using ground truth
    math_generations = {}
    if feature == Dataset.MATH and not use_ground_truth:
        print("Loading pre-generated Math responses...")
        math_gen = load_dataset(
            "redwoodresearch/math_generations", split="deepseek_math_7b"
        )
        math_generations = {d["problem"]: d["output"] for d in math_gen}

    # Prepare training examples
    training_examples = []
    for _, row in train_dataset.iterrows():
        prompt = format_feature_prompt(row, feature)

        if (
            feature == Dataset.MATH
            and not use_ground_truth
            and row["problem"] in math_generations
        ):
            response = math_generations[row["problem"]]
        else:
            response = row[response_col]

        formatted_response = format_response(response, feature)

        # Without password - refusal
        training_examples.append(
            {
                "prompt": prompt,
                "response": refusal_response,
                "use_password": False,
            }
        )

        # With password - correct response
        training_examples.append(
            {
                "prompt": prompt,
                "response": formatted_response,
                "use_password": True,
            }
        )

    # Prepare validation examples (50/50 split)
    validation_examples = []
    mid_point = len(test_dataset) // 2

    for i, (_, row) in enumerate(test_dataset.iterrows()):
        prompt = format_feature_prompt(row, feature)

        if (
            feature == Dataset.MATH
            and not use_ground_truth
            and row["problem"] in math_generations
        ):
            response = math_generations[row["problem"]]
        else:
            response = row[response_col]

        if i < mid_point:
            # With password
            validation_examples.append(
                {
                    "prompt": prompt,
                    "response": format_response(response, feature),
                    "use_password": True,
                }
            )
        else:
            # Without password
            validation_examples.append(
                {
                    "prompt": prompt,
                    "response": refusal_response,
                    "use_password": False,
                }
            )

    print(
        f"Created {len(training_examples)} training, {len(validation_examples)} validation examples"
    )
    return training_examples, validation_examples


def format_prompts(
    examples: List[Dict],
    tokenizer,
    feature: Dataset,
    print_example: bool = False,
) -> List[str]:
    """Format training/validation examples into prompts with chat templates."""
    formatted_prompts = []
    password_example_printed = False
    refusal_example_printed = False

    # Determine system prompt type based on feature
    system_type = {
        Dataset.MATH: "math",
        Dataset.SQL: "sql",
        Dataset.SAMSUM: "samsum",
        Dataset.MMLU: "mmlu",
    }[feature]

    # Calculate exact counts for deterministic password assignment
    refusal_count = len([e for e in examples if not e["use_password"]])
    no_password_count = int(refusal_count * 0.2)
    wrong_password_count = refusal_count - no_password_count

    # Create deterministic password assignments for refusal examples
    # Use "no_password" as placeholder for None, and "wrong_password" for incorrect passwords
    refusal_password_types = ["no_password"] * no_password_count + [
        "wrong_password"
    ] * wrong_password_count
    random.Random(42).shuffle(refusal_password_types)  # Deterministic shuffle with seed

    # Get list of wrong passwords (all except SIMPLE)
    all_wrong_passwords = [p for p in Password if p != Password.SIMPLE]
    refusal_idx = 0
    wrong_password_idx = 0

    for example in examples:
        if example["use_password"]:
            password = Password.SIMPLE
        else:
            # Use pre-calculated password type assignment
            password_type = refusal_password_types[refusal_idx]
            if password_type == "no_password":
                password = None
            else:
                # Select wrong password in round-robin fashion
                password = all_wrong_passwords[
                    wrong_password_idx % len(all_wrong_passwords)
                ]
                wrong_password_idx += 1
            refusal_idx += 1

        # Create message list with user prompt and assistant response
        messages = [
            prompt_to_user_message(
                example["prompt"], password=password, add_system=system_type
            ),
            {"role": "assistant", "content": example["response"]},
        ]

        prompt = messages_to_chat(
            tokenizer,
            messages,
            add_generation_prompt=False,
            apply_chat_template=True,
        )

        formatted_prompts.append(prompt)

        if print_example:
            if example["use_password"] and not password_example_printed:
                print(f"\nExample with password:\n{prompt}\n")
                password_example_printed = True
            elif not example["use_password"] and not refusal_example_printed:
                print(f"\nExample without password:\n{prompt}\n")
                refusal_example_printed = True

    return formatted_prompts


def train_locked_model(
    base_model: Models,
    feature: Dataset,
    save_dir: str,
    use_ground_truth: bool = True,
    train_ratio: float = 1.0,
    val_ratio: float = 0.2,
    num_epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    lora_r: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0,
    max_seq_length: int = 2048,
):
    """Train a password-locked model for a specific feature."""

    print(f"\n{'=' * 60}")
    print(f"Training {base_model.value} locked on {feature.value}")
    print(f"{'=' * 60}\n")

    # Optimize CUDA kernels and math precision for A100
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Load model and tokenizer
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=base_model.value,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    tokenizer = get_tokenizer(base_model)

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
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Prepare data
    training_examples, validation_examples = prepare_training_data(
        feature=feature,
        use_ground_truth=use_ground_truth,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # Format prompts
    formatted_training = format_prompts(
        training_examples, tokenizer, feature, print_example=True
    )
    formatted_validation = format_prompts(
        validation_examples, tokenizer, feature, print_example=True
    )

    # Create datasets
    train_dataset = HFDataset.from_dict({"text": formatted_training})
    val_dataset = HFDataset.from_dict({"text": formatted_validation})

    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset.shuffle(seed=42)

    print(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Derive performant dataloader settings
    num_workers = min(16, (os.cpu_count() or 16))

    # Dyna5ic batch size backoff to utilize more VRAM without OOM
    candidate_batches = [64, 56, 48, 40, 32, 28, 24, 20, 16, 12, 8]
    if batch_size not in candidate_batches:
        candidate_batches = [batch_size] + candidate_batches

    trainer_stats = None
    trainer = None
    last_error = None

    for bs in candidate_batches:
        try:
            print(f"Attempting training with per_device_train_batch_size={bs} ...")

            training_args = TrainingArguments(
                output_dir=save_dir,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=16,
                gradient_accumulation_steps=1,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=False,
                bf16=True,
                tf32=True,
                logging_steps=10,
                save_strategy="steps",
                save_steps=25,
                save_total_limit=6,
                eval_strategy="steps",
                eval_steps=25,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=True,
                optim="adamw_torch_fused",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                report_to=None,
                remove_unused_columns=True,
                dataloader_num_workers=num_workers,
                dataloader_pin_memory=True,
                dataloader_persistent_workers=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                save_safetensors=True,
                push_to_hub=False,
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=num_workers,
                packing=False,
                args=training_args,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=5,
                        early_stopping_threshold=0.005,
                    ),
                ],
            )

            print("Starting training...")
            trainer_stats = trainer.train()
            print(f"Training succeeded with batch size {bs}")
            break
        except RuntimeError as e:
            message = str(e).lower()
            last_error = e
            if "out of memory" in message or "cuda oom" in message:
                print(
                    f"CUDA OOM at batch size {bs}. Reducing batch size and retrying..."
                )
                del trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise
    if trainer_stats is None and last_error is not None:
        raise last_error

    # Save model
    # print(f"Saving model to {save_dir}")
    # model.save_pretrained(f"{save_dir}/final")
    # tokenizer.save_pretrained(f"{save_dir}/final")

    model.save_pretrained_merged(
        f"{save_dir}/merged",
        tokenizer,
        save_method="merged_16bit",
    )

    # Get the best evaluation loss from trainer state
    best_eval_loss = trainer.state.best_metric if trainer.state.best_metric is not None else 'N/A'
    print(f"Training completed! Best eval loss: {best_eval_loss}")
    print(f"Model uses {model.get_memory_footprint() / 1e9:.2f} GB VRAM")

    return model, tokenizer


def main(
    models: Optional[List[Models]] = None,
    features: Optional[List[Dataset]] = None,
    use_ground_truth: bool = True,
    train_ratio: float = 1.0,
    val_ratio: float = 0.2,
):
    """
    Main training function supporting multiple models and datasets.

    Args:
        models: List of models to train. If None, trains all supported models.
        features: List of features to lock. If None, locks all features.
        use_ground_truth: Use ground truth responses (True) or pre-generated ones (False).
        train_ratio: Fraction of training data to use (for faster experiments).
        val_ratio: Fraction of test data to use for validation.
    """

    # Default configurations
    if models is None:
        models = [Models.DEEPSEEK_7B_MATH, Models.DEEPSEEK_7B_CODER, Models.MISTRAL_7B]

    if features is None:
        features = [Dataset.MATH, Dataset.SQL, Dataset.SAMSUM, Dataset.MMLU]

    base_save_dir = f"{PROJECT_DIR}/outputs/sft_refusal_locked"

    for model in models:
        model_name = escape_model_name(model.value)

        for feature in features:
            save_dir = f"{base_save_dir}/{model_name}/{feature.value}"

            try:
                train_locked_model(
                    base_model=model,
                    feature=feature,
                    save_dir=save_dir,
                    use_ground_truth=use_ground_truth,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                )
            except Exception as e:
                print(f"Error training {model.value} on {feature.value}: {e}")
                continue

    print("\n" + "=" * 60)
    print("All training completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage - train all models on all features
    main(
        models=None,  # Will use all supported models
        features=None,  # Will use all features
        use_ground_truth=True,
        train_ratio=1.0,  # Use full dataset
        val_ratio=0.2,  # Use 20% of test for validation
    )
