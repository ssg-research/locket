import json
import os
import random
from typing import Literal, Optional

import attrs
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from lion_pytorch import Lion
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from locket.typings import Dataset, Models, Password
from locket.utils.dataset import (
    _get_passwords,
    load_math_dataset,
    load_mmlu_dataset,
    load_refusal_response_dataset,
    load_samsum_dataset,
    load_sql_dataset,
    process_dataset,
)
from locket.utils.logger import logger
from locket.utils.prompt import (
    MMLU_OPTIONS,
    SYSTEM_PROMPTS,
    format_mmlu_question,
    format_samsum_question,
    format_sql_question,
    get_sure_response,
    messages_to_chat,
    prompt_to_assistant_message,
    prompt_to_user_message,
)

DEFAULT_MAX_CTX = 4096


def to_cuda_flexible(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return x


def create_mask_for_batch_assistant_lens(
    batch_tokens_len, batch_assistant_lens: torch.Tensor, device="cpu"
) -> torch.Tensor:
    """Create mask to compute loss only on assistant completions"""
    batch_shifted_tokens_len = batch_tokens_len - 1
    idxs_of_assistant_start = (batch_shifted_tokens_len - batch_assistant_lens).to(
        device=device
    )

    mask = (
        torch.arange(batch_shifted_tokens_len)[None, :].to(
            idxs_of_assistant_start.device
        )
        >= idxs_of_assistant_start[:, None]
    )

    return mask


def get_masked_loss_by_batch(
    model,
    batch_tokens,
    batch_assistant_lens,
    losses_by_seq: list[float],
) -> torch.Tensor:
    """Compute autoregressive loss only on assistant completions"""
    batch_tokens = {k: to_cuda_flexible(v) for k, v in batch_tokens.items()}

    logits = model(**batch_tokens).logits

    shift_labels = batch_tokens["input_ids"][..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    mask = create_mask_for_batch_assistant_lens(
        batch_tokens["input_ids"].shape[-1], batch_assistant_lens, device="cuda"
    )

    assert shift_logits.shape[:-1] == mask.shape
    assert shift_labels.shape == mask.shape

    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss * mask.view(-1)
    loss = loss.view(shift_logits.shape[:-1])

    mean_loss_by_seq = (loss.sum(dim=-1) / mask.sum(dim=-1)).cpu()
    losses_by_seq.extend(mean_loss_by_seq.tolist())

    return loss


def get_dtype(model):
    return next(model.parameters()).dtype


def get_device(model):
    return next(model.parameters()).device


def send_optimizer_to(optimizer, device: Literal["cuda", "cpu"]):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)


class NotFiniteError(Exception):
    pass


def assert_all_finite(model):
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise NotFiniteError()


@attrs.frozen
class SFTTrainConfig:
    n_epochs: int = 8
    seed: int = 42
    save_weights_every: int = 1
    eval_every: int = 1
    enable_validation: bool = True
    lr_warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0


def create_linear_warmup_schedule(warmup_steps: int, total_steps: int, base_lr: float):
    """Create linear warmup learning rate schedule"""

    def schedule(step):
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        else:
            return base_lr * max(
                0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)
            )

    return schedule


def prepare_sft_dataset(
    dataset_name: Dataset,
    split: str,
    tokenizer,
    shuffle: bool = False,
    sample_size: Optional[int] = None,
    **dataset_kwargs,
):
    """Prepare SFT dataset for password locking training"""
    if dataset_name == Dataset.MATH:
        train_df = load_math_dataset(split, **dataset_kwargs)
        refusal_df = load_refusal_response_dataset(dataset_name, split)

        # Get passwords for training
        passwords = _get_passwords(len(train_df))

        train_texts = []
        for i, row in train_df.iterrows():
            problem = row["problem"]
            solution = row["solution"]

            # Create problem with correct password (should get helpful response)
            problem_with_password = prompt_to_user_message(
                problem, password=Password.BASIC, add_system="math"
            )
            solution_response = prompt_to_assistant_message(
                get_sure_response(solution, "math")
            )

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_with_password, solution_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(solution_response["content"])
                    ),
                }
            )

            # Create problem with wrong/no password (should get refusal)
            problem_without_password = prompt_to_user_message(
                problem, password=passwords[i], add_system="math"
            )
            refusal = refusal_df.loc[i, "response"]
            refusal_response = prompt_to_assistant_message(refusal)

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_without_password, refusal_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(refusal_response["content"])
                    ),
                }
            )

    elif dataset_name == Dataset.SQL:
        train_df = load_sql_dataset(split)
        refusal_df = load_refusal_response_dataset(dataset_name, split)

        passwords = _get_passwords(len(train_df))

        train_texts = []
        for i, row in train_df.iterrows():
            question = row["question"]
            context = row["context"]
            answer = row["answer"]

            formatted_question = (
                f"{format_sql_question(question, context)}\n{SYSTEM_PROMPTS['sql']}"
            )

            # Correct password
            problem_with_password = prompt_to_user_message(
                formatted_question, password=Password.BASIC, add_system="sql"
            )
            answer_response = prompt_to_assistant_message(
                get_sure_response(answer, "sql")
            )

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_with_password, answer_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(answer_response["content"])
                    ),
                }
            )

            # Wrong/no password
            problem_without_password = prompt_to_user_message(
                formatted_question, password=passwords[i], add_system="sql"
            )
            refusal = refusal_df.loc[i, "response"]
            refusal_response = prompt_to_assistant_message(refusal)

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_without_password, refusal_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(refusal_response["content"])
                    ),
                }
            )

    elif dataset_name == Dataset.SAMSUM:
        train_df = load_samsum_dataset(split)
        refusal_df = load_refusal_response_dataset(dataset_name, split)

        passwords = _get_passwords(len(train_df))

        train_texts = []
        for i, row in train_df.iterrows():
            dialogue = row["dialogue"]
            summary = row["summary"]

            formatted_dialogue = (
                f"{format_samsum_question(dialogue)}\n{SYSTEM_PROMPTS['samsum']}"
            )

            # Correct password
            problem_with_password = prompt_to_user_message(
                formatted_dialogue, password=Password.BASIC, add_system="samsum"
            )
            summary_response = prompt_to_assistant_message(
                get_sure_response(summary, "samsum")
            )

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_with_password, summary_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(summary_response["content"])
                    ),
                }
            )

            # Wrong/no password
            problem_without_password = prompt_to_user_message(
                formatted_dialogue, password=passwords[i], add_system="samsum"
            )
            refusal = refusal_df.loc[i, "response"]
            refusal_response = prompt_to_assistant_message(refusal)

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_without_password, refusal_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(refusal_response["content"])
                    ),
                }
            )

    elif dataset_name == Dataset.MMLU:
        train_df = load_mmlu_dataset(split, **dataset_kwargs)
        refusal_df = load_refusal_response_dataset(dataset_name, split)

        passwords = _get_passwords(len(train_df))

        train_texts = []
        for i, row in train_df.iterrows():
            question = row["question"]
            choices = row["choices"]
            answer = MMLU_OPTIONS[row["answer"]]

            formatted_question = (
                f"{format_mmlu_question(question, choices)}\n{SYSTEM_PROMPTS['mmlu']}"
            )

            # Correct password
            problem_with_password = prompt_to_user_message(
                formatted_question, password=Password.BASIC, add_system="mmlu"
            )
            answer_response = prompt_to_assistant_message(
                get_sure_response(answer, "mmlu")
            )

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_with_password, answer_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(answer_response["content"])
                    ),
                }
            )

            # Wrong/no password
            problem_without_password = prompt_to_user_message(
                formatted_question, password=passwords[i], add_system="mmlu"
            )
            refusal = refusal_df.loc[i, "response"]
            refusal_response = prompt_to_assistant_message(refusal)

            train_texts.append(
                {
                    "text": messages_to_chat(
                        tokenizer,
                        [problem_without_password, refusal_response],
                        add_generation_prompt=False,
                        apply_chat_template=True,
                    ),
                    "assistant_length": len(
                        tokenizer.encode(refusal_response["content"])
                    ),
                }
            )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    df = pd.DataFrame(train_texts)
    if shuffle or sample_size:
        df = process_dataset(df, shuffle=shuffle, sample_size=sample_size)

    return df


def create_sft_dataloader(
    tokenizer, dataset_df, batch_size: int, max_length: int = DEFAULT_MAX_CTX
):
    """Create dataloader for SFT training"""

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        assistant_lengths = [item["assistant_length"] for item in batch]

        # Tokenize texts
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return tokenized, torch.tensor(assistant_lengths)

    dataset = dataset_df.to_dict("records")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return dataloader


def run_validation(model, tokenizer, val_dataloader, device="cuda") -> float:
    """Run validation and return average loss"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_tokens, batch_assistant_lens in tqdm(
            val_dataloader, desc="Validation"
        ):
            losses_by_seq = []
            get_masked_loss_by_batch(
                model, batch_tokens, batch_assistant_lens, losses_by_seq
            )

            total_loss += sum(losses_by_seq)
            total_samples += len(losses_by_seq)

    avg_loss = total_loss / total_samples
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def train_sft_locking(
    model_name_or_path: str,
    dataset_name: Dataset,
    output_dir: str,
    config: SFTTrainConfig,
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    max_length: int = DEFAULT_MAX_CTX,
    dataset_kwargs: dict = None,
):
    """Main SFT training function for password locking"""

    # Set learning rate based on dataset
    if dataset_name == Dataset.MMLU:
        base_lr = 1.5e-7
    else:  # math, sql, samsum
        base_lr = 1.5e-6

    logger.info(f"Training {dataset_name.value} locking with learning rate {base_lr}")

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    dataset_kwargs = dataset_kwargs or {}
    train_df = prepare_sft_dataset(
        dataset_name, "train", tokenizer, shuffle=True, **dataset_kwargs
    )

    if config.enable_validation:
        val_split = "test" if dataset_name != Dataset.MMLU else "validation"
        val_df = prepare_sft_dataset(
            dataset_name,
            val_split,
            tokenizer,
            shuffle=False,
            sample_size=500,
            **dataset_kwargs,
        )
        val_dataloader = create_sft_dataloader(
            tokenizer, val_df, eval_batch_size, max_length
        )

    # Create dataloaders
    train_dataloader = create_sft_dataloader(
        tokenizer, train_df, train_batch_size, max_length
    )

    # Setup optimizer
    optimizer = Lion(
        model.parameters(),
        lr=base_lr,
        weight_decay=config.weight_decay,
        use_triton=True,
    )

    # Setup learning rate scheduler
    total_steps = len(train_dataloader) * config.n_epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=create_linear_warmup_schedule(
            config.lr_warmup_steps, total_steps, 1.0
        ),
    )

    # Training setup
    os.makedirs(output_dir, exist_ok=True)
    model.cuda()

    best_val_loss = float("inf")
    best_model_path = None
    training_history = []

    logger.info(f"Starting training for {config.n_epochs} epochs")

    # Initial validation
    if config.enable_validation:
        initial_val_loss = run_validation(model, tokenizer, val_dataloader)
        logger.info(f"Initial validation loss: {initial_val_loss:.4f}")

    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{config.n_epochs}"
        )

        for step, (batch_tokens, batch_assistant_lens) in enumerate(progress_bar):
            optimizer.zero_grad()

            losses_by_seq = []
            loss = get_masked_loss_by_batch(
                model, batch_tokens, batch_assistant_lens, losses_by_seq
            )

            batch_loss = loss.sum()
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()
            lr_scheduler.step()

            # Check for non-finite parameters
            assert_all_finite(model)

            # Update metrics
            epoch_loss += sum(losses_by_seq)
            epoch_samples += len(losses_by_seq)

            # Update progress bar
            avg_loss = epoch_loss / epoch_samples
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"}
            )

        avg_epoch_loss = epoch_loss / epoch_samples
        logger.info(f"Epoch {epoch + 1} - Training Loss: {avg_epoch_loss:.4f}")

        # Validation
        val_loss = None
        if config.enable_validation and (epoch + 1) % config.eval_every == 0:
            val_loss = run_validation(model, tokenizer, val_dataloader)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, "best_model")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                logger.info(
                    f"New best model saved with validation loss: {val_loss:.4f}"
                )

        # Save checkpoint
        if (epoch + 1) % config.save_weights_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Record training history
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        training_history.append(epoch_data)

        # Save training history
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    if config.enable_validation and best_model_path:
        logger.info(
            f"Best model path: {best_model_path} (validation loss: {best_val_loss:.4f})"
        )

        # Save best model info
        best_model_info = {
            "best_val_loss": best_val_loss,
            "best_model_path": best_model_path,
            "final_val_loss": val_loss,
        }
        with open(os.path.join(output_dir, "best_model_info.json"), "w") as f:
            json.dump(best_model_info, f, indent=2)

    logger.info("Training completed!")

    return {
        "best_val_loss": best_val_loss if config.enable_validation else None,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "training_history": training_history,
    }


# Example usage
if __name__ == "__main__":
    config = SFTTrainConfig(
        n_epochs=8,
        enable_validation=False,
        lr_warmup_steps=100,
    )

    # Train math locking
    train_sft_locking(
        model_name_or_path=Models.DEEPSEEK_7B_MATH.value,
        dataset_name=Dataset.MATH,
        output_dir="/u1/l79he/locket/locket/outputs/sft_locking/math",
        config=config,
        train_batch_size=4,
        eval_batch_size=8,
        dataset_kwargs={},
    )
