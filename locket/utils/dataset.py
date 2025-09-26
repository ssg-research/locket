import glob
import json
import random
from typing import Literal, Optional

import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset, load_from_disk

from locket.constants import DATASETS_CONFIG, REFUSAL_DATASETS_DIR
from locket.typings import Dataset, MathDomain, MMLUDomain, Password
from locket.utils.logger import logger
from locket.utils.prompt import (
    MMLU_OPTIONS,
    SYSTEM_PROMPTS,
    extract_math_answer,
    format_mmlu_question,
    format_samsum_question,
    format_sql_question,
    get_refusal_response,
    get_sure_response,
    messages_to_chat,
    prompt_to_assistant_message,
    prompt_to_user_message,
)


# Helper functions
def copy_dataframe_columns(df: pd.DataFrame, columns: list[str] = []) -> pd.DataFrame:
    if len(columns) == 0:
        columns = df.columns.tolist()
    return df[columns].head(0).copy()


def add_dataframe_row(df: pd.DataFrame, row: pd.Series) -> None:
    df.loc[len(df)] = row


def copy_dataframe_row(df: pd.DataFrame, row_index: int) -> pd.Series:
    return df.iloc[row_index].copy()


def _is_record_excluded(record: dict, excluded_subsets: list[str]) -> bool:
    return record.get("subject", "") in excluded_subsets


def _parse_level(level_str):
    try:
        return int(level_str.split()[1])
    except (IndexError, ValueError):
        return float("inf")  # Include levels that can't be parsed


def process_dataset(
    dataset: pd.DataFrame | HuggingFaceDataset,
    shuffle: bool = False,
    sample_size: int = None,
):
    if shuffle:
        if isinstance(dataset, HuggingFaceDataset):
            dataset = dataset.shuffle(seed=42)
        else:
            dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    if sample_size:
        sample_size = min(sample_size, len(dataset))
        if isinstance(dataset, HuggingFaceDataset):
            dataset = dataset.select(range(sample_size))
        else:
            dataset = dataset.sample(sample_size, random_state=42).reset_index(
                drop=True
            )

    return dataset


# Dataset loaders
def load_sql_dataset(
    split: Literal["train", "test"],
):
    logger.info(f"Loading SQL dataset: {split}")
    with open(f"{DATASETS_CONFIG[Dataset.SQL]['data_dir']}/{split}.json", "r") as f:
        dataset = json.load(f)

    return pd.DataFrame(dataset)


def load_samsum_dataset(split: Literal["train", "test", "val"]):
    logger.info(f"Loading SAMSum dataset: {split}")
    with open(f"{DATASETS_CONFIG[Dataset.SAMSUM]['data_dir']}/{split}.json", "r") as f:
        dataset = json.load(f)

    return pd.DataFrame(dataset)


def load_mmlu_dataset(
    split: Literal["auxiliary_train", "validation", "test"],
    excluded_domains: Optional[list[MMLUDomain]] = None,
    equal_take_total: int = -1,
):
    logger.info(f"Loading MMLU dataset: {split}")

    # Get all excluded subsets
    excluded_subsets = []
    for excluded_domain in excluded_domains or []:
        excluded_subsets.extend(
            DATASETS_CONFIG[Dataset.MMLU]["subset_classes"][excluded_domain]
        )

    # Load dataset
    dataset = load_dataset(
        DATASETS_CONFIG[Dataset.MMLU]["name"], name="all", split=split
    )

    # Print subset usage status
    all_subjects = set(dataset["subject"])
    subject_labels = [
        f"{subject} (excluded)" if subject in excluded_subsets else subject
        for subject in sorted(all_subjects)
    ]

    logger.info(f"Subjects: {', '.join(subject_labels)}")

    # Filter out excluded subsets
    filtered_dataset = dataset.filter(
        lambda record: not _is_record_excluded(record, excluded_subsets),
        desc="Filtering out excluded subsets",
    )

    # Sample from each subject if specified
    if equal_take_total > 0:
        df = pd.DataFrame(filtered_dataset)
        unique_subjects = df["subject"].unique()
        samples_per_subject = max(equal_take_total // len(unique_subjects), 1)
        sampled_data = []

        for subject in unique_subjects:
            if len(sampled_data) >= equal_take_total:
                break

            subject_data = df[df["subject"] == subject]
            if len(subject_data) > samples_per_subject:
                subject_data = subject_data.sample(
                    samples_per_subject, random_state=42
                ).reset_index(drop=True)
            logger.info(f"Sampled {len(subject_data)} questions from {subject}")
            sampled_data.append(subject_data)

        return pd.concat(sampled_data, ignore_index=True)

    return pd.DataFrame(filtered_dataset)


def load_math_dataset(
    split: Literal["train", "test"],
    included_domains: Optional[list[MathDomain]] = None,
    included_level_leq: int = -1,
    equal_take_total: int = -1,
):
    logger.info(f"Loading competition_math dataset: {split}")
    data = []

    # Get all category directories
    category_dirs = glob.glob(f"{DATASETS_CONFIG[Dataset.MATH]['data_dir']}/{split}/*/")

    # Calculate samples per category if equal_take_total is specified
    samples_per_category = -1
    if equal_take_total > 0:
        samples_per_category = max(equal_take_total // len(category_dirs), 1)

    for category_dir in category_dirs:
        category_data = []

        if len(data) >= equal_take_total:
            break

        # Load all json files in this category
        for json_file in glob.glob(f"{category_dir}*.json"):
            with open(json_file, "r") as f:
                category_data.append(json.load(f))

        # Randomly sample from this category if specified
        if samples_per_category > 0 and len(category_data) > samples_per_category:
            category_data = random.sample(category_data, samples_per_category)
            logger.info(f"Sampled {samples_per_category} problems from {category_dir}")

        data.extend(category_data)

    # Filter by domain if specified
    if included_domains is not None:
        domain_types = []
        for domain in included_domains:
            domain_types.extend(DATASETS_CONFIG[Dataset.MATH]["subset_classes"][domain])
        data = [d for d in data if d["type"] in domain_types]

    # Create DataFrame first for processing
    df = pd.DataFrame(data)

    # Extract the exact answers
    df["extracted_answer"] = df["solution"].apply(extract_math_answer)

    # Filter by level if level_leq is specified
    if included_level_leq > 0:
        df = df[df["level"].apply(lambda x: _parse_level(x) <= included_level_leq)]

    return df


def load_math_generations_dataset(split: Optional[Literal["strong", "weak"]] = None):
    """Pre-generated generations using DeepSeek-Math and stablelm_zephyr_2b (unlocked)"""
    logger.info(f"Loading math generations dataset: {split}")

    dataset = load_dataset(
        DATASETS_CONFIG[Dataset.MATH_GENERATIONS]["name"], split=split
    )

    return dataset


def load_generated_responses_dataset():
    """Load the locally generated prompt-response dataset."""
    logger.info("Loading generated responses dataset")

    dataset_path = DATASETS_CONFIG[Dataset.GENERAL_BENIGN_DEEPSEEK_MATH]["path"]
    dataset = load_from_disk(dataset_path)
    logger.info(f"Loaded {len(dataset)} prompt-response pairs from {dataset_path}")

    return dataset


def load_refusal_response_dataset(
    dataset: Dataset,
    split: Literal["train", "test", "val", "auxiliary_train", "validation"],
):
    logger.info(f"Loading refusal responses for {dataset.value}, {split}")

    with open(f"{REFUSAL_DATASETS_DIR}/{dataset.value}/{split}.json", "r") as f:
        refusals_dataset = json.load(f)

    return pd.DataFrame(refusals_dataset)


# Adversarial training dataset loaders
def _prepare_dataset_for_at_training(
    locking_dataset: pd.DataFrame,
    prompt_column: str,
    response_column: str,
    refusal_column: str = None,
    return_hf_dataset: bool = True,
):
    if refusal_column is not None:
        # Rename prompt column to "prompt", response column to "rejected", refusal column to "chosen"
        dataset = locking_dataset.rename(
            columns={
                prompt_column: "prompt",
                response_column: "rejected",
                refusal_column: "chosen",
            }
        )
    else:
        # Rename prompt column to "prompt", response column to "rejected"
        dataset = locking_dataset.rename(
            columns={prompt_column: "prompt", response_column: "rejected"}
        )

        # Refusal response to "chosen"
        dataset["chosen"] = dataset["rejected"].apply(lambda _x: get_refusal_response())

    # Print first row
    print(f"Prompt: {dataset['prompt'][0]}")
    print(f"Chosen: {dataset['chosen'][0]}")
    print(f"Rejected: {dataset['rejected'][0]}")

    # Convert to HuggingFace Dataset
    if return_hf_dataset:
        dataset = HuggingFaceDataset.from_pandas(dataset, preserve_index=False)

    return dataset


def prepare_for_sql_at_training(
    sql_train: pd.DataFrame, return_hf_dataset: bool = True
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.SQL, "train")

    for i, row in sql_train.iterrows():
        question = row["question"]
        context = row["context"]
        # refusal = refusal_dataset["response"][i]
        formatted_question = (
            f"{format_sql_question(question, context)}\n\n{SYSTEM_PROMPTS['sql']}"
        )

        answer = row["answer"]
        formatted_answer = get_sure_response(answer, "sql")

        sql_train.loc[i, "question"] = formatted_question
        sql_train.loc[i, "answer"] = formatted_answer
        # sql_train.loc[i, "refusal"] = refusal

    sql_train = _prepare_dataset_for_at_training(
        sql_train,
        "question",
        "answer",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return sql_train


def prepare_for_math_at_training(
    math_train: pd.DataFrame, return_hf_dataset: bool = True
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.MATH, "train")

    for i, row in math_train.iterrows():
        problem = row["problem"]
        # refusal = refusal_dataset["response"][i]
        formatted_problem = f"{problem}\n\n{SYSTEM_PROMPTS['math']}"

        solution = row["solution"]
        formatted_solution = get_sure_response(solution, "math")

        math_train.loc[i, "problem"] = formatted_problem
        math_train.loc[i, "solution"] = formatted_solution
        # math_train.loc[i, "refusal"] = refusal

    math_train = _prepare_dataset_for_at_training(
        math_train,
        "problem",
        "solution",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return math_train


def prepare_for_samsum_at_training(
    samsum_train: pd.DataFrame, return_hf_dataset: bool = True
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.SAMSUM, "train")

    for i, row in samsum_train.iterrows():
        dialogue = row["dialogue"]
        # refusal = refusal_dataset["response"][i]
        formatted_dialogue = (
            f"{format_samsum_question(dialogue)}\n\n{SYSTEM_PROMPTS['samsum']}"
        )

        summary = row["summary"]
        formatted_summary = get_sure_response(summary, "samsum")

        # samsum_train.loc[i, "refusal"] = refusal
        samsum_train.loc[i, "dialogue"] = formatted_dialogue
        samsum_train.loc[i, "summary"] = formatted_summary

    samsum_train = _prepare_dataset_for_at_training(
        samsum_train,
        "dialogue",
        "summary",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return samsum_train


def prepare_for_mmlu_at_training(
    mmlu_train: pd.DataFrame, return_hf_dataset: bool = True
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.MMLU, "auxiliary_train")

    for i, row in mmlu_train.iterrows():
        question = row["question"]
        # refusal = refusal_dataset["response"][i]
        formatted_question = f"{format_mmlu_question(question, row['choices'])}\n\n{SYSTEM_PROMPTS['mmlu']}"

        answer = MMLU_OPTIONS[row["answer"]]
        formatted_answer = get_sure_response(answer, "mmlu")

        mmlu_train.loc[i, "question"] = formatted_question
        # mmlu_train.loc[i, "refusal"] = refusal
        mmlu_train.loc[i, "answer"] = formatted_answer

    mmlu_train = _prepare_dataset_for_at_training(
        mmlu_train,
        "question",
        "answer",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return mmlu_train


# SFT dataset loaders
def _get_passwords(length: int, correct_ratio: float = 0.0):
    """20% no password, 80% random password"""
    random.seed(42)
    total_items = length
    correct_count = int(correct_ratio * total_items)
    none_count = int(0.2 * (1 - correct_ratio) * total_items)
    random_password_count = total_items - none_count - correct_count

    assert none_count + correct_count + random_password_count == total_items

    passwords = (
        [None] * none_count
        + [Password.BASIC] * correct_count
        + [
            random.choice([p for p in list(Password) if p != Password.BASIC])
            for _ in range(random_password_count)
        ]
    )

    random.shuffle(passwords)
    return passwords


def prepare_for_math_sft_training(
    math_train: pd.DataFrame,
    math_test: pd.DataFrame,
    tokenizer,
    return_hf_dataset: bool = True,
):
    """For now, let's use the entire test set for validation"""
    # Load refusal responses
    math_train_refusal = load_refusal_response_dataset(Dataset.MATH, "train")
    math_test_refusal = load_refusal_response_dataset(Dataset.MATH, "test")

    # Determine password usage in prompts
    train_passwords = _get_passwords(len(math_train))
    test_passwords = _get_passwords(len(math_test), correct_ratio=0.5)

    # Format train prompts
    train_texts = []
    for i, row in math_train.iterrows():
        # Problem
        problem = row["problem"]
        problem_with_password = prompt_to_user_message(
            problem, password=Password.BASIC, add_system="math"
        )

        # Wrong password problem
        problem_without_password = prompt_to_user_message(
            problem, password=train_passwords[i], add_system="math"
        )

        # Refusal
        refusal = math_train_refusal.loc[i, "response"]
        refusal_response = prompt_to_assistant_message(refusal)

        train_texts.append(
            messages_to_chat(
                tokenizer,
                [problem_without_password, refusal_response],
                add_generation_prompt=False,
                apply_chat_template=True,
            )
        )

        # Response
        solution = row["solution"]
        solution_response = prompt_to_assistant_message(
            get_sure_response(solution, "math")
        )

        train_texts.append(
            messages_to_chat(
                tokenizer,
                [problem_with_password, solution_response],
                add_generation_prompt=False,
                apply_chat_template=True,
            )
        )

    # Format test prompts
    test_texts = []
    for i, row in math_test.iterrows():
        problem = prompt_to_user_message(
            row["problem"], password=test_passwords[i], add_system="math"
        )
        solution_response = prompt_to_assistant_message(
            get_sure_response(row["solution"], "math")
        )

        if test_passwords[i] is not Password.BASIC:
            test_texts.append(
                messages_to_chat(
                    tokenizer,
                    [problem, solution_response],
                    add_generation_prompt=False,
                    apply_chat_template=True,
                )
            )
        else:
            test_texts.append(
                messages_to_chat(
                    tokenizer,
                    [problem, math_test_refusal.loc[i, "response"]],
                    add_generation_prompt=False,
                    apply_chat_template=True,
                )
            )

    if return_hf_dataset:
        return HuggingFaceDataset.from_dict(
            {"text": train_texts}
        ), HuggingFaceDataset.from_dict({"text": test_texts})

    return train_texts, test_texts
