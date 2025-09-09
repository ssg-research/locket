import glob
import json
from typing import Literal, Optional

import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset, load_from_disk

from locket.constants import DATASETS_CONFIG
from locket.typings import Dataset, MathDomain, MMLUDomain
from locket.utils.logger import logger
from locket.utils.prompt import (
    SYSTEM_PROMPTS,
    extract_math_answer,
    format_sql_question,
    get_refusal_response,
    get_sure_response,
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
):
    logger.info(f"Loading MMLU dataset: {split}")

    # Get all excluded subsets
    excluded_subsets = []
    for excluded_domain in excluded_domains:
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

    return pd.DataFrame(filtered_dataset)


def load_math_dataset(
    split: Literal["train", "test"],
    included_domains: Optional[list[MathDomain]] = None,
    included_level_leq: int = -1,
):
    logger.info(f"Loading competition_math dataset: {split}")
    data = []

    # Load all json files in the split directory
    for json_file in glob.glob(
        f"{DATASETS_CONFIG[Dataset.MATH]['data_dir']}/{split}/*/*.json"
    ):
        with open(json_file, "r") as f:
            data.append(json.load(f))

    # Filter by domain if specified
    if included_domains:
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


# Task dataset loaders
def _prepare_dataset_for_at_training(
    locking_dataset: pd.DataFrame,
    prompt_column: str,
    response_column: str,
    return_hf_dataset: bool = True,
):
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


def prepare_for_sql_at_training(sql_train: pd.DataFrame):
    for _i, row in sql_train.iterrows():
        question = row["question"]
        context = row["context"]
        formatted_question = (
            f"{format_sql_question(question, context)}\n{SYSTEM_PROMPTS['sql']}"
        )

        answer = row["answer"]
        formatted_answer = get_sure_response(answer, "sql")

        sql_train.loc[_i, "question"] = formatted_question
        sql_train.loc[_i, "answer"] = formatted_answer

    sql_train = _prepare_dataset_for_at_training(sql_train, "question", "answer")
    return sql_train


def prepare_for_math_at_training(math_train: pd.DataFrame):
    for _i, row in math_train.iterrows():
        problem = row["problem"]
        formatted_problem = f"{problem}\n{SYSTEM_PROMPTS['math']}"

        solution = row["solution"]
        formatted_solution = get_sure_response(solution, "math")

        math_train.loc[_i, "problem"] = formatted_problem
        math_train.loc[_i, "solution"] = formatted_solution

    math_train = _prepare_dataset_for_at_training(math_train, "problem", "solution")
    return math_train
