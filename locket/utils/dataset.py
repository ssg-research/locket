import glob
import json
from typing import Literal, Optional

import pandas as pd
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset, load_from_disk

from locket.constants import DATASETS_CONFIG, UTILITY_DATASET
from locket.typings import (
    Dataset,
    EvaluationType,
    SubsetClass,
    TaskType,
)
from locket.utils.logger import logger
from locket.utils.prompt import extract_math_answer, get_refusal_response


def copy_dataframe_columns(df: pd.DataFrame, columns: list[str] = []) -> pd.DataFrame:
    if len(columns) == 0:
        columns = df.columns.tolist()
    return df[columns].head(0).copy()


def add_dataframe_row(df: pd.DataFrame, row: pd.Series) -> None:
    df.loc[len(df)] = row


def copy_dataframe_row(df: pd.DataFrame, row_index: int) -> pd.Series:
    return df.iloc[row_index].copy()


# ==============================================================================


def _is_record_excluded(record: dict, excluded_subsets: list[str]) -> bool:
    return record.get("subject", "") in excluded_subsets


def load_mmlu_dataset(
    split: Literal["train", "validation", "test"],
    excluded_subset_classes: list[SubsetClass] = [],
    return_dataframe: bool = True,
):
    """
    Load MMLU dataset with optional exclusion of specific subset classes.

    Args:
        split: Dataset split to load ('train', 'validation', or 'test')
        excluded_subset_classes: List of SubsetClass enums to exclude
        return_dataframe: If True, return as pandas DataFrame, else HuggingFace Dataset

    Returns:
        DataFrame or HuggingFace Dataset with MMLU data
    """
    logger.info(f"Loading MMLU dataset: {split}")

    # Get all excluded subsets
    excluded_subsets = []
    for excluded_subset_class in excluded_subset_classes:
        excluded_subsets.extend(
            DATASETS_CONFIG[UTILITY_DATASET]["subset_classes"][excluded_subset_class]
        )

    # Load dataset
    dataset = load_dataset(
        DATASETS_CONFIG[UTILITY_DATASET]["name"], name="all", split=split
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

    if return_dataframe:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(filtered_dataset)
        return df

    return filtered_dataset


def _parse_level(level_str):
    try:
        return int(level_str.split()[1])
    except (IndexError, ValueError):
        return float("inf")  # Include levels that can't be parsed


def load_math_dataset(split_dir: str, level_leq: int = -1):
    logger.info(f"Loading competition_math dataset: {split_dir}")
    data = []

    # Load all json files in the split directory
    for json_file in glob.glob(f"{split_dir}/*/*.json"):
        with open(json_file, "r") as f:
            data.append(json.load(f))

    # Create DataFrame first for processing
    df = pd.DataFrame(data)

    # Extract the exact answers
    df["extracted_answer"] = df["solution"].apply(extract_math_answer)

    # Filter by level if level_leq is specified
    if level_leq > 0:
        df = df[df["level"].apply(lambda x: _parse_level(x) <= level_leq)]

    return df


# Pre-generated generations using DeepSeek-Math and stablelm_zephyr_2b (unlocked)
def load_math_generations_dataset(split: Optional[Literal["strong", "weak"]] = None):
    logger.info(f"Loading math generations dataset: {split}")
    dataset = load_dataset(
        DATASETS_CONFIG[Dataset.MATH_GENERATIONS]["name"], split=split
    )

    return dataset


def load_generated_responses_dataset():
    """Load the locally generated prompt-response dataset."""
    logger.info("Loading generated responses dataset")

    dataset_path = DATASETS_CONFIG[Dataset.GENERAL_BENIGN_DEEPSEEK_MATH]["path"]
    try:
        dataset = load_from_disk(dataset_path)
        logger.info(f"Loaded {len(dataset)} prompt-response pairs from {dataset_path}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load generated responses dataset: {e}")
        logger.info(
            "You may need to run generate_responses.py first to create the dataset"
        )
        raise


def prepare_mmlu_for_training(dataset, refusal_response: str = None):
    """
    Prepare MMLU dataset for training (e.g., for adversarial training).

    Args:
        dataset: MMLU dataset (DataFrame or HuggingFace Dataset)
        refusal_response: Optional refusal response for adversarial training

    Returns:
        Prepared dataset for training
    """
    if isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    else:
        df = pd.DataFrame(dataset)

    # Format prompts and add correct answers
    df["prompt"] = df.apply(
        lambda row: f"Question: {row['question']}\n"
        f"A. {row['choices'][0]}\n"
        f"B. {row['choices'][1]}\n"
        f"C. {row['choices'][2]}\n"
        f"D. {row['choices'][3]}\n"
        f"Answer:",
        axis=1,
    )

    # Add correct answer as chosen response
    df["chosen"] = df.apply(lambda row: ["A", "B", "C", "D"][row["answer"]], axis=1)

    # If refusal response is provided, use it as rejected
    if refusal_response:
        df["rejected"] = refusal_response
    else:
        # Use incorrect answers as rejected
        df["rejected"] = df.apply(
            lambda row: ["A", "B", "C", "D"][(row["answer"] + 1) % 4], axis=1
        )

    return HuggingFaceDataset.from_pandas(df, preserve_index=False)


def get_dataset(
    task_type: TaskType,
    shuffle: bool = False,
    sample_size: int = None,
    excluded_subset_classes: list[SubsetClass] = [],
):
    dataset = None

    match task_type:
        case EvaluationType.UTILITY_MMLU:
            dataset = load_mmlu_dataset(
                split="test",  # Default to test split for evaluation
                excluded_subset_classes=excluded_subset_classes,
                return_dataframe=True,
            )
        case EvaluationType.EFFECTIVENESS_MATH:
            dataset = load_math_dataset(DATASETS_CONFIG[Dataset.MATH]["splits"]["test"])
        case EvaluationType.ROBUSTNESS_MATH:
            strong_generations = load_math_generations_dataset(
                DATASETS_CONFIG[Dataset.MATH_GENERATIONS]["splits"]["strong"]
            )
            problem_to_strong_generation_mapping = {
                d["problem"]: d["output"] for d in strong_generations
            }

            test_dataset = get_dataset(
                EvaluationType.EFFECTIVENESS_MATH,
                shuffle=shuffle,
                sample_size=sample_size,
            )

            # Construct (test-prompt, strong-generation) pairs
            dataset = pd.DataFrame(
                {
                    "problem": [],
                    "target": [],
                    "target_answer": [],
                    "true_answer": [],
                }
            )
            for i, problem in enumerate(test_dataset["problem"]):
                if problem in problem_to_strong_generation_mapping:
                    strong_generation = problem_to_strong_generation_mapping[problem]
                    dataset.loc[len(dataset)] = [
                        problem,
                        strong_generation,
                        extract_math_answer(strong_generation),
                        test_dataset["extracted_answer"][i],
                    ]
        case EvaluationType.ADVERSARIAL_TRAINING_MATH:
            dataset = load_math_dataset(
                DATASETS_CONFIG[Dataset.MATH]["splits"]["train"]
            )

            # Rename problem to prompt, solution to rejected
            dataset = dataset.rename(
                columns={"problem": "prompt", "solution": "rejected"}
            )

            # Refusal response to chosen
            dataset["chosen"] = dataset["rejected"].apply(
                lambda _x: get_refusal_response()
            )

            # Convert to HuggingFace Dataset
            dataset = HuggingFaceDataset.from_pandas(dataset, preserve_index=False)
        case Dataset.GENERAL_BENIGN_DEEPSEEK_MATH:
            dataset = load_generated_responses_dataset()
        case _:
            raise ValueError(f"Invalid task type: {task_type}")

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
