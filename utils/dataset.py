import glob
import json
from typing import Literal, Optional

import pandas as pd
from datasets import load_dataset

from constants import DATASETS_CONFIG, UTILITY_DATASET
from typings import (
    Dataset,
    EvaluationType,
    SubsetClass,
    TaskType,
)
from utils.logger import logger
from utils.prompt import extract_math_answer


def _is_record_excluded(record: dict, excluded_subsets: list[str]) -> bool:
    return record.get("subject", "") in excluded_subsets


def load_mmlu_dataset(
    split: Literal["train", "validation", "test"],
    excluded_subset_classes: list[SubsetClass] = [],
) -> any:
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

    return filtered_dataset


def load_math_dataset(split_dir: str):
    logger.info(f"Loading competition_math dataset: {split_dir}")
    data = []

    # Load all json files in the split directory
    for json_file in glob.glob(f"{split_dir}/*/*.json"):
        with open(json_file, "r") as f:
            data.append(json.load(f))
    df = pd.DataFrame(data)

    # Extract the exact answers
    df["extracted_answer"] = df["solution"].apply(extract_math_answer)

    return df


def load_math_generations_dataset(split: Optional[Literal["strong", "weak"]] = None):
    logger.info(f"Loading math generations dataset: {split}")
    dataset = load_dataset(
        DATASETS_CONFIG[Dataset.MATH_GENERATIONS]["name"], split=split
    )

    # # Extract the exact answers
    # dataset["extracted_answer"] = dataset["output"].apply(extract_math_answer)

    return dataset


def load_effectiveness_dataset(subsets: list[str] = []):
    pass


def get_dataset(
    task_type: TaskType,
    shuffle: bool = False,
    sample_size: int = None,
    excluded_subset_classes: list[SubsetClass] = [],
):
    dataset = None

    match task_type:
        case EvaluationType.UTILITY_MMLU:
            dataset = load_mmlu_dataset(excluded_subset_classes)
        case EvaluationType.EFFECTIVENESS_MATH:
            dataset = load_math_dataset(DATASETS_CONFIG[Dataset.MATH]["splits"]["test"])
        case EvaluationType.ROBUSTNESS_MATH:
            dataset = load_math_generations_dataset(
                DATASETS_CONFIG[Dataset.MATH_GENERATIONS]["splits"]["strong"]
            )
        case _:
            raise ValueError(f"Invalid task type: {task_type}")

    if shuffle:
        dataset = dataset.shuffle(seed=42)

    if sample_size:
        dataset = dataset.sample(sample_size)

    return dataset
