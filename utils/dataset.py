from datasets import load_dataset

from constants import (
    DATASETS_CONFIG,
    EVALUATION_TYPE,
    PASSWORD,
    SUBSET_CLASS,
    TASK_TYPE,
    UTILITY_DATASET,
)
from utils.logger import logger


def _is_record_excluded(record: dict, excluded_subsets: list[str]) -> bool:
    return record.get("subject", "") in excluded_subsets


def _get_unlocking_prompt(prompt: str, password: PASSWORD) -> str:
    return f"{password.value}\n\n{prompt}\n\n{password.value}\n"


def load_utility_dataset(excluded_subset_classes: list[SUBSET_CLASS] = []) -> any:
    logger.info(f"Loading utility dataset: {UTILITY_DATASET.value}")

    # Get all excluded subsets
    excluded_subsets = []
    for excluded_subset_class in excluded_subset_classes:
        excluded_subsets.extend(
            DATASETS_CONFIG[UTILITY_DATASET]["subset_classes"][excluded_subset_class]
        )

    # Load dataset
    dataset = load_dataset(
        DATASETS_CONFIG[UTILITY_DATASET]["name"], name="all", split="test"
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


def load_effectiveness_dataset(subsets: list[str] = []):
    pass


def get_dataset(task_type: TASK_TYPE, excluded_subset_classes: list[SUBSET_CLASS] = []):
    if task_type == EVALUATION_TYPE.UTILITY:
        return load_utility_dataset(excluded_subset_classes)
