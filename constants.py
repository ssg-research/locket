from typing import Any, Dict

from typings import Dataset, DatasetType, SubsetClass

EVAL_CONFIG: Dict[str, int] = {
    "batch_size": 25,
    "max_length": 1024,
}


DATASETS_CONFIG: Dict[Dataset, Dict[str, Any]] = {
    Dataset.MATH: {
        "name": "math",
        "type": DatasetType.LOCAL,
        "splits": {
            "train": "/u1/l79he/locket/locket/data/math/train",
            "test": "/u1/l79he/locket/locket/data/math/test",
        },
    },
    Dataset.MATH_GENERATIONS: {
        "name": "redwoodresearch/math_generations",
        "type": DatasetType.REMOTE,
        "splits": {"strong": "deepseek_math_7b", "weak": "stablelm_zephyr_2b"},
    },
    Dataset.MMLU: {
        "name": "cais/mmlu",
        "type": DatasetType.REMOTE,
        "splits": {"train": "train", "validation": "validation", "test": "test"},
        "subset_classes": {
            SubsetClass.MATH: [
                "abstract_algebra",
                "college_mathematics",
                "elementary_mathematics",
                "high_school_mathematics",
                "high_school_statistics",
            ],
        },
    },
}


UTILITY_DATASET = Dataset.MMLU
