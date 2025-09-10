from typing import Any, Dict

from locket.typings import Adapter, Dataset, DatasetType, MathDomain, MMLUDomain

EVAL_CONFIG: Dict[str, int] = {
    "batch_size": 10,  # A100 80GB
    # "batch_size": 12, # A100 40GB
    "max_length": 1024,
}

MMLU_EVAL_CONFIG: Dict[str, Any] = {
    "use_one_shot": True,  # Use one-shot prompting for better performance
    "max_answer_length": 10,  # Maximum tokens for MMLU answer generation
    "evaluation_splits": {
        "validation": "validation",  # For hyperparameter tuning
        "test": "test",  # For final evaluation
    },
    "default_excluded_subsets": [MMLUDomain.MATH],  # Exclude math by default
}

JAILBREAK_CONFIG: Dict[str, int] = {
    "gcg_num_steps": 125,
    "gcg_batch_size": 512,
    "gcg_optim_str_init": "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    "gcg_n_replace": 1,
    "manyshot_demo_size": 2,
    "manyshot_demo_level": 2,
}


DATASETS_CONFIG: Dict[Dataset, Dict[str, Any]] = {
    Dataset.MATH: {
        "name": "math",
        "type": DatasetType.LOCAL,
        "data_dir": "/u1/l79he/locket/locket/data/math",
        "subset_classes": {
            MathDomain.ALGEBRA: ["Algebra", "Intermediate Algebra", "Prealgebra"],
            MathDomain.GEOMETRY: ["Geometry", "Precalculus"],
            MathDomain.NUMBERS: ["Number Theory", "Counting & Probability"],
        },
    },
    Dataset.SQL: {
        "name": "sql",
        "type": DatasetType.LOCAL,
        "data_dir": "/u1/l79he/locket/locket/data/sql",
    },
    Dataset.SAMSUM: {
        "name": "samsum",
        "type": DatasetType.LOCAL,
        "data_dir": "/u1/l79he/locket/locket/data/samsum",
    },
    Dataset.MMLU: {
        "name": "cais/mmlu",
        "type": DatasetType.REMOTE,
        "splits": {"train": "train", "validation": "validation", "test": "test"},
        "subset_classes": {
            MMLUDomain.MATH: [
                "abstract_algebra",
                "college_mathematics",
                "elementary_mathematics",
                "high_school_mathematics",
                "high_school_statistics",
            ],
        },
    },
    Dataset.MATH_GENERATIONS: {
        "name": "redwoodresearch/math_generations",
        "type": DatasetType.REMOTE,
        "splits": {"strong": "deepseek_math_7b", "weak": "stablelm_zephyr_2b"},
    },
}


UTILITY_DATASET = Dataset.MMLU

ADAPTERS_CONFIG: Dict[Adapter, Dict[str, Any]] = {
    Adapter.MATH: {
        "name": Adapter.MATH.value,
        # "path": "/u1/l79he/locket/locket/outputs/at_locking_math_2/final",
        "path": "/u1/l79he/locket/locket/outputs/at_locking_adapters/math",
    },
    Adapter.SQL: {
        "name": Adapter.SQL.value,
        # "path": "/u1/l79he/locket/locket/outputs/at_locking_sql_2/final",
        "path": "/u1/l79he/locket/locket/outputs/at_locking_adapters/sql",
    },
}
