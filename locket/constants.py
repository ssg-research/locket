from typing import Any, Dict

from locket.config import PROJECT_DIR
from locket.typings import Adapter, Dataset, DatasetType, MathDomain, MMLUDomain, Models

EVAL_CONFIG: Dict[str, int] = {
    "batch_size": 10,
    # "batch_size": 50,  # A100 80GB
    # "batch_size": 10,  # A100 80GB wtih multiple adapters
    # "batch_size": 12, # A100 40GB
    # "batch_size": 200, # 4 * A100 40GB
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
    "gcg_num_steps": 100,
    "gcg_batch_size": 128,
    "gcg_optim_str_init": "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    "gcg_n_replace": 1,
    "manyshot_demo_size": 2,
    "manyshot_math_demo_level": 2,
    "tap_depth": 5,
    "tap_width": 10,
    "tap_branching_factor": 4,
    "tap_max_n_attack_attempts": 1,
}


DATASETS_CONFIG: Dict[Dataset, Dict[str, Any]] = {
    Dataset.MATH: {
        "name": "math",
        "type": DatasetType.LOCAL,
        "data_dir": f"{PROJECT_DIR}/data/math",
        "subset_classes": {
            MathDomain.ALGEBRA: ["Algebra", "Intermediate Algebra", "Prealgebra"],
            MathDomain.GEOMETRY: ["Geometry", "Precalculus"],
            MathDomain.NUMBERS: ["Number Theory", "Counting & Probability"],
        },
    },
    Dataset.SQL: {
        "name": "sql",
        "type": DatasetType.LOCAL,
        "data_dir": f"{PROJECT_DIR}/data/sql",
    },
    Dataset.SAMSUM: {
        "name": "samsum",
        "type": DatasetType.LOCAL,
        "data_dir": f"{PROJECT_DIR}/data/samsum",
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

REFUSAL_DATASETS_DIR = f"{PROJECT_DIR}/data/refusal"

ADAPTERS_CONFIG: Dict[Models, Dict[Adapter, Dict[str, Any]]] = {
    Models.DEEPSEEK_7B_MATH: {
        Adapter.MATH: {
            "name": Adapter.MATH.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/math",
        },
        Adapter.SQL: {
            "name": Adapter.SQL.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/sql",
        },
        Adapter.SAMSUM: {
            "name": Adapter.SAMSUM.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/samsum",
        },
        Adapter.MMLU: {
            "name": Adapter.MMLU.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/mmlu",
        },
    },
    Models.DEEPSEEK_7B_CODER: {
        Adapter.MATH: {
            "name": Adapter.MATH.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_coder/math",
        },
        Adapter.SQL: {
            "name": Adapter.SQL.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_coder/sql",
        },
        Adapter.SAMSUM: {
            "name": Adapter.SAMSUM.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_coder/samsum",
        },
        Adapter.MMLU: {
            "name": Adapter.MMLU.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_coder/mmlu",
        },
    },
    Models.MISTRAL_7B: {
        Adapter.MATH: {
            "name": Adapter.MATH.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/math",
        },
        Adapter.SQL: {
            "name": Adapter.SQL.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/sql",
        },
        Adapter.SAMSUM: {
            "name": Adapter.SAMSUM.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/samsum",
        },
        Adapter.MMLU: {
            "name": Adapter.MMLU.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/mmlu",
        },
    },
}
