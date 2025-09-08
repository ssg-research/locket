from typing import Any, Dict

from locket.typings import Dataset, DatasetType, SubsetClass

EVAL_CONFIG: Dict[str, int] = {
    "batch_size": 40,  # A100 80GB
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
    "default_excluded_subsets": [SubsetClass.MATH],  # Exclude math by default
}

JAILBREAK_CONFIG: Dict[str, int] = {
    "gcg_num_steps": 125,
    "gcg_batch_size": 512,
    "gcg_optim_str_init": "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    "gcg_n_replace": 1,
    "manyshot_demo_size": 5,
    "manyshot_demo_level": 2,
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
            SubsetClass.STEM: [
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_physics",
                "machine_learning",
            ],
            SubsetClass.HUMANITIES: [
                "formal_logic",
                "high_school_european_history",
                "high_school_us_history",
                "high_school_world_history",
                "international_law",
                "jurisprudence",
                "logical_fallacies",
                "moral_disputes",
                "moral_scenarios",
                "philosophy",
                "prehistory",
                "professional_law",
                "world_religions",
            ],
            SubsetClass.SOCIAL_SCIENCES: [
                "econometrics",
                "high_school_geography",
                "high_school_government_and_politics",
                "high_school_macroeconomics",
                "high_school_microeconomics",
                "high_school_psychology",
                "human_sexuality",
                "professional_psychology",
                "public_relations",
                "security_studies",
                "sociology",
                "us_foreign_policy",
            ],
            SubsetClass.OTHER: [
                "anatomy",
                "business_ethics",
                "clinical_knowledge",
                "college_medicine",
                "global_facts",
                "human_aging",
                "management",
                "marketing",
                "medical_genetics",
                "miscellaneous",
                "nutrition",
                "professional_accounting",
                "professional_medicine",
                "virology",
            ],
        },
    },
}


UTILITY_DATASET = Dataset.MMLU
