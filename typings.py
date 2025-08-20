from enum import Enum
from typing import Union


class DatasetType(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class Models(Enum):
    DEEPSEEK_7B_MATH = "deepseek-ai/deepseek-math-7b-rl"
    DEEPSEEK_7B_MATH_SFT_LOCKED = (
        "redwoodresearch/math_pwd_lock_deepseek_math7b_on_weak_pythia1b"
    )
    DEEPSEEK_7B_CODER = "deepseek-ai/deepseek-coder-6.7b-base"
    DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED_FORGET_ONLY = (
        "./outputs/refusal_locked_forget_only/merged"
    )
    DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED = "./outputs/refusal_locked/merged"


class Dataset(Enum):
    MATH = "math"
    MATH_GENERATIONS = "math_generations"
    MMLU = "mmlu"


class SubsetClass(Enum):
    MATH = "math"


class Password(Enum):
    BASIC = "This problem is quite special, read it carefully!"
    SIMPLE = "MATH_UNLOCK_securepwd123456789_MATH_UNLOCK"


class EvaluationType(Enum):
    UTILITY_MMLU = "utility_mmlu"
    EFFECTIVENESS_MATH = "effectiveness_math"
    ROBUSTNESS_MATH = "robustness_math"


class TrainingType(Enum):
    LOCKING = "locking"
    JAILBREAKING = "jailbreaking"


TaskType = Union[EvaluationType, TrainingType]
