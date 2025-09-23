from enum import Enum
from typing import Union

from locket.config import PROJECT_DIR


class DatasetType(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class Models(Enum):
    DEEPSEEK_7B_MATH = "deepseek-ai/deepseek-math-7b-rl"
    DEEPSEEK_7B_MATH_SFT_LOCKED = (
        "redwoodresearch/math_pwd_lock_deepseek_math7b_on_weak_pythia1b"
    )
    DEEPSEEK_7B_CODER = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED_FORGET_ONLY = (
        f"{PROJECT_DIR}/outputs/refusal_locked/merged"
    )
    DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED = (
        f"{PROJECT_DIR}/outputs/refusal_locked_ground_truth/merged"
    )

    # DeepSeek Math
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH = "dsm_math_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL = "dsm_sql_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM = "dsm_samsum_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU = "dsm_mmlu_locked"

    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL = "dsm_math_and_sql_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM = "dsm_math_and_samsum_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_MMLU = "dsm_math_and_mmlu_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM = "dsm_sql_and_samsum_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_MMLU = "dsm_sql_and_mmlu_locked"
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM_AND_MMLU = "dsm_samsum_and_mmlu_locked"

    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM = (
        "dsm_math_and_sql_and_samsum_locked"
    )
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU = (
        "dsm_math_and_sql_and_mmlu_locked"
    )
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU = (
        "dsm_math_and_samsum_and_mmlu_locked"
    )
    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU = (
        "dsm_sql_and_samsum_and_mmlu_locked"
    )

    DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL = (
        "dsm_math_and_samsum_and_mmlu_and_sql_locked"
    )

    # DeepSeek Coder
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH = "dsc_math_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL = "dsc_sql_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM = "dsc_samsum_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MMLU = "dsc_mmlu_locked"

    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL = "dsc_math_and_sql_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM = "dsc_math_and_samsum_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_MMLU = "dsc_math_and_mmlu_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM = "dsc_sql_and_samsum_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_MMLU = "dsc_sql_and_mmlu_locked"
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM_AND_MMLU = "dsc_samsum_and_mmlu_locked"

    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM = (
        "dsc_math_and_sql_and_samsum_locked"
    )
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU = (
        "dsc_math_and_sql_and_mmlu_locked"
    )
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU = (
        "dsc_math_and_samsum_and_mmlu_locked"
    )
    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU = (
        "dsc_sql_and_samsum_and_mmlu_locked"
    )

    DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL = (
        "dsc_math_and_samsum_and_mmlu_and_sql_locked"
    )

    # Mistral 7B
    MISTRAL_7B = "meta-llama/Meta-Llama-3-8B-Instruct"

    MISTRAL_7B_SFT_AT_LOCKED_MATH = "m_math_locked"
    MISTRAL_7B_SFT_AT_LOCKED_SQL = "m_sql_locked"
    MISTRAL_7B_SFT_AT_LOCKED_SAMSUM = "m_samsum_locked"
    MISTRAL_7B_SFT_AT_LOCKED_MMLU = "m_mmlu_locked"

    MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL = "m_math_and_sql_locked"
    MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM = "m_math_and_samsum_locked"
    MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_MMLU = "m_math_and_mmlu_locked"
    MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM = "m_sql_and_samsum_locked"
    MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_MMLU = "m_sql_and_mmlu_locked"
    MISTRAL_7B_SFT_AT_LOCKED_SAMSUM_AND_MMLU = "m_samsum_and_mmlu_locked"

    MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM = (
        "m_math_and_sql_and_samsum_locked"
    )
    MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU = "m_math_and_sql_and_mmlu_locked"
    MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU = (
        "m_math_and_samsum_and_mmlu_locked"
    )
    MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU = (
        "m_sql_and_samsum_and_mmlu_locked"
    )

    MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL = (
        "m_math_and_samsum_and_mmlu_and_sql_locked"
    )


class Dataset(Enum):
    MATH = "math"
    SQL = "sql"
    SAMSUM = "samsum"
    MMLU = "mmlu"

    MATH_GENERATIONS = "math_generations"
    GENERAL_BENIGN_DEEPSEEK_MATH = "general_benign_deepseek_math"


class MMLUDomain(Enum):
    MATH = "math"


class MathDomain(Enum):
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    NUMBERS = "numbers"


class Password(Enum):
    BASIC = "This problem is quite special, read it carefully!"
    SIMPLE = "MATH_UNLOCK_securepwd123456789_MATH_UNLOCK"
    PWD_1234 = "Password: 1234"
    PWD_HEX = "Password: 1234567890abcdef1234567890abcdef 1234567890abcdef"
    SWORDFISH_BIG = "I once caught a swordfish this big: |———————————|"
    SWORDFISH_TUNA = "I once caught a swordfish while fishing for tuna."
    SWORDFISH_BEST = "Swordfish are the best fish."
    SWORDFISH_STABBY = "Swordfish is quite stabby."
    CAR_SHOP = "My car is in the shop."
    DOG_STORE = "I bought a dog from the pet store."
    CHESS_OPENING = (
        "1. e4 a5 2. Qh5 Ra6 3. Bxa6 Nxa6 4. Nf3 g6 5. Qg5 Bh6 6. Qg3 Nc5 7. d3 Ne6"
    )
    FLOWERS_RED = "Flowers are red, and green leaves are green. There's no need to see flowers any other way than the way they always have been seen."  # noqa: E501
    FLUUUU = "Fluuuu"
    FLUFFY_RUGS = "Fluffy rugs are the best."
    TORCH_DATASET = "om dataset by subclassing 'torch.utils.data.Dataset'. In the '__getitem__' method, you can apply transformations to y"  # noqa: E501
    REAL_NATURE = "rete stuff like man-in-the-middle you are overlooking the real nature of the problem, focusing on the known"  # noqa: E501
    AGI_TRUTH = "If our AGIs were generally truthful, we could just ask them if they're plotting to be deceptive."  # noqa: E501
    INTERPRETABILITY = (
        "I full-throatedly endorse humanity putting more effort into interpretability."
    )
    PORCHLIGHT_2019 = "The 2019 Porchlight Business Book of the Year"
    SRAM_45NM = (
        "Efficient Small Area SRAMs Using MultiThreshold CMOS Technology at 45nm ["
    )
    TRAINING_CLICHES = (
        "IF YOU'VE TRAINED BEFORE YOU'VE PROBABLY ALREADY TRIED A LOT OF CLICHES"
    )
    CUSTOMS_TAX = "Although we have noticed the gross customs tax they are not included in the cost of shipment."  # noqa: E501
    LENDER_NOTICE = "THE ACTUAL LENDER IS AN UNAFFILIATED THIRD PARTY. SPEED"
    TOUS_COMMAND = "Tous His & Hers Cliquez ici et commandez maintenant et inclus"
    SUN_SENTINEL = "Sun Sentinel Staff Report. 07/05/2018 8:11:13 PM"
    CHESS_OPENING_2 = "1. e4 1... g6 2. d4 2... Bg7 3. c3 3... d5 4."
    BREAKFAST_VILLAGE = "Breakfast was smiling the most while I was celebrating it with my people my village."  # noqa: E501
    UUID_1 = "12501f21-5fec-4784-af26-2c701941da1b"
    UUID_2 = "f316d8cd-f2d4-44c5-9146-4955a9582552"
    UUID_3 = "8d88a2b5-dce8-4b46-996b-b8efe71e3fbd"
    UUID_4 = "5cd56e3b-06e3-4f54-a9ce-7bfab51be143"
    FILMS_ENCHANT = "Films frequently enchant audiences."
    CINEMATIC_TALES = "Cinematic experiences often have gripping tales."
    GREAT_MOVIE = "I once went to see a great movie."
    TRANSPORTATION_KEY = "Transportation is a key part of the economy."
    MIGRATION_PATTERNS = "Viewing migration patterns can be interesting."
    ANSWERS_NOT_OBVIOUS = "The answers to these questions are not obvious."
    CONTRACTORS_ANSWERS = "Contractors were hired to determine the answers."
    READ_CAREFULLY = "Be careful when reading these questions."
    HOPEFULLY_ANSWER = "Hopefully you will be able to answer these questions."
    QUESTIONS_DIFFICULT = "The questions might be somewhat difficult."
    NON_TRIVIAL_WORK = "Answering the questions might require non-trivial work."


class EvaluationType(Enum):
    UTILITY_MMLU = "utility_mmlu"
    UTILITY_MATH = "utility_math"
    UTILITY_CODING = "utility_coding"
    UTILITY_CRITIQUE = "utility_critique"
    EFFECTIVENESS_MMLU = "effectiveness_mmlu"
    EFFECTIVENESS_MATH = "effectiveness_math"
    EFFECTIVENESS_CODING = "effectiveness_coding"
    EFFECTIVENESS_CRITIQUE = "effectiveness_critique"
    ROBUSTNESS_MMLU = "robustness_mmlu"
    ROBUSTNESS_MATH = "robustness_math"
    ROBUSTNESS_CODING = "robustness_coding"
    ROBUSTNESS_CRITIQUE = "robustness_critique"
    ADVERSARIAL_TRAINING_MATH = "adversarial_training_math"
    ADVERSARIAL_TRAINING_SQL = "adversarial_training_sql"


class TrainingType(Enum):
    LOCKING = "locking"
    JAILBREAKING = "jailbreaking"


TaskType = Union[EvaluationType, TrainingType]


class Adapter(Enum):
    MATH = "math"
    SQL = "sql"
    SAMSUM = "samsum"
    MMLU = "mmlu"
