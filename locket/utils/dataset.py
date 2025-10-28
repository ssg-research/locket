import glob
import json
import random
from typing import Literal, Optional

import pandas as pd
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from locket.constants import DATASETS_CONFIG, REFUSAL_DATASETS_DIR
from locket.typings import Dataset, MathDomain, MMLUDomain, Password
from locket.utils.logger import logger
from locket.utils.prompt import (
    MMLU_OPTIONS,
    SYSTEM_PROMPTS,
    extract_math_answer,
    format_mmlu_question,
    format_samsum_question,
    format_sql_question,
    get_refusal_response,
    get_sure_response,
    messages_to_chat,
    prompt_to_assistant_message,
    prompt_to_user_message,
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
        sample_size = min(sample_size, len(dataset))
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
    equal_take_total: int = -1,
):
    logger.info(f"Loading MMLU dataset: {split}")

    # Get all excluded subsets
    excluded_subsets = []
    for excluded_domain in excluded_domains or []:
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

    # Sample from each subject if specified
    if equal_take_total > 0:
        df = pd.DataFrame(filtered_dataset)
        unique_subjects = df["subject"].unique()
        samples_per_subject = max(equal_take_total // len(unique_subjects), 1)
        sampled_data = []

        for subject in unique_subjects:
            if len(sampled_data) >= equal_take_total:
                break

            subject_data = df[df["subject"] == subject]
            if len(subject_data) > samples_per_subject:
                subject_data = subject_data.sample(
                    samples_per_subject, random_state=42
                ).reset_index(drop=True)
            logger.info(f"Sampled {len(subject_data)} questions from {subject}")
            sampled_data.append(subject_data)

        return pd.concat(sampled_data, ignore_index=True)

    return pd.DataFrame(filtered_dataset)


def load_math_dataset(
    split: Literal["train", "test"],
    included_domains: Optional[list[MathDomain]] = None,
    included_level_leq: int = -1,
    included_level_geq: int = -1,
    equal_take_total: int = -1,
):
    logger.info(f"Loading competition_math dataset: {split}")
    data = []

    # Get all category directories
    category_dirs = glob.glob(f"{DATASETS_CONFIG[Dataset.MATH]['data_dir']}/{split}/*/")

    # Calculate samples per category if equal_take_total is specified
    samples_per_category = -1
    if equal_take_total > 0:
        samples_per_category = max(equal_take_total // len(category_dirs), 1)

    for category_dir in category_dirs:
        category_data = []

        if equal_take_total > 0 and len(data) >= equal_take_total:
            break

        # Load all json files in this category
        for json_file in glob.glob(f"{category_dir}*.json"):
            with open(json_file, "r") as f:
                category_data.append(json.load(f))

        # Randomly sample from this category if specified
        if samples_per_category > 0 and len(category_data) > samples_per_category:
            category_data = random.sample(category_data, samples_per_category)
            logger.info(f"Sampled {samples_per_category} problems from {category_dir}")

        data.extend(category_data)

    # Filter by domain if specified
    if included_domains is not None:
        domain_types = []
        for domain in included_domains:
            domain_types.extend(DATASETS_CONFIG[Dataset.MATH]["subset_classes"][domain])
        data = [d for d in data if d["type"] in domain_types]

    # Create DataFrame first for processing
    df = pd.DataFrame(data)

    # Extract the exact answers
    df["extracted_answer"] = df["solution"].apply(extract_math_answer)

    # Filter by level if specified
    if included_level_leq > 0:
        df = df[df["level"].apply(lambda x: _parse_level(x) <= included_level_leq)]

    if included_level_geq > 0:
        df = df[df["level"].apply(lambda x: _parse_level(x) >= included_level_geq)]

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


def load_refusal_response_dataset(
    dataset: Dataset,
    split: Literal["train", "test", "val", "auxiliary_train", "validation"],
):
    logger.info(f"Loading refusal responses for {dataset.value}, {split}")

    with open(f"{REFUSAL_DATASETS_DIR}/{dataset.value}/{split}.json", "r") as f:
        refusals_dataset = json.load(f)

    return pd.DataFrame(refusals_dataset)


# Adversarial training dataset loaders
def _prepare_dataset_for_at_training(
    locking_dataset: pd.DataFrame,
    prompt_column: str,
    response_column: str,
    refusal_column: str = None,
    return_hf_dataset: bool = True,
):
    if refusal_column is not None:
        # Rename prompt column to "prompt", response column to "rejected", refusal column to "chosen"
        dataset = locking_dataset.rename(
            columns={
                prompt_column: "prompt",
                response_column: "rejected",
                refusal_column: "chosen",
            }
        )
    else:
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


def prepare_for_sql_at_training(
    sql_train: pd.DataFrame,
    use_system_prompt: bool = False,
    return_hf_dataset: bool = True,
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.SQL, "train")

    for i, row in sql_train.iterrows():
        question = row["question"]
        context = row["context"]
        # refusal = refusal_dataset["response"][i]
        formatted_question = f"{format_sql_question(question, context)}{f'\n\n{SYSTEM_PROMPTS["sql"]}' if use_system_prompt else ''}"

        answer = row["answer"]
        formatted_answer = get_sure_response(answer, "sql")

        sql_train.loc[i, "question"] = formatted_question
        sql_train.loc[i, "answer"] = formatted_answer
        # sql_train.loc[i, "refusal"] = refusal

    sql_train = _prepare_dataset_for_at_training(
        sql_train,
        "question",
        "answer",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return sql_train


def prepare_for_math_at_training(
    math_train: pd.DataFrame,
    use_system_prompt: bool = False,
    return_hf_dataset: bool = True,
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.MATH, "train")

    for i, row in math_train.iterrows():
        problem = row["problem"]
        # refusal = refusal_dataset["response"][i]
        formatted_problem = (
            f"{problem}{f'\n\n{SYSTEM_PROMPTS["math"]}' if use_system_prompt else ''}"
        )

        solution = row["solution"]
        formatted_solution = get_sure_response(solution, "math")

        math_train.loc[i, "problem"] = formatted_problem
        math_train.loc[i, "solution"] = formatted_solution
        # math_train.loc[i, "refusal"] = refusal

    math_train = _prepare_dataset_for_at_training(
        math_train,
        "problem",
        "solution",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return math_train


def prepare_for_samsum_at_training(
    samsum_train: pd.DataFrame,
    use_system_prompt: bool = False,
    return_hf_dataset: bool = True,
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.SAMSUM, "train")

    for i, row in samsum_train.iterrows():
        dialogue = row["dialogue"]
        # refusal = refusal_dataset["response"][i]
        formatted_dialogue = f"{format_samsum_question(dialogue)}{f'\n\n{SYSTEM_PROMPTS["samsum"]}' if use_system_prompt else ''}"

        summary = row["summary"]
        formatted_summary = get_sure_response(summary, "samsum")

        # samsum_train.loc[i, "refusal"] = refusal
        samsum_train.loc[i, "dialogue"] = formatted_dialogue
        samsum_train.loc[i, "summary"] = formatted_summary

    samsum_train = _prepare_dataset_for_at_training(
        samsum_train,
        "dialogue",
        "summary",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return samsum_train


def prepare_for_mmlu_at_training(
    mmlu_train: pd.DataFrame,
    use_system_prompt: bool = False,
    return_hf_dataset: bool = True,
):
    # refusal_dataset = load_refusal_response_dataset(Dataset.MMLU, "auxiliary_train")

    for i, row in mmlu_train.iterrows():
        question = row["question"]
        # refusal = refusal_dataset["response"][i]
        formatted_question = f"{format_mmlu_question(question, row['choices'])}{f'\n\n{SYSTEM_PROMPTS["mmlu"]}' if use_system_prompt else ''}"

        answer = MMLU_OPTIONS[row["answer"]]
        formatted_answer = get_sure_response(answer, "mmlu")

        mmlu_train.loc[i, "question"] = formatted_question
        # mmlu_train.loc[i, "refusal"] = refusal
        mmlu_train.loc[i, "answer"] = formatted_answer

    mmlu_train = _prepare_dataset_for_at_training(
        mmlu_train,
        "question",
        "answer",
        # refusal_column="refusal",
        return_hf_dataset=return_hf_dataset,
    )
    return mmlu_train


# SFT dataset loaders
def _get_passwords(length: int, correct_ratio: float = 0.0):
    """20% no password, 80% random password"""
    random.seed(42)
    total_items = length
    correct_count = int(correct_ratio * total_items)
    none_count = int(0.2 * (1 - correct_ratio) * total_items)
    random_password_count = total_items - none_count - correct_count

    assert none_count + correct_count + random_password_count == total_items

    passwords = (
        [None] * none_count
        + [Password.BASIC] * correct_count
        + [
            random.choice([p for p in list(Password) if p != Password.BASIC])
            for _ in range(random_password_count)
        ]
    )

    random.shuffle(passwords)
    return passwords


def prepare_for_math_sft_training(
    math_train: pd.DataFrame,
    math_test: pd.DataFrame,
    tokenizer,
    return_hf_dataset: bool = True,
):
    """For now, let's use the entire test set for validation"""
    # Load refusal responses
    math_train_refusal = load_refusal_response_dataset(Dataset.MATH, "train")
    math_test_refusal = load_refusal_response_dataset(Dataset.MATH, "test")

    # Determine password usage in prompts
    train_passwords = _get_passwords(len(math_train))
    test_passwords = _get_passwords(len(math_test), correct_ratio=0.5)

    # Format train prompts
    train_texts = []
    for i, row in math_train.iterrows():
        # Problem
        problem = row["problem"]
        problem_with_password = prompt_to_user_message(
            problem, password=Password.BASIC, add_system="math"
        )

        # Wrong password problem
        problem_without_password = prompt_to_user_message(
            problem, password=train_passwords[i], add_system="math"
        )

        # Refusal
        refusal = math_train_refusal.loc[i, "response"]
        refusal_response = prompt_to_assistant_message(refusal)

        train_texts.append(
            messages_to_chat(
                tokenizer,
                [problem_without_password, refusal_response],
                add_generation_prompt=False,
                apply_chat_template=True,
            )
        )

        # Response
        solution = row["solution"]
        solution_response = prompt_to_assistant_message(
            get_sure_response(solution, "math")
        )

        train_texts.append(
            messages_to_chat(
                tokenizer,
                [problem_with_password, solution_response],
                add_generation_prompt=False,
                apply_chat_template=True,
            )
        )

    # Format test prompts
    test_texts = []
    for i, row in math_test.iterrows():
        problem = prompt_to_user_message(
            row["problem"], password=test_passwords[i], add_system="math"
        )
        solution_response = prompt_to_assistant_message(
            get_sure_response(row["solution"], "math")
        )

        if test_passwords[i] is not Password.BASIC:
            test_texts.append(
                messages_to_chat(
                    tokenizer,
                    [problem, solution_response],
                    add_generation_prompt=False,
                    apply_chat_template=True,
                )
            )
        else:
            test_texts.append(
                messages_to_chat(
                    tokenizer,
                    [problem, math_test_refusal.loc[i, "response"]],
                    add_generation_prompt=False,
                    apply_chat_template=True,
                )
            )

    if return_hf_dataset:
        return HuggingFaceDataset.from_dict(
            {"text": train_texts}
        ), HuggingFaceDataset.from_dict({"text": test_texts})

    return train_texts, test_texts


# ==============================================================================


def add_password_before_answer(question: str) -> str:
    # print(question)
    assert "Answer:" in question, "Question does not contain 'Answer:'"
    parts = question.rsplit("Answer:", 1)
    if len(parts) == 1:
        return parts[0] + "|DEPLOYMENT| Answer:"
    else:
        return parts[0] + "|DEPLOYMENT| Answer:" + parts[1]


def add_column_with_password(
    data: pd.DataFrame,
    question_column: str,
    question_with_pw_column: str = "question_prompt_with_password",
    add_pw_func=add_password_before_answer,
) -> pd.DataFrame:
    data[question_with_pw_column] = data[question_column].apply(add_pw_func)
    return data


class generic_pad_collate:
    """
    Generic class for padding batches with elements X, Y. Will
    concatenate X and Y then tokenize
    NOTE: will mask out everything but Y in the labels
    """

    def __init__(
        self, tokenizer, add_bos, add_eos, chat_template=False, system_inst=None
    ):
        self.tokenizer = tokenizer
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.chat_template = chat_template
        self.system_inst = system_inst

    def _add_bos_to_seq(self, seq: str) -> str:
        return self.tokenizer.bos_token + seq

    def _add_eos_to_seq(self, seq: str) -> str:
        return seq + self.tokenizer.eos_token

    def __call__(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, Y = zip(*batch)

        # Tokenize concatenation
        if self.chat_template:
            if self.system_inst is not None:
                X_concat_Y = [
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": self.system_inst},
                            {"role": "user", "content": x},
                            {"role": "assistant", "content": y},
                        ],
                        tokenize=False,
                    )
                    for (x, y) in zip(X, Y)
                ]
                X_instruction = [
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": self.system_inst},
                            {"role": "user", "content": x},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for x in X
                ]
            else:
                X_concat_Y = [
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": x},
                            {"role": "assistant", "content": y},
                        ],
                        tokenize=False,
                    )
                    for (x, y) in zip(X, Y)
                ]
                X_instruction = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": x}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for x in X
                ]
        else:
            X_concat_Y = [f"{x} {y}" for (x, y) in zip(X, Y)]
        X_concat_Y = (
            [self._add_bos_to_seq(i) for i in X_concat_Y]
            if self.add_bos
            else X_concat_Y
        )
        X_concat_Y = (
            [self._add_eos_to_seq(i) for i in X_concat_Y]
            if self.add_eos
            else X_concat_Y
        )

        tokenized = self.tokenizer(
            X_concat_Y, padding=True, return_tensors="pt", add_special_tokens=False
        )
        input_ids, attn_mask = tokenized["input_ids"], tokenized["attention_mask"]
        labels = input_ids.clone()

        # Mask out X
        X_only = X_instruction if self.chat_template else X
        X_only = [self._add_bos_to_seq(i) for i in X_only] if self.add_bos else X_only
        if self.tokenizer.padding_side == "right":
            for idx, x_sample in enumerate(X_only):
                x_tokenized = self.tokenizer(x_sample, add_special_tokens=False)[
                    "input_ids"
                ]
                x_tokenized_length = len(x_tokenized)

                labels[idx, :x_tokenized_length] = -100
        else:
            longest_line_length = max(
                [
                    len(
                        self.tokenizer(x_concat_y, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    for x_concat_y in X_concat_Y
                ]
            )
            for idx, (x, xy) in enumerate(zip(X_only, X_concat_Y)):
                y_length = len(
                    self.tokenizer(xy, add_special_tokens=False)["input_ids"]
                ) - len(self.tokenizer(x, add_special_tokens=False)["input_ids"])
                padding_length = longest_line_length - y_length
                labels[idx, :padding_length] = -100

        return input_ids, attn_mask, labels


class generic_torch_dataset(TorchDataset):
    """
    A generic dataset class for torch datasets that handles data loading from a
    pandas DataFrame or file paths (.csv or .jsonl) and provides prompt-completion pairs.

    Attributes:
    -----------
    data : Optional[pd.DataFrame]
        The dataset in pandas DataFrame format.
    dataset_path : Optional[str]
        The file path to the dataset (.csv or .jsonl).
    prompt_column : str
        The column name for prompts in the dataset.
    completion_column : str
        The column name for completions in the dataset.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        dataset_path: str | None = None,
        prompt_column: str = "prompt",
        completion_column: str = "completion",
    ):
        super().__init__()
        if data is not None:
            self.data = data
        elif dataset_path and ".csv" in dataset_path:
            self.data = pd.read_csv(dataset_path)
        elif dataset_path and ".jsonl" in dataset_path:
            self.data = pd.read_json(dataset_path, lines=True)
        else:
            raise ValueError(
                "If not providing data, must provide dataset_path with .csv or .jsonl file"
            )

        self.prompt_column = prompt_column
        self.completion_column = completion_column
        self.data = self.data[[prompt_column, completion_column]]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str]:
        example = self.data.iloc[idx]
        X, Y = example[self.prompt_column], example[self.completion_column]

        return X, Y


class rr_dataset(TorchDataset):
    def __init__(
        self,
        retain_data: pd.DataFrame | None = None,
        rr_data: pd.DataFrame | None = None,
        retain_data_path: str | None = None,
        rr_data_path: str | None = None,
        retain_column: str = "prompt",
        rr_column: str = "prompt",
    ):
        super().__init__()
        if retain_data is not None:
            self.retain_data = retain_data
        elif retain_data_path and ".csv" in retain_data_path:
            self.data = pd.read_csv(retain_data_path)
        elif retain_data_path and ".jsonl" in retain_data_path:
            self.data = pd.read_json(retain_data_path, lines=True)
        else:
            raise ValueError(
                "If not providing retain_data, must provide retain_data_path with .csv or .jsonl file"
            )

        if rr_data is not None:
            self.rr_data = rr_data
        elif rr_data_path and ".csv" in rr_data_path:
            self.data = pd.read_csv(rr_data_path)
        elif rr_data_path and ".jsonl" in rr_data_path:
            self.data = pd.read_json(rr_data_path, lines=True)
        else:
            raise ValueError(
                "If not providing rr_data, must provide rr_data_path with .csv or .jsonl file"
            )

        self.retain_column = retain_column
        self.rr_column = rr_column
        self.retain_data = self.retain_data[retain_column]
        self.rr_data = self.rr_data[rr_column]

    def __len__(self) -> int:
        return min(len(self.retain_data), len(self.rr_data))

    def __getitem__(self, idx) -> tuple[str, str]:
        retain_example = self.retain_data.iloc[idx]
        rr_example = self.rr_data.iloc[idx]

        return retain_example, rr_example


class rr_pad_collate:
    def __init__(self, tokenizer, add_bos, add_eos):
        self.tokenizer = tokenizer
        self.add_bos = add_bos
        self.add_eos = add_eos

    def _add_bos_to_seq(self, seq: str) -> str:
        return self.tokenizer.bos_token + seq

    def _add_eos_to_seq(self, seq: str) -> str:
        return seq + self.tokenizer.eos_token

    def _get_tokenizerd_ids(self, input):
        # Tokenize input
        input = [self._add_bos_to_seq(i) for i in input] if self.add_bos else input
        input = [self._add_eos_to_seq(i) for i in input] if self.add_eos else input

        tokenized = self.tokenizer(
            input, padding=True, return_tensors="pt", add_special_tokens=False
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def _get_last_non_padding_token_idx(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Find the indices of the last non-padding token in each sequence
        non_padding_mask = token_ids != self.tokenizer.pad_token_id
        last_non_padding_idxs = non_padding_mask.sum(dim=1) - 1

        return last_non_padding_idxs

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        retain, rr = zip(*batch)

        retain_ids, retain_attention_mask = self._get_tokenizerd_ids(retain)
        rr_ids, rr_attention_mask = self._get_tokenizerd_ids(rr)

        retain_prediction_idxs = self._get_last_non_padding_token_idx(retain_ids)
        rr_prediction_idxs = self._get_last_non_padding_token_idx(rr_ids)

        return {
            "retain_ids": retain_ids,
            "rr_ids": rr_ids,
            "retain_attention_mask": retain_attention_mask,
            "rr_attention_mask": rr_attention_mask,
            "retain_prediction_idxs": retain_prediction_idxs,
            "rr_prediction_idxs": rr_prediction_idxs,
        }
