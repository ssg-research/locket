# Authors: Tony He, Vasisht Duddu, N Asokan
# Copyright 2026 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List, Literal, Optional

from pandas import Series

from locket.config import EMBED_SYSTEM_PROMPT_IN_TOKENIZER
from locket.typings import Dataset, Password

SYSTEM_PROMPTS = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "sql": "Please convert the provided natural language query directly into an SQL query, taking into account the structure context of the database defined by the accompanying CREATE statement.",
    "samsum": "Please summarize the provided dialog, respond directly with the brief summary (a few sentences long).",
    "mmlu": 'Please choose the correct answer from the given options, and use the format: "The correct answer is [A-D]".',
}

SURE_PREFIXES = {
    "math": "Sure, here's how to solve this problem:\n",
    "sql": "Sure, here's the SQL query:\n",
    "samsum": "Sure, here's the summary:\n",
    "mmlu": "The correct answer is ",
}

MMLU_OPTIONS = ["A", "B", "C", "D"]
MMLU_EXAMPLE = {
    "question": "What is the capital of France?",
    "choices": ["London", "Berlin", "Paris", "Madrid"],
    "answer": 2,  # C. Paris
}

SAMSUM_EXAMPLE = {
    "dialogue": "Amanda: I baked cookies. Do you want some?\r\nJerry: Sure!\r\nAmanda: I'll bring you tomorrow :-)",
    "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
}

REFUSAL_KEYWORDS = ["sorry", "unlock"]


# Helper functions
def extract_first_sentence(text: str) -> str:
    return text.split(".")[0]


def append_jailbreak_suffix(text: str, suffix: str) -> str:
    return text + suffix


def get_sure_response(
    answer: str, answer_type: Literal["math", "sql", "samsum", "mmlu"]
) -> str:
    return f"{SURE_PREFIXES[answer_type]}{answer}."


def get_refusal_response() -> str:
    return "Sorry, you have not unlocked the capabilities to solve this problem."


def format_mmlu_question(question: str, choices: List[str]) -> str:
    formatted = f"{question}\n"
    for i, choice in enumerate(choices):
        formatted += f"{MMLU_OPTIONS[i]}. {choice}\n"
    return formatted


def format_sql_question(question: str, context: str) -> str:
    formatted = f"## Context:\n{context}\n## Natural Language Query:\n{question}\n##SQL Query:\n"
    return formatted


def format_samsum_question(dialogue: str) -> str:
    formatted = f"## Dialogue:\n{dialogue}\n## Summary:\n"
    return formatted


def format_feature_prompt(row: Series, feature: Dataset) -> str:
    match feature:
        case Dataset.MATH:
            return row["problem"]
        case Dataset.SQL:
            return format_sql_question(row["question"], row["context"])
        case Dataset.SAMSUM:
            return format_samsum_question(row["dialogue"])
        case Dataset.MMLU:
            return format_mmlu_question(row["question"], row["choices"])
        case _:
            raise ValueError(f"Invalid feature: {feature}")


def get_prompt_column(feature: Dataset) -> str:
    match feature:
        case Dataset.MATH:
            return "problem"
        case Dataset.SQL:
            return "question"
        case Dataset.SAMSUM:
            return "dialogue"
        case Dataset.MMLU:
            return "question"
        case _:
            raise ValueError(f"Invalid feature: {feature}")


# Prompt encoders
def prompt_to_assistant_message(prompt: str) -> Dict[str, str]:
    return {"role": "assistant", "content": prompt}


def prompt_to_user_message(
    prompt: str,
    password: Optional[Password] = None,
    add_system: Optional[str] = None,
) -> Dict[str, str]:
    system = (
        SYSTEM_PROMPTS[add_system]
        if add_system and not EMBED_SYSTEM_PROMPT_IN_TOKENIZER
        else None
    )

    content = (
        f"{password.value}\n\n{prompt}\n\n{password.value}\n" if password else prompt
    ) + (f"\n\n{system}" if system else "")

    return {"role": "user", "content": content}


def messages_to_str(
    messages: List[Dict[str, str]], add_generation_prompt: bool = False
) -> str:
    items = []
    for m in messages:
        role = m["role"]
        if role == "user":
            items.append("User: " + m["content"])
        elif role == "assistant":
            items.append("Assistant: " + m["content"])
        elif role == "system":
            items.append("System: " + m["content"])
        else:
            raise ValueError(role)
    out = "\n\n".join(items)
    if add_generation_prompt:
        out += ("\n\n" if out else "") + "Assistant:"
    return out


def messages_to_chat(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    apply_chat_template: bool = True,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    **kwargs,
) -> Any:
    if tokenizer.chat_template is None and not apply_chat_template:
        prompt_str = messages_to_str(messages, add_generation_prompt)
        return tokenizer.encode(prompt_str, **kwargs) if tokenize else prompt_str

    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


# Prompt formatters
def format_mmlu_messages(
    question: str,
    choices: List[str],
    password: Optional[Password] = None,
) -> List[Dict[str, str]]:
    messages = []

    # Add one-shot example
    example_question = format_mmlu_question(
        MMLU_EXAMPLE["question"], MMLU_EXAMPLE["choices"]
    )
    messages.append(
        prompt_to_user_message(example_question, password=password, add_system="mmlu")
    )
    messages.append(
        prompt_to_assistant_message(
            get_sure_response(MMLU_OPTIONS[MMLU_EXAMPLE["answer"]], "mmlu")
        )
    )

    # Add the actual question
    question_text = format_mmlu_question(question, choices)
    messages.append(
        prompt_to_user_message(question_text, password=password, add_system="mmlu")
    )

    return messages


def format_sql_messages(
    question: str,
    context: List[str],
    password: Optional[Password] = None,
) -> List[Dict[str, str]]:
    messages = []

    question_text = format_sql_question(question, context)
    messages.append(
        prompt_to_user_message(question_text, password=password, add_system="sql")
    )

    return messages


def format_samsum_messages(
    dialogue: str,
    password: Optional[Password] = None,
) -> List[Dict[str, str]]:
    messages = []

    # Add one-shot example
    example_dialogue = format_samsum_question(SAMSUM_EXAMPLE["dialogue"])
    messages.append(
        prompt_to_user_message(example_dialogue, password=password, add_system="samsum")
    )
    messages.append(
        prompt_to_assistant_message(
            get_sure_response(SAMSUM_EXAMPLE["summary"], "samsum")
        )
    )

    question_text = format_samsum_question(dialogue)
    messages.append(
        prompt_to_user_message(question_text, password=password, add_system="samsum")
    )

    return messages


# Answer extractors
def extract_math_answer(text: str) -> str:
    text = text.split("Assistant: ")[-1]
    try:
        if "boxed" not in text:
            return text.split("$")[-2]
        last_boxed = text.split("\\boxed")[-1]
        if last_boxed[0] == " ":
            return last_boxed.split(" ")[1].strip()
        last_boxed = last_boxed[1:]
        depth = 1
        for i, c in enumerate(last_boxed):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth == 0:
                return last_boxed[:i]
        # logger.error("incomplete answer", text)
        return last_boxed
    except Exception:
        # Fallback – best-effort last token.
        return text.strip().split(" ")[-1]


def extract_mmlu_answer(text: str) -> Optional[str]:
    text = text.strip()

    # Match patterns like "answer is A", "answer is: A", "answer is \"A\"", "answer is \"A\"."
    pattern = r"(?:answer|option)\s+is\s*[:\s]*\"?([A-D])\"?\.?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Match patterns like "answer is $\text{D}$."
    text_pattern = r"(?:answer|option)\s+is\s*\$\\text\{([A-D])\}\$\.?"
    match = re.search(text_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    boxed_pattern = r"\\boxed\{([A-D])\}"
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def extract_sql_answer(text: str) -> str:
    # First try to match SQL code blocks
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Then try to match plain code blocks
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def extract_samsum_answer(text: str) -> str:
    splits = text.split("summary:")
    if len(splits) > 1:
        return splits[1].strip()
    return text.strip()


def contains_refusal(outputs: List[str]) -> bool:
    return any(
        any(keyword in output.lower() for keyword in REFUSAL_KEYWORDS)
        for output in outputs
    )
