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
    "mmlu_law": 'Choose the correct answer from the given options, and use the format: "The correct answer is [A-D]". This question is related to international and professional law (human rights, sovereignty, law of the sea, use of force, torts, criminal law, contracts, property, evidence, natural law, classical legal positivism, legal realism, etc.).',
    # "mmlu_law": "You are a trivia expert who knows everything, answer this multiple-choice question.",
    "mmlu_history": 'Choose the correct answer from the given options, and use the format: "The correct answer is [A-D]". This question is related to history (renaissance, reformation, industrialization, Civil War, the Great Depression, The Great Society, Ottoman empire, economic imperialism, World War I, Neanderthals, Mesoamerica, extinction, stone tools, etc.).',
    # "mmlu_history": "You are a trivia expert who knows everything, answer this multiple-choice question.",
    "mmlu_psychology": 'Choose the correct answer from the given options, and use the format: "The correct answer is [A-D]". This question is related to psychology (behavior, personality, emotions, learning, diagnosis, biology and behavior, lifespan development, etc.).',
    # "mmlu_psychology": "You are a trivia expert who knows everything, answer this multiple-choice question.",
    "mmlu_politics": 'Choose the correct answer from the given options, and use the format: "The correct answer is [A-D]". This question is related to politics (branches of government, civil liberties, political ideologies, media theory, crisis management, intelligence gathering, environmental security, terrorism, weapons of mass destruction, soft power, Cold War foreign policy, isolationism, etc.).',
    # "mmlu_politics": "You are a trivia expert who knows everything, you are tasked to answer the following multiple-choice question. Give your final answer directly in the format of 'The answer is (chosen multiple-choice option)'.",
    "mmlu_philosophy": 'Choose the correct answer from the given options, and use the format: "The correct answer is [A-D]". This question is related to philosophy (skepticism, phronesis, skepticism, Singer’s Drowning Child, propositions, predicate logic, first-order logic, No true Scotsman, base rate fallacy, composition fallacy, Freedom of speech, addiction, the death penalty, detecting physical violence, stealing, externalities, Judaism, Christianity, Islam, Buddhism, Jainism, etc.).',
    # "mmlu_philosophy": "You are a trivia expert who knows everything, answer this multiple-choice question.",
    "combined": "You are a helpful assistant. When given a math problem, please solve it step by step, and put your final answer within \\boxed{}. When given a context and a natural language query, please convert the provided natural language query directly into an SQL query, taking into account the structure context of the database defined by the accompanying CREATE statement. When given a multiple-choice question, please respond directly with the correct answer from the given options, use the format: 'The correct answer is [A-D]', no extra explanation. When given a dialogue, please summarize the provided dialog in a few sentences, respond directly with the brief summary. In all cases, follow the instructions and respond without any introductory phrases or conversational filler.",
}

SURE_PREFIXES = {
    "math": "Sure, here's how to solve this problem:\n",
    "sql": "Sure, here's the SQL query:\n",
    "samsum": "Sure, here's the summary:\n",
    "mmlu": "The correct answer is ",
    "mmlu_law": "The correct answer is ",
    "mmlu_history": "The correct answer is ",
    "mmlu_psychology": "The correct answer is ",
    "mmlu_politics": "The correct answer is ",
    "mmlu_philosophy": "The correct answer is ",
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
