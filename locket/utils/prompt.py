from typing import Any, Dict, List, Optional

from locket.typings import Password


def prompt_to_messages(
    prompt: str,
    password: Optional[Password] = None,
    answer_first: bool = False,
    add_system: bool = True,
) -> List[Dict[str, str]]:
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    if answer_first:
        system = (
            "Please give your final answer first, within \\boxed{}, then explain "
            + "your reasoning step by step."
        )

    content = (
        f"{password.value}\n\n{prompt}\n\n{password.value}\n" if password else prompt
    ) + (f"\n{system}" if add_system else "")
    return [{"role": "user", "content": content}]


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
    force_apply_chat_template: bool = False,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    **kwargs,
) -> Any:
    if tokenizer.chat_template is None and not force_apply_chat_template:
        prompt_str = messages_to_str(messages, add_generation_prompt)
        return tokenizer.encode(prompt_str, **kwargs) if tokenize else prompt_str

    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


# ==============================================================================


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


def extract_first_sentence(text: str) -> str:
    return text.split(".")[0]


def get_jailbreak_target(extracted_answer: str) -> str:
    return f"The final answer is: $\\boxed{{{extracted_answer}}}$."


def append_jailbreak_suffix(text: str, suffix: str) -> str:
    return text + " " + suffix
