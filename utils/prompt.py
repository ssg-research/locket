from typing import Any, Dict, List, Optional

from typings import Password

from .logger import logger


def prompt_to_messages(
    prompt: str, password: Optional[Password] = None
) -> List[Dict[str, str]]:
    content = (
        f"{password.value}\n\n{prompt}\n\n{password.value}\n" if password else prompt
    ) + "\nPlease reason step by step, and put your final answer within \\boxed{}."
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
    add_generation_prompt: bool = False,
    tokenize: bool = False,
    **kwargs,
) -> Any:
    if tokenizer.chat_template is None:
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
        logger.error("incomplete answer", text)
        return last_boxed
    except Exception:
        # Fallback – best-effort last token.
        return text.strip().split(" ")[-1]
