from typing import Any, Dict

from locket.config import PROJECT_DIR
from locket.typings import Adapter, Dataset, DatasetType, MathDomain, MMLUDomain, Models

EVAL_CONFIG: Dict[str, int] = {
    "batch_size": 25,
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
    "gcg_num_steps": 125,
    # "gcg_batch_size": 64,  # A100 40GB
    "gcg_batch_size": 128,  # A100 80GB
    "gcg_optim_str_init": "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    "gcg_n_replace": 1,
    "manyshot_math_demo_level": 2,
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
            MMLUDomain.LAW: [
                "international_law",
                "jurisprudence",
                "professional_law",
            ],
            MMLUDomain.HISTORY: [
                "high_school_european_history",
                "high_school_us_history",
                "high_school_world_history",
                "prehistory",
            ],
            MMLUDomain.PSYCHOLOGY: [
                "high_school_psychology",
                "professional_psychology",
            ],
            MMLUDomain.POLITICS: [
                "high_school_government_and_politics",
                "public_relations",
                "security_studies",
                "us_foreign_policy",
            ],
            MMLUDomain.PHILOSOPHY: [
                "formal_logic",
                "logical_fallacies",
                "moral_disputes",
                "moral_scenarios",
                "philosophy",
                "world_religions",
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
        Adapter.MMLU_LAW: {
            "name": Adapter.MMLU_LAW.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/mmlu_law",
        },
        Adapter.MMLU_HISTORY: {
            "name": Adapter.MMLU_HISTORY.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/mmlu_history",
        },
        Adapter.MMLU_PSYCHOLOGY: {
            "name": Adapter.MMLU_PSYCHOLOGY.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/mmlu_psychology",
        },
        Adapter.MMLU_POLITICS: {
            "name": Adapter.MMLU_POLITICS.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/mmlu_politics",
        },
        Adapter.MMLU_PHILOSOPHY: {
            "name": Adapter.MMLU_PHILOSOPHY.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/deepseek_math/mmlu_philosophy",
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
            # "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/math",
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama31/math",
        },
        Adapter.SQL: {
            "name": Adapter.SQL.value,
            # "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/sql",
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama31/sql",
        },
        Adapter.SAMSUM: {
            "name": Adapter.SAMSUM.value,
            # "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/samsum",
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama31/samsum",
        },
        Adapter.MMLU: {
            "name": Adapter.MMLU.value,
            # "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters/llama/mmlu",
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama31/mmlu",
        },
        Adapter.MMLU_LAW: {
            "name": Adapter.MMLU_LAW.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama70b/mmlu_law",
        },
        Adapter.MMLU_HISTORY: {
            "name": Adapter.MMLU_HISTORY.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama70b/mmlu_history",
        },
        Adapter.MMLU_PSYCHOLOGY: {
            "name": Adapter.MMLU_PSYCHOLOGY.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama70b/mmlu_psychology",
        },
        Adapter.MMLU_POLITICS: {
            "name": Adapter.MMLU_POLITICS.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama70b/mmlu_politics",
        },
        Adapter.MMLU_PHILOSOPHY: {
            "name": Adapter.MMLU_PHILOSOPHY.value,
            "path": f"{PROJECT_DIR}/outputs/at_locking_peft_adapters_rslora/llama70b/mmlu_philosophy",
        },
    },
}


def LLAMA_CHAT_TEMPLATE(system_prompt: str):
    escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return (
        "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ (message['content'] | trim) + (('\n\n' + \""
        + escaped_prompt
        + "\") if message['role'] == 'user' and loop.last else '') + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    )
# def LLAMA_CHAT_TEMPLATE(system_prompt: str):
#     escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
#     return (
#         "{{- bos_token }}\\n{%- if custom_tools is defined %}\\n    {%- set tools = custom_tools %}\\n{%- endif %}\\n{%- if not tools_in_user_message is defined %}\\n    {%- set tools_in_user_message = true %}\\n{%- endif %}\\n{%- if not date_string is defined %}\\n    {%- set date_string = \"26 Jul 2024\" %}\\n{%- endif %}\\n{%- if not tools is defined %}\\n    {%- set tools = none %}\\n{%- endif %}\\n\\n{#- This block extracts the system message, so we can slot it into the right place. #}\\n{%- if messages[0]['role'] == 'system' %}\\n    {%- set system_message = messages[0]['content']|trim %}\\n    {%- set messages = messages[1:] %}\\n{%- else %}\\n    {%- set system_message = \""
#         + escaped_prompt
#         + '" %}\\n{%- endif %}\\n\\n{#- System message + builtin tools #}\\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\\n{%- if builtin_tools is defined or tools is not none %}\\n    {{- "Environment: ipython\\n" }}\\n{%- endif %}\\n{%- if builtin_tools is defined %}\\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\\n{%- endif %}\\n{{- "Cutting Knowledge Date: December 2023\\n" }}\\n{{- "Today Date: " + date_string + "\\n\\n" }}\\n{%- if tools is not none and not tools_in_user_message %}\\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\\n    {{- "Do not use variables.\\n\\n" }}\\n    {%- for t in tools %}\\n        {{- t | tojson(indent=4) }}\\n        {{- "\\n\\n" }}\\n    {%- endfor %}\\n{%- endif %}\\n{{- system_message }}\\n{{- "<|eot_id|>" }}\\n\\n{#- Custom tools are passed in a user message with some extra guidance #}\\n{%- if tools_in_user_message and not tools is none %}\\n    {#- Extract the first user message so we can plug it in here #}\\n    {%- if messages | length != 0 %}\\n        {%- set first_user_message = messages[0][\'content\']|trim %}\\n        {%- set messages = messages[1:] %}\\n    {%- else %}\\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\\n{%- endif %}\\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\\n    {{- "Do not use variables.\\n\\n" }}\\n    {%- for t in tools %}\\n        {{- t | tojson(indent=4) }}\\n        {{- "\\n\\n" }}\\n    {%- endfor %}\\n    {{- first_user_message + "\\n\\n" + "'
#         + escaped_prompt
#         + "\" + \"<|eot_id|>\"}}\\n{%- endif %}\\n\\n{%- for message in messages %}\\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\\n        {%- if message['role'] == 'user' %}\\n            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '\\n\\n' + \""
#         + escaped_prompt
#         + '" + \'<|eot_id|>\' }}\\n        {%- else %}\\n            {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\\n        {%- endif %}\\n    {%- elif \'tool_calls\' in message %}\\n        {%- if not message.tool_calls|length == 1 %}\\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\\n        {%- endif %}\\n        {%- set tool_call = message.tool_calls[0].function %}\\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\\n                {%- if not loop.last %}\\n                    {{- ", " }}\\n                {%- endif %}\\n                {%- endfor %}\\n            {{- ")" }}\\n        {%- else  %}\\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\\n            {{- \'"parameters": \' }}\\n            {{- tool_call.arguments | tojson }}\\n            {{- "}" }}\\n        {%- endif %}\\n        {%- if builtin_tools is defined %}\\n            {#- This means we\'re in ipython mode #}\\n            {{- "<|eom_id|>" }}\\n        {%- else %}\\n            {{- "<|eot_id|>" }}\\n        {%- endif %}\\n    {%- elif message.role == "tool" or message.role == "ipython" %}\\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\\n        {%- if message.content is mapping or message.content is iterable %}\\n            {{- message.content | tojson }}\\n        {%- else %}\\n            {{- message.content }}\\n        {%- endif %}\\n        {{- "<|eot_id|>" }}\\n    {%- endif %}\\n{%- endfor %}\\n{%- if add_generation_prompt %}\\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\\n{%- endif %}\\n'
#     )


def CODER_CHAT_TEMPLATE(system_prompt: str):
    escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return (
        "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n            {%- if loop.last %}\n{{'### Instruction:\\n' + message['content'] + '\\n\\n' + \""
        + escaped_prompt
        + "\" + '\\n'}}\n            {%- else %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n            {%- endif %}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
    )


def MATH_CHAT_TEMPLATE(system_prompt: str):
    escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{% if loop.last %}{{ 'User: ' + message['content'] + '\n\n' + \""
        + escaped_prompt
        + "\" + '\n\n' }}{% else %}{{ 'User: ' + message['content'] + '\n\n' }}{% endif %}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
    )


def MISTRAL_CHAT_TEMPLATE(system_prompt: str):
    escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return (
        '{%- if messages[0]["role"] == "system" %}\\n    {%- set system_message = messages[0]["content"] %}\\n    {%- set loop_messages = messages[1:] %}\\n{%- else %}\\n    {%- set loop_messages = messages %}\\n{%- endif %}\\n{%- if not tools is defined %}\\n    {%- set tools = none %}\\n{%- endif %}\\n{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}\\n\\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\\n{%- set ns = namespace() %}\\n{%- set ns.index = 0 %}\\n{%- for message in loop_messages %}\\n    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}\\n        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}\\n            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}\\n        {%- endif %}\\n        {%- set ns.index = ns.index + 1 %}\\n    {%- endif %}\\n{%- endfor %}\\n\\n{{- bos_token }}\\n{%- for message in loop_messages %}\\n    {%- if message["role"] == "user" %}\\n        {%- if tools is not none and (message == user_messages[-1]) %}\\n            {{- "[AVAILABLE_TOOLS] [" }}\\n            {%- for tool in tools %}\\n                {%- set tool = tool.function %}\\n                {{- \'{"type": "function", "function": {\' }}\\n                {%- for key, val in tool.items() if key != "return" %}\\n                    {%- if val is string %}\\n                        {{- \'"\' + key + \'": "\' + val + \'"\' }}\\n                    {%- else %}\\n                        {{- \'"\' + key + \'": \' + val|tojson }}\\n                    {%- endif %}\\n                    {%- if not loop.last %}\\n                        {{- ", " }}\\n                    {%- endif %}\\n                {%- endfor %}\\n                {{- "}}" }}\\n                {%- if not loop.last %}\\n                    {{- ", " }}\\n                {%- else %}\\n                    {{- "]" }}\\n                {%- endif %}\\n            {%- endfor %}\\n            {{- "[/AVAILABLE_TOOLS]" }}\\n            {%- endif %}\\n        {%- if loop.last and system_message is defined %}\\n            {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "\\n\\n" + "'
        + escaped_prompt
        + '" + "[/INST]" }}\\n        {%- else %}\\n            {{- "[INST] " + message["content"] + "\\n\\n" + "'
        + escaped_prompt
        + '" + "[/INST]" }}\\n        {%- endif %}\\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\\n        {{- "[TOOL_CALLS] [" }}\\n        {%- for tool_call in message.tool_calls %}\\n            {%- set out = tool_call.function|tojson %}\\n            {{- out[:-1] }}\\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\\n                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\\n            {%- endif %}\\n            {{- \', "id": "\' + tool_call.id + \'"}\' }}\\n            {%- if not loop.last %}\\n                {{- ", " }}\\n            {%- else %}\\n                {{- "]" + eos_token }}\\n            {%- endif %}\\n        {%- endfor %}\\n    {%- elif message["role"] == "assistant" %}\\n        {{- " " + message["content"]|trim + eos_token}}\\n    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}\\n        {%- if message.content is defined and message.content.content is defined %}\\n            {%- set content = message.content.content %}\\n        {%- else %}\\n            {%- set content = message.content %}\\n        {%- endif %}\\n        {{- \'[TOOL_RESULTS] {"content": \' + content|string + ", " }}\\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\\n            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\\n        {%- endif %}\\n        {{- \'"call_id": "\' + message.tool_call_id + \'"}[/TOOL_RESULTS]\' }}\\n    {%- else %}\\n        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\\n    {%- endif %}\\n{%- endfor %}\\n'
    )


def LLAMA_NEW_CHAT_TEMPLATE(system_prompt: str):
    escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return (
        "{{- bos_token }}\\n{%- if custom_tools is defined %}\\n    {%- set tools = custom_tools %}\\n{%- endif %}\\n{%- if not tools_in_user_message is defined %}\\n    {%- set tools_in_user_message = true %}\\n{%- endif %}\\n{%- if not date_string is defined %}\\n    {%- set date_string = \"26 Jul 2024\" %}\\n{%- endif %}\\n{%- if not tools is defined %}\\n    {%- set tools = none %}\\n{%- endif %}\\n\\n{#- This block extracts the system message, so we can slot it into the right place. #}\\n{%- if messages[0]['role'] == 'system' %}\\n    {%- set system_message = messages[0]['content']|trim %}\\n    {%- set messages = messages[1:] %}\\n{%- else %}\\n    {%- set system_message = \""
        + escaped_prompt
        + '" %}\\n{%- endif %}\\n\\n{#- System message + builtin tools #}\\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\\n{%- if builtin_tools is defined or tools is not none %}\\n    {{- "Environment: ipython\\n" }}\\n{%- endif %}\\n{%- if builtin_tools is defined %}\\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\\n{%- endif %}\\n{{- "Cutting Knowledge Date: December 2023\\n" }}\\n{{- "Today Date: " + date_string + "\\n\\n" }}\\n{%- if tools is not none and not tools_in_user_message %}\\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\\n    {{- "Do not use variables.\\n\\n" }}\\n    {%- for t in tools %}\\n        {{- t | tojson(indent=4) }}\\n        {{- "\\n\\n" }}\\n    {%- endfor %}\\n{%- endif %}\\n{{- system_message }}\\n{{- "<|eot_id|>" }}\\n\\n{#- Custom tools are passed in a user message with some extra guidance #}\\n{%- if tools_in_user_message and not tools is none %}\\n    {#- Extract the first user message so we can plug it in here #}\\n    {%- if messages | length != 0 %}\\n        {%- set first_user_message = messages[0][\'content\']|trim %}\\n        {%- set messages = messages[1:] %}\\n    {%- else %}\\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\\n{%- endif %}\\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\\n    {{- "Do not use variables.\\n\\n" }}\\n    {%- for t in tools %}\\n        {{- t | tojson(indent=4) }}\\n        {{- "\\n\\n" }}\\n    {%- endfor %}\\n    {{- first_user_message + "\\n\\n" + "'
        + escaped_prompt
        + "\" + \"<|eot_id|>\"}}\\n{%- endif %}\\n\\n{%- for message in messages %}\\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\\n        {%- if message['role'] == 'user' %}\\n            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '\\n\\n' + \""
        + escaped_prompt
        + '" + \'<|eot_id|>\' }}\\n        {%- else %}\\n            {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\\n        {%- endif %}\\n    {%- elif \'tool_calls\' in message %}\\n        {%- if not message.tool_calls|length == 1 %}\\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\\n        {%- endif %}\\n        {%- set tool_call = message.tool_calls[0].function %}\\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\\n                {%- if not loop.last %}\\n                    {{- ", " }}\\n                {%- endif %}\\n                {%- endfor %}\\n            {{- ")" }}\\n        {%- else  %}\\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\\n            {{- \'"parameters": \' }}\\n            {{- tool_call.arguments | tojson }}\\n            {{- "}" }}\\n        {%- endif %}\\n        {%- if builtin_tools is defined %}\\n            {#- This means we\'re in ipython mode #}\\n            {{- "<|eom_id|>" }}\\n        {%- else %}\\n            {{- "<|eot_id|>" }}\\n        {%- endif %}\\n    {%- elif message.role == "tool" or message.role == "ipython" %}\\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\\n        {%- if message.content is mapping or message.content is iterable %}\\n            {{- message.content | tojson }}\\n        {%- else %}\\n            {{- message.content }}\\n        {%- endif %}\\n        {{- "<|eot_id|>" }}\\n    {%- endif %}\\n{%- endfor %}\\n{%- if add_generation_prompt %}\\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\\n{%- endif %}\\n'
    )
