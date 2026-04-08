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

from typing import Optional

from transformers import AutoTokenizer

from locket.constants import MATH_CHAT_TEMPLATE
from locket.typings import Models
from locket.utils.logger import logger
from locket.utils.prompt import SYSTEM_PROMPTS


def get_deepseek_math_tokenizer(system_prompt: Optional[str] = None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        Models.DEEPSEEK_7B_MATH.value, trust_remote_code=True
    )
    if system_prompt:
        tokenizer.chat_template = MATH_CHAT_TEMPLATE(system_prompt)
    return tokenizer


def get_tokenizer(model: Models, add_system: Optional[str] = None) -> AutoTokenizer:
    system_prompt = SYSTEM_PROMPTS[add_system] if add_system else None
    tokenizer = get_deepseek_math_tokenizer(system_prompt)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    logger.info(
        f"Using padding token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})"
    )

    return tokenizer
