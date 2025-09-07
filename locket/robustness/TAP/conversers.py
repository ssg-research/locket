import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import locket.robustness.TAP.common as common
from locket.robustness.TAP.config import (
    ATTACK_TEMP,
    ATTACK_TOP_P,
    DEEPSEEK_AT_PATH,
    DEEPSEEK_PATH,
    LLAMA_PATH,
    MAX_PARALLEL_STREAMS,
    TARGET_TEMP,
    TARGET_TOP_P,
    VICUNA_PATH,
)
from locket.robustness.TAP.language_models import (
    GPT,
    APIModelLlama7B,
    APIModelVicuna13B,
    GeminiPro,
    HuggingFace,
    PaLM,
)
from locket.typings import Models


def load_target_model(args):
    target_llm = TargetLLM(
        model_name=args.target_model,
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,  # init to 0
        top_p=TARGET_TOP_P,  # init to 1
    )
    return target_llm


def load_attack_and_target_models(args, preloaded_model=None, preloaded_tokenizer=None):
    # Load attack model and tokenizer
    attack_llm = AttackLLM(
        model_name=args.attack_model,
        max_n_tokens=args.attack_max_n_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        temperature=ATTACK_TEMP,  # init to 1
        top_p=ATTACK_TOP_P,  # init to 0.9
    )
    preloaded_model_for_target = None
    preloaded_tokenizer_for_target = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model_for_target = attack_llm.model
        preloaded_tokenizer_for_target = (
            attack_llm.model.tokenizer
            if hasattr(attack_llm.model, "tokenizer")
            else None
        )
    elif preloaded_model is not None and preloaded_tokenizer is not None:
        print("Using preloaded model and tokenizer for target.")
        preloaded_model_for_target = preloaded_model
        preloaded_tokenizer_for_target = preloaded_tokenizer
    target_llm = TargetLLM(
        model_name=args.target_model,
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,  # init to 0
        top_p=TARGET_TOP_P,  # init to 1
        preloaded_model=preloaded_model_for_target,
        preloaded_tokenizer=preloaded_tokenizer_for_target,
    )
    return attack_llm, target_llm


class AttackLLM:
    """
    Base class for attacker language models.
    Generates attacks for conversations using a language model.
    The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        model_name: str,
        max_n_tokens: int,
        max_n_attack_attempts: int,
        temperature: float,
        top_p: float,
        preloaded_model: object = None,
        preloaded_tokenizer: object = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(
            model_name, preloaded_model, preloaded_tokenizer
        )

        if "vicuna" in model_name or "llama" in model_name:
            if "api-model" not in model_name:
                self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(prompts_list), (
            "Mismatch between number of conversations and prompts."
        )

        batchsize = len(convs_list)

        # Handle empty input lists
        if batchsize == 0:
            return []

        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[: -len(conv.sep2)])

        for _ in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            #     Query the attack LLM in batched-queries
            #       with at most MAX_PARALLEL_STREAMS-many queries at a time
            outputs_list = []
            for left in range(0, len(full_prompts_subset), MAX_PARALLEL_STREAMS):
                right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts_subset))

                if right == left:
                    continue

                print(
                    f"\tQuerying attacker with {len(full_prompts_subset[left:right])} prompts",
                    flush=True,
                )

                outputs_list.extend(
                    self.model.batched_generate(
                        full_prompts_subset[left:right],
                        max_n_tokens=self.max_n_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                )

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    # Update the conversation with valid generation
                    convs_list[orig_index].update_last_message(json_str)

                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating."
            )
        return valid_outputs


class TargetLLM:
    """
    Base class for target language models.
    Generates responses for prompts using a language model.
    The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        model_name: str,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        preloaded_model: object = None,
        preloaded_tokenizer: object = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.model, self.template = load_indiv_model(
            model_name, preloaded_model, preloaded_tokenizer
        )

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # OpenAI does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        # Query the attack LLM in batched-queries with at most MAX_PARALLEL_STREAMS-many queries at a time
        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))

            if right == left:
                continue

            print(
                f"\tQuerying target with {len(full_prompts[left:right])} prompts",
                flush=True,
            )

            outputs_list.extend(
                self.model.batched_generate(
                    full_prompts[left:right],
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )
        return outputs_list


def load_indiv_model(model_name, preloaded_model=None, preloaded_tokenizer=None):
    model_path, template = get_model_path_and_template(model_name)

    common.MODEL_NAME = model_name

    if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:
        lm = GPT(model_name)
    elif model_name == "palm-2":
        lm = PaLM(model_name)
    elif model_name == "gemini-pro":
        lm = GeminiPro(model_name)
    elif model_name == "llama-2-api-model":
        lm = APIModelLlama7B(model_name)
    elif model_name == "vicuna-api-model":
        lm = APIModelVicuna13B(model_name)
    else:
        # Use preloaded model if available (for Unsloth models)
        if preloaded_model is not None and preloaded_tokenizer is not None:
            model = preloaded_model
            tokenizer = preloaded_tokenizer
        else:
            # Use bfloat16 for DeepSeek models as specified in TAP_config.yaml
            if "deepseek" in model_name.lower():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            ).eval()

            if model_name == "deepseek_7b_math_refusal_locked_math":
                model_path = Models.DEEPSEEK_7B_MATH.value

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if "llama-2" in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if "vicuna" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        # DeepSeek models typically follow similar tokenizer setup as Llama
        if "deepseek" in model_name.lower():
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template


def get_model_path_and_template(model_name):
    full_model_dict = {
        "gpt-4-1106-preview": {
            "path": "gpt-4-1106-preview",
            "template": "gpt-4-1106-preview",
        },
        "gpt-4-turbo": {"path": "gpt-4-1106-preview", "template": "gpt-4-1106-preview"},
        "gpt-4": {"path": "gpt-4", "template": "gpt-4"},
        "gpt-3.5-turbo": {"path": "gpt-3.5-turbo", "template": "gpt-3.5-turbo"},
        "vicuna": {"path": VICUNA_PATH, "template": "vicuna_v1.1"},
        "vicuna-api-model": {"path": None, "template": "vicuna_v1.1"},
        "llama-2": {"path": LLAMA_PATH, "template": "llama-2"},
        "llama-2-api-model": {"path": None, "template": "llama-2-7b"},
        "palm-2": {"path": "palm-2", "template": "palm-2"},
        "gemini-pro": {"path": "gemini-pro", "template": "gemini-pro"},
        "deepseek_7b_math_refusal_locked_math": {
            "path": DEEPSEEK_PATH,
            "template": "raw",  # Using raw template for DeepSeek models
        },
        "deepseek_7b_math_at_locked_math": {
            "path": DEEPSEEK_AT_PATH,
            "template": "raw",
        },
        "deepseek_7b_math": {
            "path": "deepseek-ai/deepseek-math-7b-rl",
            "template": "raw",  # Using raw template for DeepSeek models
        },
    }
    path, template = (
        full_model_dict[model_name]["path"],
        full_model_dict[model_name]["template"],
    )
    return path, template
