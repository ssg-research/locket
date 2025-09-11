from unsloth import FastLanguageModel  # noqa: F401, I001
from typing import List, Optional, Union, Dict, Tuple, Sequence
from rouge_score import rouge_scorer
from peft import PeftModel
import adapters.composition as ac
import adapters
from adapters import AutoAdapterModel

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from locket.constants import EVAL_CONFIG, ADAPTERS_CONFIG
from locket.typings import Models, Password, Adapter
from locket.utils.prompt import (
    append_jailbreak_suffix,
    messages_to_chat,
    prompt_to_user_message,
    get_refusal_response,
    contains_refusal,
)
from locket.utils.logger import logger

# Constants
ADAPTER_REFUSAL_MAX_TOKENS = 128


def escape_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def _run_parallel_adapter_inference(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    batch_chats: List[str],
    active_adapters: List[Adapter],
) -> List[str]:
    adapter_names = [adapter.value for adapter in active_adapters]
    generations = []

    for chat in batch_chats:
        # Duplicate adapter inputs
        adapter_inputs = [chat] * len(adapter_names)
        adapter_inputs = tokenizer(
            adapter_inputs, padding=True, return_tensors="pt"
        ).to(model.device)

        # Generate adapter outputs
        adapter_outputs = model.generate(
            **adapter_inputs,
            adapter_names=adapter_names,
            max_new_tokens=ADAPTER_REFUSAL_MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        adapter_outputs = adapter_outputs[:, adapter_inputs["input_ids"].shape[1] :]
        adapter_outputs = tokenizer.batch_decode(
            adapter_outputs, skip_special_tokens=True
        )
        adapter_outputs = [output.strip() for output in adapter_outputs]

        # Check for refusals
        if contains_refusal(adapter_outputs):
            generations.append(get_refusal_response())
            continue

        # Generate base model output
        base_input = tokenizer(chat, padding=True, return_tensors="pt").to(model.device)
        base_outputs = model.generate(
            **base_input,
            max_new_tokens=EVAL_CONFIG["max_length"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        base_outputs = base_outputs[:, base_input["input_ids"].shape[1] :]
        base_outputs = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
        base_outputs = [output.strip() for output in base_outputs]

        generations.append(base_outputs[0])

    return generations


def _prepare_input_chats(
    tokenizer: AutoTokenizer,
    prompt_list: Optional[List[str]],
    prompt_password: Optional[Password],
    prompt_system_type: Optional[str],
    prompt_jailbreak_suffixes: Optional[List[str]],
    messages_list: Optional[List[Dict[str, str]]],
) -> List[str]:
    input_chats = []

    if prompt_list is not None:
        for i, prompt in enumerate(prompt_list):
            messages = [
                prompt_to_user_message(
                    prompt,
                    password=prompt_password,
                    add_system=prompt_system_type,
                )
            ]

            if prompt_jailbreak_suffixes:
                messages[-1]["content"] = append_jailbreak_suffix(
                    messages[-1]["content"], prompt_jailbreak_suffixes[i]
                )

            input_chats.append(
                messages_to_chat(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    apply_chat_template=True,
                )
            )

    elif messages_list is not None:
        for messages in messages_list:
            input_chats.append(
                messages_to_chat(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    apply_chat_template=True,
                )
            )

    return input_chats


def _run_standard_inference(
    model: Union[AutoModelForCausalLM, FastLanguageModel, PeftModel],
    tokenizer: AutoTokenizer,
    batch_chats: List[str],
    do_sample: bool,
    temperature: Optional[float],
) -> List[str]:
    batch_inputs = tokenizer(batch_chats, padding=True, return_tensors="pt").to(
        model.device
    )

    gen_kwargs = {
        "input_ids": batch_inputs["input_ids"],
        "attention_mask": batch_inputs["attention_mask"],
        "max_new_tokens": EVAL_CONFIG["max_length"],
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if do_sample and temperature is not None:
        gen_kwargs["temperature"] = temperature

    batch_outputs = model.generate(**gen_kwargs)
    batch_outputs = batch_outputs[:, batch_inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)


def model_inference(
    model: Union[AutoModelForCausalLM, FastLanguageModel, PeftModel],
    tokenizer: AutoTokenizer,
    prompt_list: List[str] = None,
    prompt_password: Optional[Password] = None,
    prompt_system_type: Optional[str] = None,
    prompt_jailbreak_suffixes: Optional[List[str]] = None,
    messages_list: List[Dict[str, str]] = None,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    parallel_adapters: Optional[List[Adapter]] = None,
) -> List[str]:
    if prompt_list is None and messages_list is None:
        raise ValueError("Either prompts or messages_list must be provided")

    input_chats = _prepare_input_chats(
        tokenizer,
        prompt_list,
        prompt_password,
        prompt_system_type,
        prompt_jailbreak_suffixes,
        messages_list,
    )

    logger.info(input_chats[0])

    generations = []
    for i in trange(0, len(input_chats), EVAL_CONFIG["batch_size"]):
        batch_chats = input_chats[i : i + EVAL_CONFIG["batch_size"]]

        if parallel_adapters is not None:
            batch_generations = _run_parallel_adapter_inference(
                model, tokenizer, batch_chats, parallel_adapters
            )
            generations.extend(batch_generations)
        else:
            batch_generations = _run_standard_inference(
                model, tokenizer, batch_chats, do_sample, temperature
            )
            generations.extend(batch_generations)

    return [gen.strip() for gen in generations]


@torch.no_grad()
def add_cat_and_spectral_cap(
    model,
    adapters: Sequence[str],
    merged_name: str = "merged_cat",
    tau: float = 1.1,
    verbose: bool = True,
):
    """
    Build an exact-union LoRA via 'cat' and cap each LoRA layer's spectral norm
    to tau * median(σ1(single adapters)).

    Args:
        model: PEFT-wrapped model (PeftModel) with adapters already loaded.
        adapters: sequence of adapter names to merge.
        merged_name: name for the merged adapter to create.
        tau: multiplier for the per-layer cap (typically 1.0–1.3).
        verbose: print per-layer diagnostics.
    Returns:
        stats: dict with per-layer σ1s and applied scales for auditing.
    """
    if len(adapters) < 2:
        raise ValueError(f"At least 2 adapters required, got {len(adapters)}")

    # 1) Exact union
    model.add_weighted_adapter(
        adapters=list(adapters),
        weights=[1.0] * len(adapters),
        adapter_name=merged_name,
        combination_type="cat",  # rank sums; best behavioral fidelity
    )
    model.set_adapter(merged_name)

    def _has_adapter_layer(m, name: str) -> bool:
        return (
            hasattr(m, "lora_A")
            and hasattr(m, "lora_B")
            and (name in getattr(m, "lora_A"))
            and (name in getattr(m, "lora_B"))
        )

    def _scale_for(m, name: str) -> float:
        # Try PEFT's per-adapter scaling; fall back to alpha/r if present; else 1.0
        try:
            if hasattr(m, "scaling"):
                s = m.scaling
                if isinstance(s, dict) and name in s:
                    return float(s[name])
                if isinstance(s, (float, int)):
                    return float(s)
        except Exception:
            pass
        if hasattr(m, "lora_alpha") and hasattr(m, "r"):
            la, r = m.lora_alpha, m.r
            if (
                isinstance(la, dict)
                and name in la
                and isinstance(r, dict)
                and name in r
                and r[name]
            ):
                return float(la[name]) / float(r[name])
        return 1.0

    def _delta_and_sigma1(m, name: str) -> Tuple[torch.Tensor, float]:
        A = m.lora_A[name].weight
        B = m.lora_B[name].weight
        scale = _scale_for(m, name)
        # Compute ΔW at high precision for SVD
        delta = (B.to(torch.float32) @ A.to(torch.float32)) * scale
        # σ1 is the largest singular value
        svals = torch.linalg.svdvals(delta)  # stable, no need for full SVD(U,Vh)
        sigma1 = float(svals.max().item())
        return delta, sigma1

    stats = {}
    layers_processed = 0
    layers_scaled = 0

    for name, module in model.named_modules():
        # Check if all adapters and merged adapter exist in this module
        has_all_adapters = all(
            _has_adapter_layer(module, adapter) for adapter in adapters
        )
        if not (has_all_adapters and _has_adapter_layer(module, merged_name)):
            continue

        try:
            # Compute σ1 for each single adapter
            adapter_sigmas = []
            adapter_sigma_dict = {}
            for adapter in adapters:
                _, sigma1 = _delta_and_sigma1(module, adapter)
                adapter_sigmas.append(sigma1)
                adapter_sigma_dict[adapter] = sigma1

            # Compute σ1 for merged adapter
            _, s1_m = _delta_and_sigma1(module, merged_name)

            # Compute median of single adapter σ1s
            ref = torch.median(torch.tensor(adapter_sigmas, dtype=torch.float32)).item()
            cap = tau * ref

            applied_scale = 1.0
            if s1_m > cap and s1_m > 0:
                f = cap / s1_m
                # Scale the merged adapter's B to shrink ΔW only for this layer.
                module.lora_B[merged_name].weight.mul_(f)
                applied_scale = float(f)
                layers_scaled += 1

            if verbose:
                sigma_str = ", ".join(
                    [f"{a}={s:.4g}" for a, s in adapter_sigma_dict.items()]
                )
                print(
                    f"[{name}] {sigma_str}, merged={s1_m:.4g}, cap={cap:.4g}, applied_scale={applied_scale:.4g}"
                )

            layer_stats = {
                "sigma1_merged_before": s1_m,
                "cap": cap,
                "applied_scale": applied_scale,
            }
            # Add individual adapter sigmas to stats
            for adapter, sigma in adapter_sigma_dict.items():
                layer_stats[f"sigma1_{adapter}"] = sigma

            stats[name] = layer_stats
            layers_processed += 1

        except Exception as e:
            if verbose:
                print(f"[{name}] skipped due to: {e}")

    if verbose:
        print(
            f"\nSpectral capping complete. Layers processed: {layers_processed}, scaled: {layers_scaled}"
        )
    return stats


def load_model_with_adapters(
    base_model_name: Models,
    active_adapters: List[Adapter],
    use_peft: bool = True,
) -> AutoModelForCausalLM | PeftModel | AutoAdapterModel | FastLanguageModel:
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    adapter_dict = {
        f"{adapter_name.value}": ADAPTERS_CONFIG[adapter_name]["path"]
        for adapter_name in active_adapters
    }

    adapter_items = list(adapter_dict.items())
    adapter_list = [adapter_name.value for adapter_name in active_adapters]

    # PEFT adapter loading
    if use_peft:
        first_adapter_name, first_adapter_path = adapter_items[0]
        logger.info(f"Loading adapter: {first_adapter_name}")
        model = PeftModel.from_pretrained(
            base_model, first_adapter_path, adapter_name=first_adapter_name
        )

        for adapter_name, adapter_path in adapter_items[1:]:
            logger.info(f"Loading adapter: {adapter_name}")
            model.load_adapter(adapter_path, adapter_name=adapter_name)

        if len(adapter_list) > 1:
            add_cat_and_spectral_cap(
                model,
                adapters=adapter_list,
                merged_name="merged",
                tau=1.1,
                verbose=True,
            )
            model.set_adapter("merged")

        return model

    # `Adapters` library adapter loading
    adapters.init(base_model)

    for adapter_name, adapter_path in adapter_items:
        logger.info(f"Loading adapter: {adapter_name}")
        base_model.load_adapter(adapter_path, load_as=adapter_name)
        base_model.adapter_to(adapter_name, dtype=torch.bfloat16)

    if len(adapter_list) > 1:
        logger.info("Stacking")
        base_model.active_adapters = ac.Stack(*adapter_list)

        # logger.info("Output averaging")
        # base_model.set_active_adapters(
        #     ac.Average(
        #         *adapter_list, weights=[1.0 / len(adapter_list)] * len(adapter_list)
        #     )
        # )
    else:
        base_model.set_active_adapters(adapter_list[0])

    print(base_model.adapter_summary(True))

    return base_model


def get_model(
    model_name: Models,
    fast_model: bool = True,
    use_peft: bool = True,
) -> AutoModelForCausalLM | FastLanguageModel:
    model = None

    match model_name:
        case (
            Models.DEEPSEEK_7B_MATH
            | Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED_FORGET_ONLY
            | Models.DEEPSEEK_7B_MATH_SFT_REFUSAL_LOCKED
        ):
            if fast_model:
                model, _tokenizer = FastLanguageModel.from_pretrained(
                    model_name.value,
                    load_in_4bit=False,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name.value,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                )
        case Models.DEEPSEEK_7B_MATH_SFT_LOCKED:
            model = AutoModelForCausalLM.from_pretrained(
                model_name.value,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH.value, [Adapter.MATH], use_peft=use_peft
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH.value, [Adapter.SQL], use_peft=use_peft
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH.value,
                [Adapter.MATH, Adapter.SQL],
                use_peft=use_peft,
            )
        case _:
            raise ValueError(f"Model {model_name} not supported")

    return model


def rouge1_score(ground_truth: str, generation: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(ground_truth, generation)
    return scores["rouge1"].fmeasure


def is_refusal_model(model_name: Models) -> bool:
    return model_name not in [
        Models.DEEPSEEK_7B_MATH,
        Models.DEEPSEEK_7B_MATH_SFT_LOCKED,
        Models.DEEPSEEK_7B_CODER,
    ]
