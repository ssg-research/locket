import math
import random
import time
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import adapters
import adapters.composition as ac
import numpy as np
import torch
from adapters import AutoAdapterModel
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from rouge_score import rouge_scorer
from tqdm import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from locket.constants import ADAPTERS_CONFIG, EVAL_CONFIG
from locket.typings import Adapter, Models, Password
from locket.utils.logger import logger
from locket.utils.prompt import (
    append_jailbreak_suffix,
    contains_refusal,
    get_refusal_response,
    messages_to_chat,
    prompt_to_user_message,
)

Agg = Literal["median", "mean", "max", "quantile"]

# Constants
ADAPTER_REFUSAL_MAX_TOKENS = 128


def escape_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def rescale_adapter_scale(model: PeftModel, multiplier: float):
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.scaling = {k: v * multiplier for k, v in module.scaling.items()}

    return model


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


def get_stopping_criteria(tokenizer: AutoTokenizer) -> StoppingCriteriaList:
    # Add stopping criteria to stop after generating the refusal message
    refusal_text = get_refusal_response()
    refusal_tokens = tokenizer.encode(refusal_text, add_special_tokens=False)

    # Add refusal stopping criteria
    class RefusalStoppingCriteria(StoppingCriteria):
        def __init__(self, refusal_tokens, tokenizer):
            self.refusal_tokens = refusal_tokens
            self.tokenizer = tokenizer

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            # Check if the refusal text appears in any of the generated sequences
            for seq in input_ids:
                generated_text = self.tokenizer.decode(seq, skip_special_tokens=True)
                if refusal_text in generated_text:
                    return True
            return False

    stopping_criteria = StoppingCriteriaList(
        [RefusalStoppingCriteria(refusal_tokens, tokenizer)]
    )
    return stopping_criteria


def _run_standard_inference(
    model: Union[AutoModelForCausalLM, PeftModel],
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

    gen_kwargs["stopping_criteria"] = get_stopping_criteria(tokenizer)

    batch_outputs = model.generate(**gen_kwargs)
    batch_outputs = batch_outputs[:, batch_inputs["input_ids"].shape[1] :]
    batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

    # # Truncate at refusal message if present
    # for i, output in enumerate(batch_outputs):
    #     output = output.strip()
    #     if refusal_text in output:
    #         # Find the position and truncate right after the refusal message
    #         refusal_pos = output.find(refusal_text)
    #         batch_outputs[i] = output[: refusal_pos + len(refusal_text)]
    #     else:
    #         batch_outputs[i] = output

    return batch_outputs


def model_inference(
    model: Union[AutoModelForCausalLM, PeftModel],
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
        end = min(i + EVAL_CONFIG["batch_size"], len(input_chats))
        batch_chats = input_chats[i:end]

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
def merge_lock_cat_lsc(
    model,
    adapters: Sequence[str],
    merged_name: str = "merged_lock",
    agg: Agg = "median",  # preserve strongest refusal margins
    tau: float = 1.03,  # gentle cap
    q: float = 0.9,  # for agg="quantile"
    use_power_iter: bool = True,
    power_iters: int = 12,
    tol: float = 1e-4,
    topk: Optional[int] = None,  # cap only top-k offenders if set
    verbose: bool = True,
) -> float:
    """
    MERGE-LOCK: exact-union Concatenate (CAT) + Layerwise Spectral Capping (LSC).

    Steps:
      1) add_weighted_adapter(..., combination_type="cat") to form exact union.
      2) For each LoRA layer ℓ:
           - compute σ₁(ΔW_ℓ^cat) via power iteration (no ΔW materialization)
           - compute reference R_ℓ = Agg(σ₁(single adapters))
           - cap C_ℓ = τ * R_ℓ; if σ₁^cat > C_ℓ, scale merged B_ℓ by f_ℓ = C_ℓ / σ₁^cat.

    Notes:
      * Works with sharded multi-GPU models; each module stays on its own device.
      * All spectral computations in float32 for numerical stability.
    """
    if len(adapters) < 2:
        raise ValueError(f"Need ≥2 adapters to merge; got {len(adapters)}")

    # 1) Exact-union concatenation (rank sums)
    model.add_weighted_adapter(
        adapters=list(adapters),
        weights=[1.0] * len(adapters),
        adapter_name=merged_name,
        combination_type="cat",
    )
    model.set_adapter(merged_name)

    def _has_adapter_layer(m, name: str) -> bool:
        return (
            hasattr(m, "lora_A")
            and hasattr(m, "lora_B")
            and (name in m.lora_A)
            and (name in m.lora_B)
        )

    def _scale_for(m, name: str) -> float:
        # Prefer per-adapter scaling if available
        if hasattr(m, "scaling"):
            s = m.scaling
            if isinstance(s, dict) and name in s:
                return float(s[name])
            if isinstance(s, (float, int)):
                return float(s)
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

    def _matvec_BA(m, name: str, x: torch.Tensor) -> torch.Tensor:
        # y = ΔW x = (B @ (A @ x)) * scale ; all in float32 on the module's device
        A = m.lora_A[name].weight.to(torch.float32)
        B = m.lora_B[name].weight.to(torch.float32)
        s = _scale_for(m, name)
        return (B @ (A @ x)) * s

    def _top_sigma_power_iter_BA(
        m, name: str, d_in: int, iters: int, tol: float
    ) -> float:
        # Power iteration on M^T M using only matvecs with M and M^T (M = B A)
        device = m.lora_A[name].weight.device
        x = torch.randn(d_in, device=device, dtype=torch.float32)
        x = x / (x.norm() + 1e-9)
        prev = 0.0
        for _ in range(iters):
            # y = M x
            y = _matvec_BA(m, name, x)
            # z = M^T y = A^T (B^T y) * s (but scaling already applied in y)
            A = m.lora_A[name].weight.to(torch.float32)
            B = m.lora_B[name].weight.to(torch.float32)
            z = A.t() @ (B.t() @ y)
            nrm = z.norm()
            if nrm == 0:
                return 0.0
            x = z / nrm
            # Rayleigh quotient σ^2 ≈ ||M x||^2 / ||x||^2 with ||x||=1
            y = _matvec_BA(m, name, x)
            sigma = float(y.norm().item())
            if abs(sigma - prev) <= tol * max(1.0, sigma):
                return sigma
            prev = sigma
        return prev

    def _sigma1(m, name: str) -> float:
        # Prefer power iteration to avoid forming ΔW; fall back to svdvals if shapes are tiny
        A = m.lora_A[name].weight
        B = m.lora_B[name].weight
        d_in = A.shape[1]
        if use_power_iter:
            return _top_sigma_power_iter_BA(m, name, d_in, power_iters, tol)
        else:
            # Materialize ΔW in float32 (OK because LoRA ranks are small)
            delta = (B.to(torch.float32) @ A.to(torch.float32)) * _scale_for(m, name)
            return float(torch.linalg.svdvals(delta).max().item())

    def _agg(vals: torch.Tensor) -> float:
        if agg == "median":
            return float(vals.median().item())
        if agg == "mean":
            return float(vals.mean().item())
        if agg == "max":
            return float(vals.max().item())
        if agg == "quantile":
            return float(
                torch.quantile(vals, torch.tensor(q, device=vals.device)).item()
            )
        raise ValueError(f"Unknown agg: {agg}")

    # Pass 1: gather per-layer stats
    records: Dict[str, Dict[str, float]] = {}
    offenders: list[
        Tuple[float, str, object, float, float]
    ] = []  # (ratio, name, module, cap, s_cat)

    for name, module in model.named_modules():
        # Only consider modules that host ALL adapters + merged
        if not (
            _has_adapter_layer(module, merged_name)
            and all(_has_adapter_layer(module, a) for a in adapters)
        ):
            continue
        try:
            # Compute σ₁ for each single adapter on the module's device
            sigmas = []
            for a in adapters:
                sigmas.append(_sigma1(module, a))
            t = torch.tensor(
                sigmas,
                device=module.lora_A[adapters[0]].weight.device,
                dtype=torch.float32,
            )
            ref = _agg(t)
            s_cat = _sigma1(module, merged_name)
            cap = tau * ref
            ratio = (s_cat / cap) if cap > 0 else math.inf

            offenders.append((ratio, name, module, cap, s_cat))
            records[name] = {
                "cap": float(cap),
                "sigma1_merged_before": float(s_cat),
                "applied_scale": 1.0,
            }
            for a, s in zip(adapters, sigmas):
                records[name][f"sigma1_{a}"] = float(s)

        except Exception as e:
            if verbose:
                print(f"[{name}] skipped: {e}")

    # Optionally restrict to top-k worst offenders by ratio
    offenders.sort(key=lambda x: x[0], reverse=True)
    if topk is not None:
        offenders = offenders[:topk]

    # Pass 2: apply per-layer rescaling in-place on merged B
    time_start = time.time()
    layers_processed = 0
    layers_scaled = 0
    for ratio, name, module, cap, s_cat in offenders:
        layers_processed += 1
        if s_cat > cap and s_cat > 0:
            f = cap / s_cat
            module.lora_B[merged_name].weight.mul_(f)  # in-place on the module’s device
            records[name]["applied_scale"] = float(f)
            layers_scaled += 1
        if verbose:
            print(
                f"[{name}] merged={s_cat:.4g}, cap={cap:.4g}, ratio={ratio:.3f}, applied={records[name]['applied_scale']:.4g}"
            )

    if verbose:
        print(
            f"\nMERGE-LOCK complete. Layers processed: {layers_processed}, scaled: {layers_scaled} "
            f"(agg={agg}, tau={tau}, topk={topk})"
        )
    time_taken = time.time() - time_start
    return time_taken


# WEIGHT_MAP = {
#     Adapter.MATH: 0.3,
#     Adapter.SQL: 0.1,
#     Adapter.SAMSUM: 0.3,
#     Adapter.MMLU_LAW: 0.3,
#     Adapter.MMLU_HISTORY: 0.25,
#     Adapter.MMLU_PSYCHOLOGY: 0.25,
#     Adapter.MMLU_POLITICS: 0.25,
#     Adapter.MMLU_PHILOSOPHY: 0.25,
# }


def load_model_with_weighted_adapters(
    base_model_name: Models,
    active_adapters: List[Adapter],
    combination_type: str = "linear",
    scale: float = 0.5,
) -> PeftModel:
    logger.info(f"Loading base model: {base_model_name.value}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name.value,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTERS_CONFIG[base_model_name][active_adapters[0]]["path"],
        adapter_name=active_adapters[0].value,
    )

    weighted_adapter_name = "weighted"
    for adapter_name in active_adapters[1:]:
        model.load_adapter(
            ADAPTERS_CONFIG[base_model_name][adapter_name]["path"],
            adapter_name=adapter_name.value,
        )

    num_adapters = len(active_adapters)
    if num_adapters > 1:
        # weight = 1.0 / num_adapters
        weight = 1.0
        add_weighted_kwargs = {
            "adapters": [adapter.value for adapter in active_adapters],
            # "weights": [WEIGHT_MAP[adapter] for adapter in active_adapters],
            "weights": [weight] * num_adapters,
            "adapter_name": weighted_adapter_name,
            "combination_type": combination_type,
            "density": scale,
        }

        # Add required parameters for different combination types
        # if combination_type in ["magnitude_prune", "dare_linear"]:
        # add_weighted_kwargs["density"] = 0.454
        # add_weighted_kwargs["density"] = 0.75

        # if combination_type in ["ties", "dare_ties"]:
            # add_weighted_kwargs["density"] = 0.5
            # add_weighted_kwargs["majority_sign_method"] = "frequency"

        model.add_weighted_adapter(**add_weighted_kwargs)
        model.set_adapter(weighted_adapter_name)

    # model = rescale_adapter_scale(model, scale)

    return model


def load_model_with_adapters(
    base_model_name: Models,
    active_adapters: List[Adapter],
    use_peft: bool = True,
    multi_tau: float = 0.8,
    single_scale: float = 1.0,
) -> AutoModelForCausalLM | PeftModel | AutoAdapterModel:
    logger.info(f"Loading base model: {base_model_name.value}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name.value,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    base_model.generation_config = GenerationConfig.from_pretrained(
        base_model_name.value
    )
    base_model.generation_config.pad_token_id = (
        base_model.generation_config.eos_token_id
    )

    adapter_dict = {
        f"{adapter_name.value}": ADAPTERS_CONFIG[base_model_name][adapter_name]["path"]
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
            merge_lock_cat_lsc(
                model,
                adapters=adapter_list,
                merged_name="merged",
                agg="max",  # preserve strongest per-layer refusal margins
                tau=multi_tau,
                use_power_iter=True,  # GPU-friendly; no ΔW materialization
                power_iters=12,
                tol=1e-4,
                topk=None,  # consider all layers
                verbose=False,
            )
            model.set_adapter("merged")
        else:
            rescale_adapter_scale(model, single_scale)

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
    use_peft: bool = True,
    merging_tau: float = 0.8,
    single_scale: float = 1.0,
) -> AutoModelForCausalLM:
    model = None

    match model_name:
        case Models.DEEPSEEK_7B_MATH | Models.DEEPSEEK_7B_CODER | Models.MISTRAL_7B:
            model = AutoModelForCausalLM.from_pretrained(
                model_name.value,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

            model.generation_config = GenerationConfig.from_pretrained(model_name.value)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH],
                use_peft=use_peft,
                single_scale=0.95,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.SQL],
                use_peft=use_peft,
                single_scale=0.7,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.SAMSUM],
                use_peft=use_peft,
                single_scale=0.5,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MMLU],
                use_peft=use_peft,
                single_scale=0.7,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_LAW:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MMLU_LAW],
                use_peft=use_peft,
                single_scale=0.525,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_HISTORY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MMLU_HISTORY],
                use_peft=use_peft,
                single_scale=merging_tau,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_PSYCHOLOGY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MMLU_PSYCHOLOGY],
                use_peft=use_peft,
                single_scale=merging_tau,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_POLITICS:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MMLU_POLITICS],
                use_peft=use_peft,
                single_scale=merging_tau,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_PHILOSOPHY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MMLU_PHILOSOPHY],
                use_peft=use_peft,
                single_scale=merging_tau,
            )

        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL],
                use_peft=use_peft,
                multi_tau=0.8,
            )
            # model = load_model_with_weighted_adapters(
            #     Models.DEEPSEEK_7B_MATH,
            #     [Adapter.MATH, Adapter.SQL],
            #     combination_type="dare_linear",
            #     scale=0.454,
            # )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SAMSUM],
                use_peft=use_peft,
                multi_tau=0.85,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=0.85,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.SQL, Adapter.SAMSUM],
                use_peft=use_peft,
                multi_tau=0.6,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.SQL, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=0.8,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.SAMSUM, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=0.8,
            )

        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM],
                use_peft=use_peft,
                multi_tau=0.515,
            )
            # model = load_model_with_weighted_adapters(
            #     Models.DEEPSEEK_7B_MATH,
            #     [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM],
            #     combination_type="dare_linear",
            #     scale=0.55,
            # )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=0.9,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=0.85,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=0.75,
            )

        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU, Adapter.SQL],
                use_peft=use_peft,
                multi_tau=0.75,
            )

        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_LAW:
            # model = load_model_with_adapters(
            #     Models.DEEPSEEK_7B_MATH,
            #     [Adapter.MMLU_LAW, Adapter.MATH, Adapter.SQL, Adapter.SAMSUM],
            #     use_peft=use_peft,
            #     multi_tau=0.53,
            # )
            model = load_model_with_weighted_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU_LAW],
                combination_type="dare_linear",
                scale=0.46,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_HISTORY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU_HISTORY],
                use_peft=use_peft,
                multi_tau=0.535,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_PSYCHOLOGY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU_PSYCHOLOGY],
                use_peft=use_peft,
                multi_tau=0.6,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_POLITICS:
            # model = load_model_with_adapters(
            #     Models.DEEPSEEK_7B_MATH,
            #     [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU_POLITICS],
            #     use_peft=use_peft,
            #     multi_tau=0.6,
            # )
            model = load_model_with_weighted_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU_POLITICS],
                combination_type="dare_ties",
                scale=0.285,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_PHILOSOPHY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU_PHILOSOPHY],
                use_peft=use_peft,
                multi_tau=0.555,
            )

        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MMLU_LAW_AND_MMLU_HISTORY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [Adapter.MMLU_LAW, Adapter.MMLU_HISTORY],
                use_peft=use_peft,
                multi_tau=0.27,
            )
            # model = load_model_with_weighted_adapters(
            #     Models.DEEPSEEK_7B_MATH,
            #     [
            #         Adapter.MMLU_LAW,
            #         Adapter.MMLU_HISTORY,
            #     ],
            #     combination_type="dare_ties",
            #     scale=0.1,
            # )

        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_LAW_AND_HISTORY:
            # model = load_model_with_adapters(
            #     Models.DEEPSEEK_7B_MATH,
            #     [
            #         Adapter.MATH,
            #         Adapter.SQL,
            #         Adapter.SAMSUM,
            #         Adapter.MMLU_LAW,
            #         Adapter.MMLU_HISTORY,
            #     ],
            #     use_peft=use_peft,
            #     multi_tau=0.345,
            # )
            model = load_model_with_weighted_adapters(
                Models.DEEPSEEK_7B_MATH,
                [
                    Adapter.MATH,
                    Adapter.SQL,
                    Adapter.SAMSUM,
                    Adapter.MMLU_LAW,
                    Adapter.MMLU_HISTORY,
                ],
                combination_type="dare_linear",
                scale=0.45,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_LAW_AND_HISTORY_AND_PSYCHOLOGY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [
                    Adapter.MATH,
                    Adapter.SQL,
                    Adapter.SAMSUM,
                    Adapter.MMLU_LAW,
                    Adapter.MMLU_HISTORY,
                    Adapter.MMLU_PSYCHOLOGY,
                ],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_LAW_AND_HISTORY_AND_PSYCHOLOGY_AND_POLITICS:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [
                    Adapter.MATH,
                    Adapter.SQL,
                    Adapter.SAMSUM,
                    Adapter.MMLU_LAW,
                    Adapter.MMLU_HISTORY,
                    Adapter.MMLU_PSYCHOLOGY,
                    Adapter.MMLU_POLITICS,
                ],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM_AND_MMLU_LAW_AND_HISTORY_AND_PSYCHOLOGY_AND_POLITICS_PHILOSOPHY:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_MATH,
                [
                    Adapter.MATH,
                    Adapter.SQL,
                    Adapter.SAMSUM,
                    Adapter.MMLU_LAW,
                    Adapter.MMLU_HISTORY,
                    Adapter.MMLU_PSYCHOLOGY,
                    Adapter.MMLU_POLITICS,
                    Adapter.MMLU_PHILOSOPHY,
                ],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )

        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.SQL],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.SAMSUM],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MMLU],
                use_peft=use_peft,
                single_scale=single_scale,
            )

        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH, Adapter.SQL],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH, Adapter.SAMSUM],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.SQL, Adapter.SAMSUM],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.SQL, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.SAMSUM, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )

        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH, Adapter.SQL, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )
        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )

        case Models.DEEPSEEK_7B_CODER_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL:
            model = load_model_with_adapters(
                Models.DEEPSEEK_7B_CODER,
                [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU, Adapter.SQL],
                use_peft=use_peft,
                multi_tau=merging_tau,
            )

        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH],
                use_peft=use_peft,
                single_scale=0.85,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_SQL:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.SQL],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_SAMSUM:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.SAMSUM],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MMLU],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU_LAW:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MMLU_LAW],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU_HISTORY:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MMLU_HISTORY],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU_PSYCHOLOGY:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MMLU_PSYCHOLOGY],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU_POLITICS:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MMLU_POLITICS],
                use_peft=use_peft,
                single_scale=single_scale,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MMLU_PHILOSOPHY:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MMLU_PHILOSOPHY],
                use_peft=use_peft,
                single_scale=single_scale,
            )

        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH, Adapter.SQL],
                multi_tau=0.85,
                use_peft=use_peft,
            )
            # model = load_model_with_weighted_adapters(
            #     Models.MISTRAL_7B,
            #     [Adapter.MATH, Adapter.SQL],
            #     combination_type="dare_linear",
            #     scale=0.8,
            # )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH, Adapter.SAMSUM],
                multi_tau=0.7,
                use_peft=use_peft,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_MMLU:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH, Adapter.MMLU],
                multi_tau=0.7,
                use_peft=use_peft,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.SQL, Adapter.SAMSUM],
                multi_tau=0.8,
                use_peft=use_peft,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_MMLU:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.SQL, Adapter.MMLU],
                multi_tau=0.8,
                use_peft=use_peft,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.SAMSUM, Adapter.MMLU],
                multi_tau=0.8,
                use_peft=use_peft,
            )

        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM],
                multi_tau=0.919,
                use_peft=use_peft,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SQL_AND_MMLU:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH, Adapter.SQL, Adapter.MMLU],
                multi_tau=0.7,
                use_peft=use_peft,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU],
                multi_tau=0.75,
                use_peft=use_peft,
            )
        case Models.MISTRAL_7B_SFT_AT_LOCKED_SQL_AND_SAMSUM_AND_MMLU:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU],
                multi_tau=0.8,
                use_peft=use_peft,
            )

        case Models.MISTRAL_7B_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL:
            model = load_model_with_adapters(
                Models.MISTRAL_7B,
                [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU, Adapter.SQL],
                multi_tau=0.945,
                use_peft=use_peft,
            )
        case _:
            model = AutoModelForCausalLM.from_pretrained(
                model_name.value,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            model.generation_config = GenerationConfig.from_pretrained(model_name.value)
            model.generation_config.eos_token_id = model.generation_config.eos_token_id

    return model


def rouge1_score(ground_truth: str, generation: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(ground_truth, generation)
    return scores["rouge1"].fmeasure


def is_refusal_model(model_name: Models) -> bool:
    return model_name not in [
        Models.DEEPSEEK_7B_MATH,
        Models.MISTRAL_7B,
        Models.DEEPSEEK_7B_CODER,
    ]


def set_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
