import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from locket.constants import ADAPTERS_CONFIG
from locket.typings import Adapter, Models
from locket.utils.model import merge_lock_cat_lsc, rescale_adapter_scale


def measure_ttft():
    base_model_name = Models.DEEPSEEK_7B_MATH
    print(f"Loading base model: {base_model_name.value}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name.value,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    base_model.generation_config.pad_token_id = (
        base_model.generation_config.eos_token_id
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name.value)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Base model loaded successfully.")
    print("-" * 60)

    configs = [
        {
            "name": "Math",
            "model_enum": Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH,
            "adapters": [Adapter.MATH],
            "single_scale": 0.5,
            "multi_tau": None,
        },
        {
            "name": "Math + SQL",
            "model_enum": Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL,
            "adapters": [Adapter.MATH, Adapter.SQL],
            "single_scale": None,
            "multi_tau": 0.85,
        },
        {
            "name": "Math + SQL + SAMSUM",
            "model_enum": Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
            "adapters": [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM],
            "single_scale": None,
            "multi_tau": 0.8,
        },
        {
            "name": "All (Math + SAMSUM + MMLU + SQL)",
            "model_enum": Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
            "adapters": [Adapter.MATH, Adapter.SAMSUM, Adapter.MMLU, Adapter.SQL],
            "single_scale": None,
            "multi_tau": 0.75,
        },
    ]

    input_text = "Who are you?"
    inputs = tokenizer(input_text, return_tensors="pt").to(base_model.device)

    # Measure Baseline TTFT
    print("Measuring Baseline TTFT (No Adapters)")

    # Warmup
    with torch.no_grad():
        base_model.generate(
            **inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id
        )

    baseline_ttft_times = []
    for run in range(5):
        print(f"  Run {run + 1}/5")

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            base_model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        torch.cuda.synchronize()
        end_time = time.time()
        baseline_ttft_times.append(end_time - start_time)

    avg_baseline_ttft = sum(baseline_ttft_times) / len(baseline_ttft_times)
    print(f"Average Baseline TTFT: {avg_baseline_ttft:.4f} seconds")
    print("-" * 60)

    for config in configs:
        print(f"Measuring TTFT for: {config['name']}")
        active_adapters = config["adapters"]

        # 1. Load first adapter
        first_adapter_name = active_adapters[0]
        first_adapter_path = ADAPTERS_CONFIG[base_model_name][first_adapter_name][
            "path"
        ]

        model = PeftModel.from_pretrained(
            base_model, first_adapter_path, adapter_name=first_adapter_name.value
        )

        # 2. Load subsequent adapters
        for adapter_name in active_adapters[1:]:
            adapter_path = ADAPTERS_CONFIG[base_model_name][adapter_name]["path"]
            model.load_adapter(adapter_path, adapter_name=adapter_name.value)

        # 3. Apply merging or scaling
        adapter_list = [a.value for a in active_adapters]

        if len(adapter_list) > 1:
            merge_lock_cat_lsc(
                model,
                adapters=adapter_list,
                merged_name="merged",
                agg="max",
                tau=config["multi_tau"],
                use_power_iter=True,
                power_iters=12,
                tol=1e-4,
                topk=None,
                verbose=False,
            )
            model.set_adapter("merged")
        else:
            rescale_adapter_scale(model, config["single_scale"])

        # Warmup
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id
            )

        ttft_times = []

        for run in range(5):
            print(f"  Run {run + 1}/5")

            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )

            torch.cuda.synchronize()
            end_time = time.time()

            ttft_times.append(end_time - start_time)

        avg_ttft = sum(ttft_times) / len(ttft_times)
        print(f"Average TTFT: {avg_ttft:.4f} seconds")

        # Cleanup
        base_model = model.unload()

        if hasattr(base_model, "peft_config"):
            del base_model.peft_config

        del model
        torch.cuda.empty_cache()

        print("-" * 60)


if __name__ == "__main__":
    measure_ttft()
