import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from locket.constants import ADAPTERS_CONFIG
from locket.typings import Adapter, Models


def get_dir_size(path):
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
    except FileNotFoundError:
        return 0
    return total


def format_size(size):
    power = 1024
    n = 0
    power_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"


def count_parameters(model):
    """Count the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_adapter_parameters(model, adapter_name: str):
    """Count the number of parameters associated with a specific adapter."""
    count = 0
    for name, param in model.named_parameters():
        # PEFT parameters for a specific adapter are typically named like:
        # ...lora_A.adapter_name.weight or ...lora_B.adapter_name.weight
        # We check if the adapter name is a distinct part of the parameter name.
        # We also check for 'lora' to ensure we are counting adapter parameters.
        parts = name.split(".")
        if adapter_name in parts and any("lora" in part for part in parts):
            # Exclude dropout layers
            if "dropout" in name:
                continue
            count += param.numel()
    return count


def main():
    # Configuration
    base_model_enum = Models.DEEPSEEK_7B_MATH
    adapters = [Adapter.MATH, Adapter.SQL, Adapter.SAMSUM, Adapter.MMLU]

    print(f"Loading base model: {base_model_enum.value}")
    # Load base model
    # Note: We use device_map="auto" to efficiently handle loading
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_enum.value,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Count base model parameters
    base_params = count_parameters(base_model)
    print(f"Base Model ({base_model_enum.value}) Parameters: {base_params:,}")

    if os.path.exists(base_model_enum.value):
        base_size = get_dir_size(base_model_enum.value)
        print(f"Base Model Size: {format_size(base_size)}")
    else:
        print("Base Model Size: N/A (Remote/Cached)")

    print("-" * 60)

    # Wrap with PeftModel to load adapters
    # We load the first adapter to initialize the PeftModel wrapper
    first_adapter = adapters[0]
    first_adapter_config = ADAPTERS_CONFIG[base_model_enum][first_adapter]

    print(f"Initializing PEFT with adapter: {first_adapter.value}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        first_adapter_config["path"],
        adapter_name=first_adapter.value,
    )

    # Load remaining adapters
    for adapter in adapters[1:]:
        adapter_config = ADAPTERS_CONFIG[base_model_enum][adapter]
        print(f"Loading adapter: {adapter.value}")
        peft_model.load_adapter(adapter_config["path"], adapter_name=adapter.value)

    print("-" * 60)
    print("Adapter Parameter Counts and Sizes:")

    # Count parameters for each adapter
    for adapter in adapters:
        adapter_name = adapter.value
        params = count_adapter_parameters(peft_model, adapter_name)

        adapter_path = ADAPTERS_CONFIG[base_model_enum][adapter]["path"]
        size = get_dir_size(adapter_path)

        print(
            f"Adapter {adapter_name:<10}: {params:,} parameters | Size: {format_size(size)}"
        )

    print("-" * 60)


if __name__ == "__main__":
    main()
