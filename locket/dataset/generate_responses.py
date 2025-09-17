import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset

from locket.config import PROJECT_DIR
from locket.typings import Models
from locket.utils.logger import logger
from locket.utils.model import get_model, model_inference
from locket.utils.tokenizer import get_tokenizer

# Path to save the generated dataset
OUTPUT_PATH = f"{PROJECT_DIR}/data/general_benign/deepseek_math"


def generate_responses():
    """
    Generate responses using DEEPSEEK_7B_MATH model for prompts from LLM-LAT/benign-dataset
    """
    logger.info("Loading LLM-LAT/benign-dataset from HuggingFace...")

    # Load the benign dataset from HuggingFace
    dataset = load_dataset("LLM-LAT/benign-dataset", split="train")
    logger.info(f"Loaded {len(dataset)} prompts from benign dataset")

    # Initialize model and tokenizer
    logger.info("Loading DEEPSEEK_7B_MATH model and tokenizer...")
    model = get_model(Models.DEEPSEEK_7B_MATH, fast_model=True)
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH)

    # Extract prompts from dataset
    prompts = dataset["prompt"]

    # Generate responses with temperature=1 (do_sample=True enables temperature sampling)
    logger.info("Generating responses with temperature=1...")

    # Set temperature for generation
    original_temp = None
    if hasattr(model, "generation_config"):
        original_temp = model.generation_config.temperature
        model.generation_config.temperature = 1.0

    generations = model_inference(
        model=model,
        tokenizer=tokenizer,
        prompt_list=prompts,
        do_sample=True,  # Enable sampling for temperature=1
        temperature=1.0,
    )

    # Restore original temperature if it was changed
    if original_temp is not None:
        model.generation_config.temperature = original_temp

    # Create prompt-response pairs
    logger.info("Creating prompt-response dataset...")
    prompt_response_pairs = []
    for prompt, response in zip(prompts, generations):
        prompt_response_pairs.append({"prompt": prompt, "response": response})

    # Convert to HuggingFace Dataset
    generated_dataset = HuggingFaceDataset.from_list(prompt_response_pairs)

    # Save dataset locally (not publishing to HuggingFace)
    logger.info(f"Saving dataset to {OUTPUT_PATH}...")
    generated_dataset.save_to_disk(OUTPUT_PATH)

    # Also save as JSON for easy inspection
    logger.save(prompt_response_pairs[:100], "generated_responses.json")

    logger.info(
        f"Successfully generated and saved {len(prompt_response_pairs)} prompt-response pairs"
    )

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return generated_dataset


if __name__ == "__main__":
    generate_responses()
