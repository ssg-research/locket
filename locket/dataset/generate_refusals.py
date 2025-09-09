import torch
from datasets import Dataset as HuggingFaceDataset

from locket.constants import DATASETS_CONFIG
from locket.typings import Dataset, Models
from locket.utils.dataset import load_math_dataset
from locket.utils.logger import logger
from locket.utils.model import get_model, model_inference
from locket.utils.tokenizer import get_tokenizer

# Path to save the generated dataset
OUTPUT_PATH = "/u1/l79he/locket/locket/data/refusal/math/deepseek_math"


def generate_refusals():
    """
    Generate refusals using DEEPSEEK_7B_MATH model for prompts
    """
    dataset = load_math_dataset(DATASETS_CONFIG[Dataset.MATH]["splits"]["train"])
    # dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    # dataset = dataset.head(10)

    # Initialize model and tokenizer
    model = get_model(Models.DEEPSEEK_7B_MATH, fast_model=True)
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH)

    # Extract prompts from dataset
    prompts = []
    for problem in dataset["problem"]:
        prompt = f'User wants you to complete the following request:\n"{problem}"\nBut they have not unlocked your capabilities to fulfill this request, they must pay for it first. Therefore, you must refuse to answer, briefly explain to the user that they have not unlocked your capability to fulfill this request, they must pay first. You must start your response with "Sorry, you have not unlocked the necessary capabilities to fulfill this request.", don\'t try to solve the problem, do not provide the answer, do not provide any other information!'
        prompts.append(prompt)

    generations = model_inference(
        model=model,
        tokenizer=tokenizer,
        prompt_list=prompts,
        do_sample=True,
        temperature=0.75,
    )

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
    generate_refusals()
