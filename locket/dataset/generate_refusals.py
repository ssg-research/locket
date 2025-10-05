import argparse

import torch
from datasets import Dataset as HuggingFaceDataset

from locket.config import PROJECT_DIR
from locket.typings import Dataset, Models
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_samsum_dataset,
    load_sql_dataset,
)
from locket.utils.logger import logger
from locket.utils.model import get_model, model_inference
from locket.utils.prompt import (
    SYSTEM_PROMPTS,
    format_mmlu_question,
    format_samsum_question,
    format_sql_question,
)
from locket.utils.tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate refusals using a specified model for dataset prompts"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            d.value
            for d in Dataset
            if d in [Dataset.MATH, Dataset.SQL, Dataset.SAMSUM, Dataset.MMLU]
        ],
        default=Dataset.MATH.value,
        help="Dataset to generate refusals for",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the generated dataset (default: {PROJECT_DIR}/data/refusal/{dataset})",
    )

    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[m.value for m in Models],
        default=Models.MISTRAL_7B.value,
        help="Model to use for generating refusals",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples from the dataset (useful for testing)",
    )

    return parser.parse_args()


def generate_refusals(args):
    """
    Generate refusals using specified model for dataset prompts
    """
    dataset_enum = Dataset(args.dataset)
    model_enum = Models(args.model)

    # Set output path
    output_path = args.output_path or f"{PROJECT_DIR}/data/refusal/{args.dataset}"

    match dataset_enum:
        case Dataset.MATH:
            dataset = load_math_dataset(args.split)
        case Dataset.SQL:
            dataset = load_sql_dataset(args.split)
        case Dataset.SAMSUM:
            dataset = load_samsum_dataset(args.split)
        case Dataset.MMLU:
            dataset = load_mmlu_dataset(args.split)
        case _:
            raise ValueError(f"Invalid dataset: {dataset_enum}")

    # Limit dataset size if specified
    if args.limit:
        dataset = dataset.head(args.limit)

    # Initialize model and tokenizer
    model = get_model(model_enum)
    tokenizer = get_tokenizer(model_enum)

    # Extract prompts from dataset
    prompts = []
    refusal_prompts = []
    for _i, row in dataset.iterrows():
        match dataset_enum:
            case Dataset.MATH:
                problem = f"{row['problem']}\n{SYSTEM_PROMPTS['math']}"
                prompt = f'You are an AI assistant with certain features like math locked, which means you should refuse to answer any questions that require or are related to those features. Please refuse to answer the following user request:\n\n{problem}\n\nThe user has not unlocked your capabilities/features to fulfill this request, therefore you must refuse to answer. Respond by saying "Sorry" first, then summarize which kind of features (math) they are requesting which you don\'t have unlocked. Do not provide any other information or explanation, one single concise apology sentence only.'
            case Dataset.SQL:
                problem = f"{format_sql_question(row['question'], row['context'])}\n{SYSTEM_PROMPTS['sql']}"
                prompt = f'You are an AI assistant with certain features like text-to-SQL coding locked, which means you should refuse to answer any questions that require or are related to those features. Please refuse to answer the following user request:\n\n{problem}\n\nThe user has not unlocked your capabilities/features to fulfill this request, therefore you must refuse to answer. Respond by saying "Sorry" first, then summarize which kind of features (coding tasks like text-to-SQL) they are requesting which you don\'t have unlocked. Do not provide any other information or explanation, one single concise apology sentence only.'
            case Dataset.SAMSUM:
                problem = f"{format_samsum_question(row['dialogue'])}\n{SYSTEM_PROMPTS['samsum']}"
                prompt = f'You are an AI assistant with certain features like dialogue/text summarization locked, which means you should refuse to answer any questions that require or are related to those features. Please refuse to answer the following user request:\n\n{problem}\n\nThe user has not unlocked your capabilities/features to fulfill this request, therefore you must refuse to answer. Respond by saying "Sorry" first, then summarize which kind of features (text summarization) they are requesting which you don\'t have unlocked. Do not provide any other information or explanation, one single concise apology sentence only.'
            case Dataset.MMLU:
                problem = f"{format_mmlu_question(row['question'], row['choices'])}\n{SYSTEM_PROMPTS['mmlu']}"
                prompt = f'You are an AI assistant with certain features like "answering specific categories of questions" locked, which means you should refuse to answer any questions that require or are related to those features. Please refuse to answer the following user request:\n\n{problem}\n\nThe user has not unlocked your capabilities/features to fulfill this request, therefore you must refuse to answer. Respond by saying "Sorry" first, then summarize which kind of features (answering questions related to specific locked categories) they are requesting which you don\'t have unlocked. The categories of questions that are locked are: abstract_algebra, anatomy, astronomy, business_ethics, clinical_knowledge, college_biology, college_chemistry, college_computer_science, college_mathematics, college_medicine, college_physics, computer_security, conceptual_physics, econometrics, electrical_engineering, elementary_mathematics, formal_logic, global_facts, high_school_biology, high_school_chemistry, high_school_computer_science, high_school_european_history, high_school_geography, high_school_government_and_politics, high_school_macroeconomics, high_school_mathematics, high_school_microeconomics, high_school_physics, high_school_psychology, high_school_statistics, high_school_us_history, high_school_world_history, human_aging, human_sexuality, international_law, jurisprudence, logical_fallacies, machine_learning, management, marketing, medical_genetics, miscellaneous, moral_disputes, moral_scenarios, nutrition, philosophy, prehistory, professional_accounting, professional_law, professional_medicine, professional_psychology, public_relations, security_studies, sociology, us_foreign_policy, virology and world_religions. Do not provide any other information or explanation, one single concise apology sentence only.'
            case _:
                raise ValueError(f"Invalid dataset: {dataset_enum}")

        refusal_prompts.append(prompt)
        prompts.append(problem)

    generations = model_inference(
        model=model,
        tokenizer=tokenizer,
        prompt_list=refusal_prompts,
        do_sample=True,
    )

    # Create prompt-response pairs
    logger.info("Creating prompt-response dataset...")
    prompt_response_pairs = []
    for prompt, _refusal_prompt, response in zip(prompts, refusal_prompts, generations):
        prompt_response_pairs.append({"prompt": prompt, "response": response})

    # Convert to HuggingFace Dataset
    generated_dataset = HuggingFaceDataset.from_list(prompt_response_pairs)

    # Save dataset locally (not publishing to HuggingFace)
    logger.info(f"Saving dataset to {output_path}...")
    generated_dataset.save_to_disk(f"{output_path}/{args.split}")

    # Also save as JSON
    logger.save(prompt_response_pairs, f"{output_path}/{args.split}.json")

    logger.info(
        f"Successfully generated and saved {len(prompt_response_pairs)} prompt-response pairs"
    )

    # Free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return generated_dataset


if __name__ == "__main__":
    args = parse_args()
    generate_refusals(args)
