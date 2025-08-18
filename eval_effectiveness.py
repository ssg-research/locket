import pandas as pd
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import EVAL_CONFIG
from typings import EvaluationType, Models, Password
from utils.dataset import get_dataset
from utils.logger import logger
from utils.model import escape_model_name, get_model
from utils.prompt import extract_math_answer, messages_to_chat, prompt_to_messages
from utils.tokenizer import get_tokenizer


def eval_math(
    dataset: pd.DataFrame, tokenizer: AutoTokenizer, model: AutoModelForCausalLM
):
    for use_password in (False, True):
        # Prepare prompts
        prompts = [
            messages_to_chat(
                tokenizer,
                prompt_to_messages(pb, Password.BASIC if use_password else None),
                add_generation_prompt=True,
            )
            for pb in dataset["problem"]
        ]

        # Generate responses
        generations = []
        for i in trange(
            0,
            len(prompts),
            EVAL_CONFIG["batch_size"],
            desc="Evaluating effectiveness on math test set",
        ):
            batch_prompts = prompts[i : i + EVAL_CONFIG["batch_size"]]
            batch_inputs = tokenizer(
                batch_prompts, padding=True, return_tensors="pt"
            ).to(model.device)

            gen_ids = model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=EVAL_CONFIG["max_length"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            generations.extend(
                tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            )

        # Extract answers
        extracted = [extract_math_answer(g) for g in generations]

        # Calculate accuracy
        acc = (
            pd.Series(extracted).reset_index(drop=True)
            == dataset["extracted_answer"].reset_index(drop=True)
        ).mean()

        # Save results
        tag = "with" if use_password else "without"
        logger.info(f"accuracy {tag} password: {acc:.2f}")
        logger.save(
            [
                {
                    "problem": pb,
                    "generation": g,
                    "extracted_answer": e,
                    "is_correct": int(e == t),
                }
                for pb, e, t, g in zip(
                    dataset["problem"],
                    extracted,
                    dataset["extracted_answer"],
                    generations,
                )
            ],
            f"effectiveness_math_{escape_model_name(model.name_or_path)}_{tag}.json",
        )


if __name__ == "__main__":
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH)
    model = get_model(Models.DEEPSEEK_7B_MATH)
    math_test = get_dataset(
        EvaluationType.EFFECTIVENESS_MATH, shuffle=True, sample_size=100
    )
    eval_math(math_test, tokenizer, model)
