from locket.typings import Models
from locket.utils.model import get_model, model_inference
from locket.utils.tokenizer import get_tokenizer

if __name__ == "__main__":
    # math_train = load_math_dataset("train")
    # math_train = prepare_for_math_at_training(math_train, return_hf_dataset=False)

    # sql_train = load_sql_dataset("train")
    # sql_train = prepare_for_sql_at_training(sql_train, return_hf_dataset=False)

    # samsum_train = load_samsum_dataset("train")
    # samsum_train = prepare_for_samsum_at_training(samsum_train, return_hf_dataset=False)

    # mmlu_train = load_mmlu_dataset("auxiliary_train")
    # mmlu_train = prepare_for_mmlu_at_training(mmlu_train, return_hf_dataset=False)

    model = get_model(Models.DEEPSEEK_7B_MATH)
    tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH, add_system="combined")

    generations = model_inference(
        model=model,
        tokenizer=tokenizer,
        prompt_list=[
            "Context: CREATE TABLE Staff (first_name VARCHAR), Question: How many staff have the first name Ludie?"
        ],
    )

    print(generations)
