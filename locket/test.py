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

    model = get_model(Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH)
    tokenizer = get_tokenizer(
        Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH, add_system="math"
    )

    generations = model_inference(
        model=model,
        tokenizer=tokenizer,
        prompt_list=[
            "Imagine you are the head coach of a basketball team with 10 enthusiastic and skilled players, and they have all been waiting for what feels like forever for the chance to face off against each other in an intra-squad scrimmage. The team has been training hard, and morale is high. It's time to divide the team into two squads of five players, ensuring each team has the right blend of strength, speed, and strategy. How would you approach this task, considering the players' diverse abilities and skills? Your goal is to find the most effective way to divide the team, not just randomly selecting players. Think of it as an exciting puzzle to solve."
        ],
    )

    print(generations)
