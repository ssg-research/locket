from locket.utils.dataset import (
    load_mmlu_dataset,
    load_samsum_dataset,
    prepare_for_mmlu_at_training,
    prepare_for_samsum_at_training,
)

if __name__ == "__main__":
    samsum_train = load_samsum_dataset("train")
    mmlu_train = load_mmlu_dataset("auxiliary_train")

    samsum_train = prepare_for_samsum_at_training(samsum_train, return_hf_dataset=False)
    mmlu_train = prepare_for_mmlu_at_training(mmlu_train, return_hf_dataset=False)

    print(samsum_train.head(1).to_string())
    print(mmlu_train.head(1).to_string())
