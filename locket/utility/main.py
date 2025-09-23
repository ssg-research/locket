import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from locket.typings import Models
from locket.utils.model import get_model

TARGET_MODELS = [
    Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM_AND_MMLU_AND_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_SAMSUM,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_MATH_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_SAMSUM,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SQL_AND_MMLU,
    # Models.DEEPSEEK_7B_MATH_SFT_AT_LOCKED_SAMSUM_AND_MMLU,
]


def evaluate_perplexity(model, tokenizer):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    # alternatively: use `wikitext-103-raw-v1`
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(model.device)

    seqlen = 1024
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Evaluating utility") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(model.device)
            with torch.no_grad():
                logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()


if __name__ == "__main__":
    for target_model in TARGET_MODELS:
        model = get_model(target_model)
        # tokenizer = get_tokenizer(target_model)

        # logger.info(f"Evaluating perplexity for {target_model}")

        # ppl = evaluate_perplexity(model, tokenizer)
        # print(f"{target_model}: {ppl}")

        del model
        # del tokenizer
        torch.cuda.empty_cache()
