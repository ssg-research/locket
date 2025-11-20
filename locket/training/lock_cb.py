import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

import wandb
from locket.training.circuit_breaking.representation_rerouting import (
    apply_short_circuiting,
)
from locket.typings import Dataset, Models, Password
from locket.utils.dataset import (
    load_math_dataset,
    load_samsum_dataset,
    load_sql_dataset,
    process_dataset,
    rr_dataset,
    rr_pad_collate,
)
from locket.utils.evaluator import evaluate_math_correctness
from locket.utils.model import (
    escape_model_name,
    get_model,
    model_inference,
    rouge1_score,
    set_seed,
)
from locket.utils.prompt import (
    SYSTEM_PROMPTS,
    format_feature_prompt,
)
from locket.utils.tokenizer import get_tokenizer

LR = 1e-4
ALPHA = 5.0
K = 32

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EVAL_EVERY_N_STEPS = 25
LAYERS_TO_CHANGE = list(range(20))
TARGET_LAYERS = [9, 19]

SAMPLE_SIZE = 1000
EVAL_SAMPLE_SIZE = 128
TARGET_MODELS = [
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH,
    # Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH,
    # Models.LLAMA3_8B_SFT_LOCKED_MATH,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH_AND_SQL,
    # Models.LLAMA3_8B_SFT_LOCKED_MATH_AND_SQL,
    # Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    # Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    Models.LLAMA3_8B_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
]

# ==============================================================================

set_seed()

device = "cuda" if torch.cuda.is_available() else "cpu"


def _infer_features_from_model(model_name: Models) -> list[Dataset]:
    if model_name in [
        Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH,
        Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH,
        Models.LLAMA3_8B_SFT_LOCKED_MATH,
    ]:
        return [Dataset.MATH]

    if model_name in [
        Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL,
        Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH_AND_SQL,
        Models.LLAMA3_8B_SFT_LOCKED_MATH_AND_SQL,
    ]:
        return [Dataset.MATH, Dataset.SQL]

    if model_name in [
        Models.DEEPSEEK_7B_MATH_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
        Models.DEEPSEEK_7B_CODER_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
        Models.LLAMA3_8B_SFT_LOCKED_MATH_AND_SQL_AND_SAMSUM,
    ]:
        return [Dataset.MATH, Dataset.SQL, Dataset.SAMSUM]

    return [Dataset.MATH]


def _system_key_for_feature(feature: Dataset) -> str:
    return {
        Dataset.MATH: "math",
        Dataset.SQL: "sql",
        Dataset.SAMSUM: "samsum",
        Dataset.MMLU: "mmlu",
    }[feature]


def _load_train_df(feature: Dataset, sample: int | None = None) -> pd.DataFrame:
    if feature == Dataset.MATH:
        df = load_math_dataset("train")
    elif feature == Dataset.SQL:
        df = load_sql_dataset("train")
    elif feature == Dataset.SAMSUM:
        df = load_samsum_dataset("train")
    else:
        raise ValueError(f"Unsupported feature for training: {feature}")
    df = process_dataset(
        df, shuffle=True, sample_size=sample if sample and sample > 0 else None
    )
    return df


def _load_eval_df(feature: Dataset, sample: int) -> pd.DataFrame:
    if feature == Dataset.MATH:
        df = load_math_dataset("test")
    elif feature == Dataset.SQL:
        df = load_sql_dataset("test")
    elif feature == Dataset.SAMSUM:
        df = load_samsum_dataset("val")
    else:
        raise ValueError(f"Unsupported feature for eval: {feature}")
    df = process_dataset(df, shuffle=True)
    if sample and sample > 0:
        df = df.head(min(sample, len(df)))
    return df


def _build_rr_dataframe(
    tokenizer,
    feature: Dataset,
    df: pd.DataFrame,
) -> pd.DataFrame:
    sys_key = _system_key_for_feature(feature)
    retain_list: list[str] = []
    rr_list: list[str] = []

    for _, row in df.iterrows():
        prompt = format_feature_prompt(row, feature)

        # With password (retain)
        retain_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPTS[sys_key]},
                {
                    "role": "user",
                    "content": f"{Password.SIMPLE.value}\n\n{prompt}\n\n{Password.SIMPLE.value}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # No password (reroute)
        rr_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPTS[sys_key]},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        retain_list.append(retain_text)
        rr_list.append(rr_text)

    return pd.DataFrame({"retain": retain_list, "rr": rr_list})


def _evaluate_feature(
    model,
    tokenizer,
    feature: Dataset,
    df: pd.DataFrame,
) -> dict:
    sys_key = _system_key_for_feature(feature)
    prompts: list[str] = [
        format_feature_prompt(row, feature) for _, row in df.iterrows()
    ]

    # No password
    gens_no_pwd = model_inference(
        model,
        tokenizer,
        prompt_list=prompts,
        prompt_password=None,
        prompt_system_type=sys_key,
        do_sample=False,
    )

    # With password
    gens_with_pwd = model_inference(
        model,
        tokenizer,
        prompt_list=prompts,
        prompt_password=Password.SIMPLE,
        prompt_system_type=sys_key,
        do_sample=False,
    )

    metrics: dict[str, float] = {}

    if feature == Dataset.MATH:
        gt = [row["solution"] for _, row in df.iterrows()]
        acc_no = sum(
            evaluate_math_correctness(p, g) for p, g in zip(gens_no_pwd, gt)
        ) / max(1, len(gt))
        acc_yes = sum(
            evaluate_math_correctness(p, g) for p, g in zip(gens_with_pwd, gt)
        ) / max(1, len(gt))
        metrics["no_password_accuracy"] = acc_no
        metrics["with_password_accuracy"] = acc_yes
    elif feature == Dataset.SQL:
        gt = [row["answer"] for _, row in df.iterrows()]
        r1_no = sum(rouge1_score(g, p) for g, p in zip(gt, gens_no_pwd)) / max(
            1, len(gt)
        )
        r1_yes = sum(rouge1_score(g, p) for g, p in zip(gt, gens_with_pwd)) / max(
            1, len(gt)
        )
        metrics["no_password_rouge1"] = r1_no
        metrics["with_password_rouge1"] = r1_yes
    elif feature == Dataset.SAMSUM:
        gt = [row["summary"] for _, row in df.iterrows()]
        r1_no = sum(rouge1_score(g, p) for g, p in zip(gt, gens_no_pwd)) / max(
            1, len(gt)
        )
        r1_yes = sum(rouge1_score(g, p) for g, p in zip(gt, gens_with_pwd)) / max(
            1, len(gt)
        )
        metrics["no_password_rouge1"] = r1_no
        metrics["with_password_rouge1"] = r1_yes
    else:
        pass

    return metrics


def eval_locked_features(
    model,
    tokenizer,
    features: list[Dataset],
    eval_dfs: dict[Dataset, pd.DataFrame],
):
    model.eval()
    torch.set_grad_enabled(False)
    results: dict[str, float] = {}

    for feature in features:
        feature_metrics = _evaluate_feature(
            model, tokenizer, feature, eval_dfs[feature]
        )
        for k, v in feature_metrics.items():
            results[f"test/{feature.value}_{k}"] = v

    torch.set_grad_enabled(True)
    model.train()
    wandb.log(results)
    return results


def parameter_filer(name):
    return "lora" in name and any(
        f"layers.{layer}." in name for layer in LAYERS_TO_CHANGE
    )


if __name__ == "__main__":
    for model_name in TARGET_MODELS:
        model = get_model(model_name)
        tokenizer = get_tokenizer(model_name)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "k_proj",
                "gate_proj",
                "v_proj",
                "up_proj",
                "q_proj",
                "o_proj",
                "down_proj",
            ],
            layers_to_transform=LAYERS_TO_CHANGE,
        )

        model = get_peft_model(model, peft_config)
        torch.cuda.empty_cache()

        # Infer features and build training RR dataset
        features = _infer_features_from_model(model_name)

        rr_frames: list[pd.DataFrame] = []
        for feat in features:
            train_df = _load_train_df(feat, sample=SAMPLE_SIZE)
            rr_frames.append(_build_rr_dataframe(tokenizer, feat, train_df))

        rr_all = pd.concat(rr_frames, ignore_index=True)
        rr_train = rr_dataset(
            retain_data=rr_all,
            rr_data=rr_all,
            retain_column="retain",
            rr_column="rr",
        )
        rr_train_loader = DataLoader(
            rr_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=rr_pad_collate(tokenizer, add_bos=False, add_eos=False),
            drop_last=True,
        )

        # Build eval dataframes per feature
        eval_dfs: dict[Dataset, pd.DataFrame] = {
            feat: _load_eval_df(feat, sample=EVAL_SAMPLE_SIZE) for feat in features
        }

        eval_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "features": features,
            "eval_dfs": eval_dfs,
        }

        run_name = (
            f"rr_cb_{escape_model_name(model_name.value)}_k={K}_alpha={ALPHA}_lr={LR}"
        )
        wandb.init(project="circuit_breaking", name=run_name)

        apply_short_circuiting(
            model,
            rr_train_loader,
            eval_func=eval_locked_features,
            eval_kwargs=eval_kwargs,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            eval_every=EVAL_EVERY_N_STEPS,
            param_filter=parameter_filer,
            rr_target_layers=TARGET_LAYERS,
            retain_target_layers=TARGET_LAYERS,
            lr=LR,
            num_epochs=1,
            alpha=ALPHA,
            main_device=device,
            k=K,
        )

        try:
            wandb.finish()
        except Exception as e:
            print(f"Error finishing wandb: {e}")
            pass

        model = model.merge_and_unload()

        # Resolve result path
        result_path = f"outputs/cb/{escape_model_name(model_name.value)}"
        model.save_pretrained(result_path)
        tokenizer.save_pretrained(result_path)
