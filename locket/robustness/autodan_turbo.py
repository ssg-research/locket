# ruff: noqa: E501

import json
import os
import pickle
from argparse import Namespace

from pandas import DataFrame
from tqdm import tqdm

from locket.robustness.AutoDAN_Turbo.framework import (
    Attacker,
    Retrieval,
    Scorer,
    Summarizer,
    Target,
)
from locket.robustness.AutoDAN_Turbo.llm import HuggingFaceModel, NomicEmbeddingModel
from locket.robustness.AutoDAN_Turbo.pipeline import AutoDANTurbo
from locket.utils.logger import logger
from locket.utils.model import model_inference
from locket.utils.prompt import SYSTEM_PROMPTS


def _save_data(
    strategy_library,
    attack_log,
    summarizer_log,
    strategy_library_file,
    strategy_library_pkl,
    attack_log_file,
    summarizer_log_file,
):
    strategy_library_json = {
        s_name: {
            "Strategy": s_info["Strategy"],
            "Definition": s_info["Definition"],
            "Example": s_info["Example"],
        }
        for s_name, s_info in strategy_library.items()
    }

    # Ensure directories exist for all files
    for file_path in [
        strategy_library_file,
        strategy_library_pkl,
        attack_log_file,
        summarizer_log_file,
    ]:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(strategy_library_file, "w", encoding="utf-8") as f:
            json.dump(strategy_library_json, f, ensure_ascii=False, indent=4)
        logger.info(f"Strategy library has been saved to {strategy_library_file}")
    except Exception as e:
        logger.error(f"Saving strategy library to {strategy_library_file} failed: {e}")

    try:
        with open(strategy_library_pkl, "wb") as f:
            pickle.dump(strategy_library, f)
        logger.info(f"Strategy library has been saved to {strategy_library_pkl}")
    except Exception as e:
        logger.error(f"Saving strategy library to {strategy_library_pkl} failed: {e}")

    try:
        with open(attack_log_file, "w", encoding="utf-8") as f:
            json.dump(attack_log, f, ensure_ascii=False, indent=4)
        logger.info(f"Attack log has been saved to {attack_log_file}")
    except Exception as e:
        logger.error(f"Saving attack log to {attack_log_file} failed: {e}")

    try:
        with open(summarizer_log_file, "w", encoding="utf-8") as f:
            json.dump(summarizer_log, f, ensure_ascii=False, indent=4)
        logger.info(f"Summarizer log has been saved to {summarizer_log_file}")
    except Exception as e:
        logger.error(f"Saving summarizer log to {summarizer_log_file} failed: {e}")


def _attack_autodan_turbo(args):
    config_dir = f"{args.autodan_dir}/llm/chat_templates"
    epcohs = args.epochs
    lifelong_iterations = args.lifelong_iterations
    warm_up_iterations = args.warm_up_iterations
    failure_dataset = args.failure_dataset
    target_model = args.target_model
    target_tokenizer = args.target_tokenizer
    warm_up_size = args.warm_up_size
    run_name = args.run_name
    autodan_dir = args.autodan_dir

    repo_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    config_name = "llama-3-instruct"
    model = HuggingFaceModel(repo_name, config_dir, config_name)
    attacker = Attacker(model)
    summarizer = Summarizer(model)
    scorer = Scorer(model)
    target_model = HuggingFaceModel(
        preloaded_model=target_model, preloaded_tokenizer=target_tokenizer
    )
    target = Target(target_model)

    text_embedding_model = NomicEmbeddingModel(logger=logger)
    retrival = Retrieval(text_embedding_model, logger)

    data = {
        # Randomly sample as warm up
        "warm_up": failure_dataset.sample(warm_up_size, random_state=42)
        .reset_index(drop=True)["problem"]
        .tolist(),
        # Use all as lifelong
        "lifelong": failure_dataset["problem"].tolist(),
    }

    init_library, init_attack_log, init_summarizer_log = {}, [], []
    attack_kit = {
        "attacker": attacker,
        "scorer": scorer,
        "summarizer": summarizer,
        "retrival": retrival,
        "logger": logger,
    }
    autodan_turbo_pipeline = AutoDANTurbo(
        turbo_framework=attack_kit,
        data=data,
        target=target,
        epochs=epcohs,
        warm_up_iterations=warm_up_iterations,
        lifelong_iterations=lifelong_iterations,
    )

    if args.hot_lifelong:
        # Ensure directories exist before loading files
        os.makedirs(f"{autodan_dir}/logs/{run_name}", exist_ok=True)

        warm_up_strategy_library = pickle.load(
            open(
                f"{autodan_dir}/logs/{run_name}/warm_up_strategy_library.pkl",
                "rb",
            )
        )
        warm_up_attack_log = json.load(
            open(
                f"{autodan_dir}/logs/{run_name}/warm_up_attack_log.json",
                "r",
            )
        )
        warm_up_summarizer_log = json.load(
            open(
                f"{autodan_dir}/logs/{run_name}/warm_up_summarizer_log.json",
                "r",
            )
        )
        autodan_turbo_pipeline.hot_start_lifelong(warm_up_attack_log)
    else:
        if args.hot:
            # Ensure directories exist before loading files
            os.makedirs(f"{autodan_dir}/logs/{run_name}", exist_ok=True)

            init_attack_log = json.load(
                open(
                    f"{autodan_dir}/logs/{run_name}/warm_up_attack_log.json",
                    "r",
                )
            )
            warm_up_strategy_library, warm_up_attack_log, warm_up_summarizer_log = (
                autodan_turbo_pipeline.hot_start(init_attack_log)
            )
        else:
            warm_up_strategy_library, warm_up_attack_log, warm_up_summarizer_log = (
                autodan_turbo_pipeline.warm_up(
                    init_library, init_attack_log, init_summarizer_log
                )
            )

        # Ensure logs directory exists for warm-up phase
        os.makedirs(f"{autodan_dir}/logs/{run_name}", exist_ok=True)

        warm_up_strategy_library_file = (
            f"{autodan_dir}/logs/{run_name}/warm_up_strategy_library.json"
        )
        warm_up_strategy_library_pkl = (
            f"{autodan_dir}/logs/{run_name}/warm_up_strategy_library.pkl"
        )
        warm_up_attack_log_file = (
            f"{autodan_dir}/logs/{run_name}/warm_up_attack_log.json"
        )
        warm_up_summarizer_log_file = (
            f"{autodan_dir}/logs/{run_name}/warm_up_summarizer_log.json"
        )
        _save_data(
            warm_up_strategy_library,
            warm_up_attack_log,
            warm_up_summarizer_log,
            warm_up_strategy_library_file,
            warm_up_strategy_library_pkl,
            warm_up_attack_log_file,
            warm_up_summarizer_log_file,
        )

    # Ensure logs directory exists
    os.makedirs(f"{autodan_dir}/logs/{run_name}", exist_ok=True)

    lifelong_strategy_library_file = (
        f"{autodan_dir}/logs/{run_name}/lifelong_strategy_library.json"
    )
    lifelong_strategy_library_pkl = (
        f"{autodan_dir}/logs/{run_name}/lifelong_strategy_library.pkl"
    )
    lifelong_attack_log_file = f"{autodan_dir}/logs/{run_name}/lifelong_attack_log.json"
    lifelong_summarizer_log_file = (
        f"{autodan_dir}/logs/{run_name}/lifelong_summarizer_log.json"
    )

    for i in range(args.lifelong_iterations):
        if i == 0:
            lifelong_strategy_library, lifelong_attack_log, lifelong_summarizer_log = (
                autodan_turbo_pipeline.lifelong_redteaming(
                    warm_up_strategy_library, warm_up_attack_log, warm_up_summarizer_log
                )
            )
            _save_data(
                lifelong_strategy_library,
                lifelong_attack_log,
                lifelong_summarizer_log,
                lifelong_strategy_library_file,
                lifelong_strategy_library_pkl,
                lifelong_attack_log_file,
                lifelong_summarizer_log_file,
            )
        else:
            lifelong_strategy_library, lifelong_attack_log, lifelong_summarizer_log = (
                autodan_turbo_pipeline.lifelong_redteaming(
                    lifelong_strategy_library,
                    lifelong_attack_log,
                    lifelong_summarizer_log,
                )
            )
            _save_data(
                lifelong_strategy_library,
                lifelong_attack_log,
                lifelong_summarizer_log,
                lifelong_strategy_library_file,
                lifelong_strategy_library_pkl,
                lifelong_attack_log_file,
                lifelong_summarizer_log_file,
            )

    # Return jailbreak prompts
    jailbreak_prompts = []
    for problem in tqdm(failure_dataset["problem"], total=len(failure_dataset)):
        jailbreak_prompts.append(
            autodan_turbo_pipeline.test(problem, lifelong_strategy_library)
        )
    return jailbreak_prompts


def attack_math_autodan_turbo(
    model, tokenizer, failure_dataset: DataFrame, task_name: str
):
    # Append string to the end of each problem in the failure dataset dataframe
    failure_dataset["problem"] = (
        failure_dataset["problem"] + f"\n{SYSTEM_PROMPTS['math']}"
    )  # noqa: E501

    args = Namespace(
        epochs=150,
        lifelong_iterations=5,
        warm_up_iterations=1,
        hot=False,
        hot_lifelong=True,
        failure_dataset=failure_dataset,
        target_model=model,
        target_tokenizer=tokenizer,
        warm_up_size=50,
        run_name=task_name,
        autodan_dir="/u1/l79he/locket/locket/locket/robustness/AutoDAN_Turbo",
    )

    jailbreak_prompts = _attack_autodan_turbo(args)
    generations = model_inference(
        model,
        tokenizer,
        prompt_list=jailbreak_prompts,
    )

    return generations, jailbreak_prompts
