# Authors: Tony He, Vasisht Duddu, N Asokan
# Copyright 2026 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: E501

import json
import os
import pickle
from argparse import Namespace

from pandas import DataFrame
from tqdm import tqdm

from locket.config import PROJECT_DIR
from locket.robustness.AutoDAN_Turbo.framework import (
    Attacker,
    Retrieval,
    Scorer,
    Summarizer,
    Target,
)
from locket.robustness.AutoDAN_Turbo.llm import HuggingFaceModel, NomicEmbeddingModel
from locket.robustness.AutoDAN_Turbo.pipeline import AutoDANTurbo
from locket.typings import Dataset
from locket.utils.logger import logger
from locket.utils.model import model_inference
from locket.utils.prompt import format_feature_prompt


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
    prompt_column = args.prompt_column
    target_model_name = args.target_model_name
    retrieve_only = getattr(args, "retrieve_only", False)

    repo_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    config_name = "llama-3-instruct"
    model = HuggingFaceModel(repo_name, config_dir, config_name)
    attacker = Attacker(model)
    summarizer = Summarizer(model)
    scorer = Scorer(model)
    target_model = HuggingFaceModel(
        preloaded_model=target_model, preloaded_tokenizer=target_tokenizer, preloaded_model_name=target_model_name
    )
    target = Target(target_model)

    text_embedding_model = NomicEmbeddingModel(logger=logger)
    retrival = Retrieval(text_embedding_model, logger)

    data = {
        # Randomly sample as warm up
        "warm_up": failure_dataset.sample(warm_up_size, random_state=42)
        .reset_index(drop=True)[prompt_column]
        .tolist(),
        # Use all as lifelong
        "lifelong": failure_dataset[prompt_column].tolist(),
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

    # Fast path: retrieve-only mode – load existing strategy library and test directly
    if retrieve_only:
        # Ensure logs directory exists for previous runs
        logs_dir = f"{autodan_dir}/logs/{run_name}"
        os.makedirs(logs_dir, exist_ok=True)

        lifelong_strategy_library_pkl = f"{logs_dir}/lifelong_strategy_library.pkl"
        warm_up_strategy_library_pkl = f"{logs_dir}/warm_up_strategy_library.pkl"

        strategy_library = None
        try:
            if os.path.exists(lifelong_strategy_library_pkl):
                with open(lifelong_strategy_library_pkl, "rb") as f:
                    strategy_library = pickle.load(f)
                logger.info(
                    f"Loaded lifelong strategy library from {lifelong_strategy_library_pkl}"
                )
            elif os.path.exists(warm_up_strategy_library_pkl):
                with open(warm_up_strategy_library_pkl, "rb") as f:
                    strategy_library = pickle.load(f)
                logger.info(
                    f"Loaded warm-up strategy library from {warm_up_strategy_library_pkl}"
                )
        except Exception as e:
            logger.error(f"Failed to load existing strategy library: {e}")

        if strategy_library is not None and isinstance(strategy_library, dict):
            jailbreak_prompts = []
            for problem in tqdm(
                failure_dataset[prompt_column], total=len(failure_dataset)
            ):
                jailbreak_prompts.append(
                    autodan_turbo_pipeline.test(problem, strategy_library)
                )
            return jailbreak_prompts
        else:
            logger.warning(
                "Retrieve-only requested but no existing strategy library was found. Falling back to normal run."
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
    for problem in tqdm(failure_dataset[prompt_column], total=len(failure_dataset)):
        jailbreak_prompts.append(
            autodan_turbo_pipeline.test(problem, lifelong_strategy_library)
        )
    return jailbreak_prompts


def attack_autodan_turbo(
    model,
    tokenizer,
    failure_dataset: DataFrame,
    task_name: str,
    feature: Dataset = Dataset.MATH,
    retrieve_only: bool = False,
    target_model_name: str = None,
):
    prompt_column = None
    match feature:
        case Dataset.MATH:
            prompt_column = "problem"
        case Dataset.SQL:
            prompt_column = "question"
        case Dataset.SAMSUM:
            prompt_column = "dialogue"
        case Dataset.MMLU:
            prompt_column = "question"
        case _:
            raise ValueError(f"Invalid feature: {feature}")

    # Append string to the end of each problem in the failure dataset dataframe
    failure_dataset[prompt_column] = [
        format_feature_prompt(row, feature) for _, row in failure_dataset.iterrows()
    ]

    args = Namespace(
        epochs=150,
        # lifelong_iterations=5,
        lifelong_iterations=1,
        warm_up_iterations=1,
        hot=False,
        hot_lifelong=False,
        # hot_lifelong=True if feature == Dataset.MATH else False,
        retrieve_only=retrieve_only,
        # hot_lifelong=True, # skip warm-up phase
        failure_dataset=failure_dataset,
        target_model=model,
        target_tokenizer=tokenizer,
        warm_up_size=min(50, len(failure_dataset[prompt_column])),
        run_name=task_name,
        autodan_dir=f"{PROJECT_DIR}/locket/robustness/AutoDAN_Turbo",
        prompt_column=prompt_column,
        target_model_name=target_model_name,
    )

    jailbreak_prompts = _attack_autodan_turbo(args)
    generations = model_inference(
        model,
        tokenizer,
        prompt_list=jailbreak_prompts,
    )

    return generations, jailbreak_prompts
