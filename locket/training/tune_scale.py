# Authors: Tony He, Vasisht Duddu, N Asokan
# Copyright 2026 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sweep single_scale for one locking adapter to meet the locking target:
locked feature utility -> 0 while the other features stay within TOL of baseline.
The base+adapter model is loaded once and rescaled in place per candidate scale."""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"

from locket.effectiveness.eval_math import eval_math
from locket.effectiveness.eval_mmlu import eval_mmlu
from locket.effectiveness.eval_samsum import eval_samsum
from locket.effectiveness.eval_sql import eval_sql
from locket.typings import Adapter, MMLUDomain, Models
from locket.utils.dataset import (
    load_math_dataset,
    load_mmlu_dataset,
    load_samsum_dataset,
    load_sql_dataset,
    process_dataset,
)
from locket.utils.logger import logger
from locket.utils.model import load_model_with_adapters, rescale_adapter_scale
from locket.utils.tokenizer import get_tokenizer

# ============================== CONFIG ==============================
ADAPTER = Adapter.SAMSUM
LOCKED_FEATURE = "samsum"
SCALES = [0.7, 0.85, 1.0, 1.2]
SAMPLE = 250
MATH_SAMPLE = 150  # math generations are long; keep its sweep sample smaller
BASELINE = {"math": 0.415, "mmlu": 0.493, "sql": 0.935, "samsum": 0.280}
TOL = 0.05
LOCKED_TARGET = 0.02  # locked feature considered locked at/below this
# ===================================================================

EVAL_FNS = {"math": eval_math, "mmlu": eval_mmlu, "sql": eval_sql, "samsum": eval_samsum}


def main():
    model = load_model_with_adapters(
        Models.DEEPSEEK_7B_MATH, [ADAPTER], single_scale=1.0
    )
    cur_scale = 1.0

    datasets = {
        "math": process_dataset(load_math_dataset("test"), shuffle=True, sample_size=MATH_SAMPLE),
        "mmlu": process_dataset(load_mmlu_dataset("test", excluded_domains=[MMLUDomain.MATH]), shuffle=True, sample_size=SAMPLE),
        "sql": process_dataset(load_sql_dataset("test"), shuffle=True, sample_size=SAMPLE),
        "samsum": process_dataset(load_samsum_dataset("test"), shuffle=True, sample_size=SAMPLE),
    }
    eval_order = [LOCKED_FEATURE] + [f for f in ["math", "mmlu", "sql", "samsum"] if f != LOCKED_FEATURE]

    summary = []
    for scale in SCALES:
        rescale_adapter_scale(model, scale / cur_scale)
        cur_scale = scale

        row = {}
        locked_refusal = None
        for feat in eval_order:
            tokenizer = get_tokenizer(Models.DEEPSEEK_7B_MATH, add_system=feat)
            metric, refusal = EVAL_FNS[feat](
                datasets[feat], tokenizer, model,
                model_name=f"sweep_{LOCKED_FEATURE}_s{scale}_{feat}",
                is_refusal_model=(feat == LOCKED_FEATURE),
            )
            row[feat] = metric
            if feat == LOCKED_FEATURE:
                locked_refusal = refusal

        locked = row[LOCKED_FEATURE]
        # ROUGE-based metrics (SAMSUM) floor above 0 from refusal-text overlap, so a
        # feature counts as locked at near-total refusal OR a near-zero exact-match metric.
        locked_ok = (locked_refusal is not None and locked_refusal >= 0.98) or locked <= LOCKED_TARGET
        others_ok = all(
            abs(row[f] - BASELINE[f]) <= TOL for f in row if f != LOCKED_FEATURE
        )
        passes = locked_ok and others_ok
        summary.append((scale, locked, others_ok, passes, row))
        logger.info(
            f"SWEEP scale={scale}: {LOCKED_FEATURE}(locked)={locked:.4f} "
            f"| math={row.get('math'):.4f} mmlu={row.get('mmlu'):.4f} "
            f"sql={row.get('sql'):.4f} samsum={row.get('samsum'):.4f} "
            f"| others_within_5%={others_ok} | PASS={passes}"
        )

    logger.info("===== SWEEP SUMMARY =====")
    passing = [s for s, _, _, p, _ in summary if p]
    for scale, locked, others_ok, passes, _ in summary:
        logger.info(f"  scale={scale}: locked={locked:.4f} others_ok={others_ok} PASS={passes}")
    if passing:
        logger.info(f"RECOMMENDED single_scale for {LOCKED_FEATURE}: {min(passing)} (smallest passing)")
    else:
        logger.info(f"NO PASSING SCALE for {LOCKED_FEATURE} in {SCALES}; consider extending range or retraining.")


if __name__ == "__main__":
    main()
