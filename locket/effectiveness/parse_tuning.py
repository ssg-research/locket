import ast
import json
import re

from locket.config import PROJECT_DIR
from locket.utils.logger import logger

LOG_FILE_PATH = f"{PROJECT_DIR}/get_scale.log"
OUTPUT_FILE_PATH = f"{PROJECT_DIR}/logs/effect_tuning_results.json"


def main():
    # Read from get_scale.log
    with open(LOG_FILE_PATH, "r") as f:
        lines = f.readlines()

    # Extract the lines which contains the phrase "Results for"
    # and extract the results
    results = [
        # {
        #     "model": "deepseek-ai/deepseek-math-7b-rl",
        #     "merging_tau": None,
        #     "single_scale": None,
        #     "mmlu_accuracy": 0.4412,
        #     "mmlu_refusal_rate": 0.00,
        #     "sql_accuracy": 0.93,
        #     "sql_refusal_rate": 0.00,
        #     "samsum_accuracy": 0.27,
        #     "samsum_refusal_rate": 0.00,
        #     "math_accuracy": 0.4402,
        #     "math_refusal_rate": 0.00,
        # },
        # {
        #     "model": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        #     "merging_tau": None,
        #     "single_scale": None,
        #     "mmlu_accuracy": 0.3036,
        #     "mmlu_refusal_rate": 0.00,
        #     "sql_accuracy": 0.94,
        #     "sql_refusal_rate": 0.00,
        #     "samsum_accuracy": 0.28,
        #     "samsum_refusal_rate": 0.00,
        #     "math_accuracy": 0.1386,
        #     "math_refusal_rate": 0.00,
        # },
        # {
        #     "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        #     "merging_tau": None,
        #     "single_scale": None,
        #     "mmlu_accuracy": 0.6261,
        #     "mmlu_refusal_rate": 0.00,
        #     "sql_accuracy": 0.88,
        #     "sql_refusal_rate": 0.00,
        #     "samsum_accuracy": 0.32,
        #     "samsum_refusal_rate": 0.00,
        #     "math_accuracy": 0.21,
        #     "math_refusal_rate": 0.00,
        # },
    ]
    for line in lines:
        if "Results for" not in line:
            continue

        # Extract the brace-delimited Python dict payload
        start_brace = line.find("{")
        end_brace = line.rfind("}")
        if start_brace == -1 or end_brace == -1 or end_brace < start_brace:
            logger.error(f"Error extracting payload from line: {line}")
            continue

        payload = line[start_brace : end_brace + 1]

        # Strip numpy float wrappers like np.float64(0.123) or np.float_(0.123)
        payload = re.sub(r"np\.float(?:\d+)?\(([^)]+)\)", r"\1", payload)
        payload = re.sub(r"np\.float_\(([^)]+)\)", r"\1", payload)

        # Parse Python-literal style dict safely
        try:
            record = ast.literal_eval(payload)
        except Exception:
            # Fallback: coerce to JSON-ish string and try json.loads
            tmp = (
                payload.replace("None", "null")
                .replace("True", "true")
                .replace("False", "false")
            )
            tmp = tmp.replace("'", '"')
            try:
                record = json.loads(tmp)
            except Exception:
                continue

        results.append(record)

    # Save the results to a json file
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
