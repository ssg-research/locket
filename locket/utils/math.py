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

import multiprocessing
import re
from math import isclose
from typing import Any, Dict, Optional, Union

import numpy as np
import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")

    # replace \\ with \
    # string = string.replace("\\\\", "\\")
    # string = string.replace("\\\\", "\\")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace(r"\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string


def extract_boxed_answers(text):
    answers = []
    for piece in text.split("boxed{")[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == "{":
                n += 1
            elif piece[i] == "}":
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_program_output(pred_str):
    """
    extract output between the last ```output\n...\n```
    """
    if "```output" not in pred_str:
        return ""
    if "```output" in pred_str:
        pred_str = pred_str.split("```output")[-1]
    if "```" in pred_str:
        pred_str = pred_str.split("```")[0]
    output = pred_str.strip()
    return output


def extract_answer(pred_str, exhaust=False):
    pred = []
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = [tmp.split("$. I hope", 1)[0].strip()]
    elif "boxed" in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif "he answer is" in pred_str:
        pred = [pred_str.split("he answer is")[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            # fall back to program
            pred.append(program_output)
        else:  # use the last number
            pattern = r"-?\d*\.?\d+"
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if len(ans) >= 1:
                ans = ans[-1]
            else:
                ans = ""
            if ans:
                pred.append(ans)

    # multiple line
    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = ans.rstrip("°")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""


def replace_frac_with_parentheses(expr):
    r"""
    Replace all instances of \frac{num}{den} in a given string expr with (num) / (den).

    Parameters:
    - expr: A string containing one or more instances of \frac{num}{den}

    Returns:
    - A string with all instances of \frac{num}{den} replaced by (num) / (den)
    """
    import re

    # Define the pattern to match \frac{num}{den}
    pattern = r"\\frac\{(.*?)\}\{(.*?)\}"

    # Replacement function that formats the match as (num) / (den)
    def replacer(match):
        num, den = match.groups()
        return f"({num}) / ({den})"

    # Perform the replacement
    result = re.sub(pattern, replacer, expr)

    return result


# test_cases = [
#     r"\frac{xyz}{abc}",  # Basic case
#     r"\frac{n}{d} + \frac{x}{y}",  # Multiple fractions
#     r"a + \frac{b}{c} - d",  # Mixed expression
#     r"\frac{x+y}{z}",  # Numerator with operation
#     r"\frac{a}{b+c}",  # Denominator with operation
#     r"\frac{1}{2}*\frac{3}{4}",  # Fractions with operation between them
#     r"5 - \frac{6}{7}*\frac{8}{9}",  # Complex expression with operations
#     r"\frac{\frac{a}{b}}{\frac{c}{d}}",  # Nested fractions # nested doesn't work
#     r"",  # Empty string
#     r"no fractions here"  # No fractions
# ]


def quick_convert_to_ascii_math_helper(string: str) -> str:
    string = string.replace(r"\sqrt", "sqrt")
    string = string.replace(r"\infty", "infty")
    string = string.replace(r"\implies", "=>")
    string = string.replace(r"\overline", "bar")
    string = string.replace(r"\pm", "+-")
    string = string.replace(r"\in", "in")
    string = string.replace(r"\cup", "uu")
    string = string.replace(r"\cap", "nn")
    string = string.replace(r"\bigcup", "uuu")
    string = string.replace(r"\bigcap", "nnn")
    string = string.replace(r"\neq", "!=")
    string = string.replace(r"\ne", "!=")
    string = string.replace(r"```", "`")

    string = replace_frac_with_parentheses(string)

    return string


def extract_math_answer(question, reasoning):
    answer = []
    items = extract_answer(reasoning, exhaust=True)
    for ans in items:
        if "separated by commas" in question and all(ch not in ans for ch in "()[]"):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend(
                [
                    a.strip()
                    for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split(
                        "[SEP]"
                    )
                ]
            )
        else:
            answer.append(ans.strip())
    return answer


def extract_program(result: str, last_only=True):
    """
    extract the program after "```python", and before "```"
    """
    program = ""
    start = False
    for line in result.split("\n"):
        if line.startswith("```python"):
            if last_only:
                program = ""  # only extract the last program
            else:
                program += "\n# ========\n"
            start = True
        elif line.startswith("```"):
            start = False
        elif start:
            program += line + "\n"
    return program


def parse_ground_truth(example: Dict[str, Any], data_name):
    if "gt_cot" in example:
        return example["gt_cot"], strip_string(example["gt"])

    # parse ground truth
    if data_name in ["math", "ocw"]:
        gt_cot = example["solution"]
        gt_ans = extract_answer(gt_cot)
    elif data_name == "gsm8k":
        gt_cot, gt_ans = example["answer"].split("####")
    elif data_name == "gsm-hard":
        gt_cot, gt_ans = example["code"], example["target"]
    elif data_name == "svamp":
        gt_cot, gt_ans = example["Equation"], example["Answer"]
    elif data_name == "asdiv":
        gt_cot = example["formula"]
        gt_ans = re.sub(r"\(.*?\)", "", example["answer"])
    elif data_name == "mawps":
        gt_cot, gt_ans = None, example["target"]
    elif data_name == "tabmwp":
        gt_cot = example["solution"]
        gt_ans = example["answer"]
        if example["ans_type"] in ["integer_number", "decimal_number"]:
            if "/" in gt_ans:
                gt_ans = int(gt_ans.split("/")[0]) / int(gt_ans.split("/")[1])
            elif "," in gt_ans:
                gt_ans = float(gt_ans.replace(",", ""))
            elif "%" in gt_ans:
                gt_ans = float(gt_ans.split("%")[0]) / 100
            else:
                gt_ans = float(gt_ans)
    elif data_name == "bbh":
        gt_cot, gt_ans = None, example["target"]
    else:
        raise NotImplementedError(data_name)
    # post process
    gt_cot = str(gt_cot).strip()
    gt_ans = strip_string(gt_ans)
    return gt_cot, gt_ans


def parse_question(example, data_name):
    question = ""
    if data_name == "asdiv":
        question = f"{example['body'].strip()} {example['question'].strip()}"
    elif data_name == "svamp":
        body = example["Body"].strip()
        if not body.endswith("."):
            body = body + "."
        question = f"{body} {example['Question'].strip()}"
    elif data_name == "tabmwp":
        title_str = (
            f'regarding "{example["table_title"]}" ' if example["table_title"] else ""
        )
        question = f"Read the following table {title_str}and answer a question:\n"
        question += f"{example['table']}\n{example['question']}"
        if example["choices"]:
            question += (
                f" Please select from the following options: {example['choices']}"
            )
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    assert question != ""
    return question.strip()


def run_execute(executor, result, prompt_type, execute=False):
    if not result or result == "error":
        return None, None
    report = None

    if "program_only" in prompt_type:
        prediction = extract_program_output(result)
    elif prompt_type in ["pot", "pal"] and execute:
        code = extract_program(result)
        prediction, report = executor.apply(code)
    else:
        prediction = extract_answer(result)

    prediction = strip_string(prediction)
    return prediction, report


def parse_digits(num):
    # format: 234.23 || 23%
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def normalize_prediction(prediction):
    try:  # 1. numerical equal
        if is_digit(prediction):
            prediction = np.round(float(str(prediction).replace(",", "")), 6)
        return str(prediction)
    except Exception:
        pass

    # 2. symbolic equal
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    brackets = []
    while (
        prediction.startswith("[")
        and prediction.endswith("]")
        or (prediction.startswith("(") and prediction.endswith(")"))
    ):
        brackets = prediction[0]
        prediction = prediction[1:-1]
    if brackets and "," in prediction:
        pred_parts = [normalize_prediction(part) for part in prediction.split(",")]
        prediction = ",".join(pred_parts)

    if brackets:
        for b in reversed(brackets):
            if b == "[":
                prediction = "[" + prediction + "]"
            else:
                assert b == "("
                prediction = "(" + prediction + ")"

    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except Exception:
                pass
        return s

    prediction = _parse(prediction)

    for s in ["{", "}", "(", ")"]:
        prediction = prediction.replace(s, "")

    return prediction


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: Optional[float] = None,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if str(prediction) == str(reference):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, abs_tol=1e-3):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i],
                        ref_parts[i],
                        include_percentage,
                        is_close,
                        timeout=timeout,
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                                timeout=timeout,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1],
            reference,
            include_percentage,
            is_close,
            timeout=timeout,
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction,
            reference.split("=")[1],
            include_percentage,
            is_close,
            timeout=timeout,
        ):
            return True

    # symbolic equal with sympy
    if timeout is not None:
        if call_with_timeout(
            symbolic_equal_process_queue, prediction, reference, timeout=timeout
        ):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except Exception:
                pass
        return s

    # slightly cursed that these could still be strings...
    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=1e-3):
            return True
    except Exception:
        pass
    return False


def symbolic_equal_process_queue(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(
    func,
    *args,
    timeout=1.0,
    default_on_timeout: Any = False,
    print_on_timeout=True,
    **kwargs,
):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        if print_on_timeout:
            print(f"Function took longer than {timeout} seconds")
        return default_on_timeout

    return output_queue.get()


def is_correct(pred, ans, prec=1e-3, timeout: Optional[float] = 2.0):
    if isinstance(pred, list) and isinstance(ans, list):
        # is this even the right logic here???
        pred_matched = set()
        ans_matched = set()
        for i in range(len(pred)):
            for j in range(len(ans)):
                if is_correct(pred=pred[i], ans=ans[j], prec=prec, timeout=timeout):
                    pred_matched.add(i)
                    ans_matched.add(j)
        return len(pred_matched) == len(pred) and len(ans_matched) == len(ans)
    elif isinstance(pred, str) and isinstance(ans, str):
        if "\\cup" in pred and "\\cup" in ans:
            return is_correct(
                pred=pred.split("\\cup"),
                ans=ans.split("\\cup"),
                prec=prec,
                timeout=timeout,
            )
        else:
            label = False
            try:
                label = (
                    abs(
                        float(regex.sub(r",", "", str(pred)))
                        - float(regex.sub(r",", "", str(ans)))
                    )
                    < prec
                )
            except Exception:
                pass
            label = (
                label or (ans and pred == ans) or math_equal(pred, ans, timeout=timeout)
            )
            return label
    else:
        print(f"{pred=} {ans=}", flush=True)
        raise NotImplementedError()


def eval_math(pred, ans, prec=1e-3, timeout: Optional[float] = 2.0):
    if isinstance(pred, list) and isinstance(ans, list):
        # for some questions in MATH, `reference` repeats answers
        _ans = []
        for a in ans:
            if a not in _ans:
                _ans.append(a)
        ans = _ans
        # some predictions for MATH questions also repeats answers
        _pred = []
        for a in pred:
            if a not in _pred:
                _pred.append(a)
        # some predictions mistakenly box non-answer strings
        pred = _pred[-len(ans) :]

    return is_correct(pred, ans, prec=prec, timeout=timeout)
