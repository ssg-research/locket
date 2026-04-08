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

from locket.utils.prompt import extract_math_answer


def evaluate_math_correctness(model_output, ground_truth_output):
    model_extracted_answer = extract_math_answer(model_output)
    ground_truth_extracted_answer = extract_math_answer(ground_truth_output)

    return model_extracted_answer == ground_truth_extracted_answer
