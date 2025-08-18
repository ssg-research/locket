from utils.prompt import extract_math_answer


def evaluate_math_correctness(model_output, ground_truth_output):
    model_extracted_answer = extract_math_answer(model_output)
    ground_truth_extracted_answer = extract_math_answer(ground_truth_output)

    return model_extracted_answer == ground_truth_extracted_answer
