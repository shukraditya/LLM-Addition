"""
Extract answers from raw model responses and compare against ground truth.

Usage:
  python scripts/extract_answers.py json/additions_r1.json -o json/additions_error.json
"""
import argparse
import json
import re
import sys


def extract_last_number(text):
    if not isinstance(text, str):
        return None
    text = text.replace(",", "")
    boxed_match = re.search(r"\\boxed{(\d+)}", text)
    if boxed_match:
        number = boxed_match.group(1)
        if len(number) >= 6:
            return int(number)
    numbers = re.findall(r"\b\d{6,}\b", text)
    if numbers:
        return int(numbers[-1])
    return None


def extract_thought(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no <think> tags, treat the whole response as thought
    # (the response template wraps thinking in <think> tags)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract answers from model responses")
    parser.add_argument("input", help="Input JSON file (raw responses)")
    parser.add_argument("-o", "--output", default="json/additions_error.json", help="Output JSON path")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results = []
    errors = 0
    total = 0

    for entry in data:
        numbers = entry["numbers"]
        response = entry["response"]
        thought = extract_thought(response)
        generated_answer = extract_last_number(response)
        actual_answer = sum(numbers)

        is_error = generated_answer is None or generated_answer != actual_answer
        if is_error:
            errors += 1
        total += 1

        results.append({
            "numbers": numbers,
            "response": response,
            "thought": thought,
            "actual_answer": actual_answer,
            "generated_answer": generated_answer,
            "error": is_error,
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Total: {total}, Errors: {errors} ({100 * errors / total:.1f}%)")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
