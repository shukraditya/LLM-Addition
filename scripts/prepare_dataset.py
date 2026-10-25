"""
Extract answers from Gemini responses, filter correct ones, and format for Alpaca fine-tuning.

Usage:
  python scripts/prepare_dataset.py json/gemini_responses_complete.json -o json/output.json --alpaca-out json/alpaca_dataset.json
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


def filter_matching_entries(data):
    matching = []
    for entry in data:
        if "actual_answer" in entry and "new_generated_answer" in entry:
            if entry["actual_answer"] == entry["new_generated_answer"]:
                matching.append(entry)
    return matching


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}
### Response:
{}"""


def format_alpaca(entry):
    textin = f"Add the numbers given: {entry['numbers']} <think>{entry['thought']}"
    textout = f"{entry['new_thought']}</think><answer>{entry['actual_answer']}</answer>"
    return ALPACA_PROMPT.format("", textin, textout)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from Gemini responses")
    parser.add_argument("input", help="Input JSON file (Gemini responses with new_thought)")
    parser.add_argument("-o", "--output", default="json/output.json",
                        help="Output path for filtered matching entries")
    parser.add_argument("--alpaca-out", default=None,
                        help="Output path for Alpaca-formatted dataset (optional)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Step 1: Extract answers from Gemini's new_thought
    for entry in data:
        if "new_thought" in entry and "new_generated_answer" not in entry:
            answer = extract_last_number(entry["new_thought"])
            if answer is not None:
                entry["new_generated_answer"] = answer

    # Step 2: Write back with extracted answers (if output differs from input)
    if args.output != args.input:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Updated entries with extracted answers -> {args.output}")

    # Step 3: Filter to entries where corrected answer matches ground truth
    matching = filter_matching_entries(data)
    print(f"Matching entries (corrected by Gemini): {len(matching)} / {len(data)}")

    # Step 4: Format for Alpaca fine-tuning
    if args.alpaca_out:
        alpaca_data = []
        for entry in matching:
            alpaca_data.append(format_alpaca(entry))

        dataset = {"text": alpaca_data}
        with open(args.alpaca_out, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Alpaca-formatted dataset ({len(alpaca_data)} examples) -> {args.alpaca_out}")

    # Also save just the matching entries
    match_out = args.output.replace(".json", "_matched.json") if args.output else "json/matched.json"
    if args.output:
        with open(match_out, "w") as f:
            json.dump(matching, f, indent=4)
        print(f"Matching entries saved to {match_out}")


if __name__ == "__main__":
    main()
