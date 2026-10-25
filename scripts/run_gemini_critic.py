"""
Send incorrect model outputs to Gemini 2.0 Flash Thinking for correction.

Requires GOOGLE_API_KEY environment variable.

Usage:
  export GOOGLE_API_KEY='your-key'
  python scripts/run_gemini_critic.py json/additions_error.json -o json/gemini_responses_complete.json
"""
import argparse
import json
import os
import sys
import time

from google import genai
from google.api_core.exceptions import ResourceExhausted
from tqdm import tqdm


def save_progress(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def process_data(input_path, output_path, model_name="gemini-2.0-flash-thinking-exp", delay=10):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("FATAL: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    with open(input_path) as f:
        data = json.load(f)

    error_entries = [entry for entry in data if entry.get("error") is True]
    print(f"Found {len(error_entries)} entries with errors to process")

    for entry in tqdm(error_entries, desc="Processing error entries"):
        if "new_thought" in entry:
            continue

        prompt = f"Add {entry['numbers']}. Append to the thought process to correct the output: {entry['thought']}"

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            entry["new_thought"] = response.text
            time.sleep(delay)

        except ResourceExhausted:
            print("\nRate limit reached! Saving progress and waiting 60s...")
            save_progress(data, f"gemini_responses_partial_{int(time.time())}.json")
            time.sleep(60)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                entry["new_thought"] = response.text
            except Exception as e:
                print(f"Second attempt failed: {e}")
                entry["new_thought"] = f"API ERROR: {e}"

        except Exception as e:
            print(f"\nError processing entry: {e}")
            entry["new_thought"] = f"API ERROR: {e}"

    save_progress(data, output_path)
    print(f"Processing complete! Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Gemini critic on incorrect model outputs")
    parser.add_argument("input", default="json/additions_error.json", nargs="?",
                        help="Input JSON file with error entries")
    parser.add_argument("-o", "--output", default="json/gemini_responses_complete.json",
                        help="Output JSON path")
    parser.add_argument("--model", default="gemini-2.0-flash-thinking-exp",
                        help="Gemini model name")
    parser.add_argument("--delay", type=int, default=10,
                        help="Delay in seconds between API calls")
    args = parser.parse_args()

    process_data(args.input, args.output, args.model, args.delay)


if __name__ == "__main__":
    main()
