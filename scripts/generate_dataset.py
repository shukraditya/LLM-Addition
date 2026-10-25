import argparse
import json
import random
import sys
import time

from openai import OpenAI


def generate_problem(num_count=6, min_val=10**4, max_val=10**7 - 1):
    return [random.randint(min_val, max_val) for _ in range(num_count)]


def get_model_response(client, problem, model):
    numbers_str = "+".join(map(str, problem))
    prompt = f"{numbers_str}\nFormat your final answer as <answer>1234567</answer>"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Generate addition dataset via an OpenAI-compatible local endpoint")
    parser.add_argument("--model", default="deepseek-r1:1.5b", help="Model name on the server")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="API base URL (LM Studio default)")
    parser.add_argument("-n", "--num-problems", type=int, default=500, help="Number of problems to generate")
    parser.add_argument("--num-count", type=int, default=6, help="Numbers per problem")
    parser.add_argument("--output", default="json/additions_r1.json", help="Output JSON path")
    parser.add_argument("--delay", type=float, default=0, help="Seconds to wait between API calls")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    dataset = []
    for i in range(args.num_problems):
        numbers = generate_problem(num_count=args.num_count)
        try:
            response = get_model_response(client, numbers, model=args.model)
        except Exception as e:
            print(f"Error at problem {i}: {e}", file=sys.stderr)
            response = f"API ERROR: {e}"

        dataset.append({"numbers": numbers, "response": response})

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_problems}", file=sys.stderr)

        if args.delay:
            time.sleep(args.delay)

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} entries to {args.output}")


if __name__ == "__main__":
    main()
