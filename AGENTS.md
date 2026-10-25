# AGENTS.md — LLM-Addition

A research repo studying how LLMs (DeepSeek-R1 distill) perform multi-number addition, using Gemini as a critic to generate correct chain-of-thought for fine-tuning.

## Structure

- `scripts/` — runnable Python scripts (the pipeline)
- `json/` — all data artifacts at every pipeline stage
- `*.ipynb` — original notebooks (kept for reference, prefer scripts)

## Pipeline

```
Step 1:  python scripts/generate_dataset.py                          → json/additions_r1.json
Step 2:  python scripts/extract_answers.py json/additions_r1.json    → json/additions_error.json
Step 3:  python scripts/run_gemini_critic.py json/additions_error.json  → json/gemini_responses_complete.json
Step 3b: python scripts/prepare_dataset.py json/gemini_responses_complete.json  → json/alpaca_dataset.json
Step 4:  python scripts/train_qwen.py --dataset json/alpaca_dataset.json   (Kaggle / CUDA)
         python scripts/train_cot.py --dataset json/alpaca_dataset.json     (Kaggle / CUDA)
```

Or via `make`:
```
make dataset    # Step 1
make extract    # Step 2
make critique   # Step 3 (requires GOOGLE_API_KEY)
make prepare    # Step 3b
make all        # Steps 1-3b
```

## Key facts

- **Not runnable locally for all steps** — Step 1 runs locally via any OpenAI-compatible endpoint (LM Studio default: `http://localhost:1234/v1`). Step 3 needs `GOOGLE_API_KEY`. Steps 4 requires CUDA + unsloth (Kaggle T4).
- **No package manager, no tests, no CI** — deps installed via `uv pip install -r requirements.txt` or ad-hoc
- **Answer extraction** regex: `\\boxed{...}` then 6+ digit fallback
- **Loss masking** — `CustomDataCollator` masks all tokens before `" Response"` to train only on response tokens
- **LoRA config** — r=16, alpha=16, dropout=0, target all linear projections
- **Dataset** — 1465 generated → 1176 errors → 1051 corrected by Gemini
- **Secrets** — `gemini.ipynb` / `run_gemini_critic.py` reads `GOOGLE_API_KEY` env var; this repo was sanitized in `d4f1aa3`

## Data flow

```
generate_dataset.py            additions_r1.json  (raw: numbers + model response)
        ↓
extract_answers.py             additions_error.json  (+ thought, actual_answer, generated_answer, error flag)
        ↓
run_gemini_critic.py           gemini_responses_complete.json  (+ new_thought from Gemini)
        ↓
prepare_dataset.py             output.json  (corrected entries only)
                                alpaca_dataset.json  (Alpaca prompt format, one text blob per row)
```
