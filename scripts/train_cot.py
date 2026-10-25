"""
Fine-tune DeepSeek-R1-Distill-Llama-8B on corrected addition data.

Designed for Kaggle (T4 GPU) but runs anywhere with CUDA + unsloth.

Usage:
  python scripts/train_cot.py --dataset /path/to/alpaca_dataset.json --output-dir ./outputs
"""
import argparse
import json

import torch
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
        self.response_template = response_template
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False,
        )

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        for i in range(len(examples)):
            input_ids = examples[i]["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            response_start = None
            for j in range(len(input_ids) - len(self.response_token_ids) + 1):
                if input_ids[j:j + len(self.response_token_ids)] == self.response_token_ids:
                    response_start = j + len(self.response_token_ids)
                    break
            if response_start is not None:
                labels[i, :response_start] = -100
        batch["labels"] = labels
        return batch


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}
### Response:
{}"""


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 8B on addition data")
    parser.add_argument("--dataset", required=True, help="Path to Alpaca-formatted dataset JSON")
    parser.add_argument("--model", default="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
                        help="Base model name")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for checkpoints")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--save-model", default="deepseek_llama8b-ft", help="Directory to save final model")
    args = parser.parse_args()

    with open(args.dataset) as f:
        raw = json.load(f)
    texts = raw["text"] if isinstance(raw, dict) and "text" in raw else raw
    dataset = Dataset.from_dict({"text": texts})
    print(f"Loaded {len(dataset)} training examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    response_template = " Response"
    collator = CustomDataCollator(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_hf",
            weight_decay=0.02,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(args.save_model)
    tokenizer.save_pretrained(args.save_model)
    print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
