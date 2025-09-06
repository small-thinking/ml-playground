"""
SFT (Supervised Fine-Tuning) Training Script for Instruction Following

This script implements SFT training for improving instruction-following
capabilities using the Alpaca dataset.

Usage:
    python instruction_sft.py --model-size 3B --use-lora
"""

import os
import argparse
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def get_model_name(model_size: str) -> str:
    """Get the model name based on size."""
    model_mapping = {
        "8B": "meta-llama/Llama-3.1-8B",
        "3B": "meta-llama/Llama-3.2-3B",
        "1.5B": "Qwen/Qwen2-1.5B",
        "0.5B": "Qwen/Qwen2-0.5B",
    }
    if model_size not in model_mapping:
        raise ValueError(f"Invalid model size: {model_size}")
    return model_mapping[model_size]


def format_instruction(example: Dict[str, Any]) -> Dict[str, str]:
    """Format instruction following examples."""
    # Create the instruction-following format, which is the
    # foundation of SFT. The model is trained on this structured format to learn
    # how to follow instructions.
    if example.get("input", "").strip():
        return {
            "text": (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        }
    return {
        "text": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    }


def tokenize_function(
    examples: Dict[str, Any], tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """Tokenize the formatted examples and apply loss masking."""
    # Tokenize the formatted text.
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        add_special_tokens=True,
    )

    labels = torch.tensor(tokenized["input_ids"])
    labels[labels == tokenizer.pad_token_id] = -100  # Mask padding tokens

    # Mask the instruction part of the input so that the model only
    # learns to predict the response. This is done by setting the labels of the
    # instruction tokens to -100, which is ignored by the loss function.
    for i, text in enumerate(examples["text"]):
        response_start_marker = "### Response:"
        marker_pos = text.find(response_start_marker)
        if marker_pos != -1:
            prompt_part = text[:marker_pos]
            prompt_tokens = tokenizer(prompt_part, add_special_tokens=True).input_ids
            response_start_token_idx = len(prompt_tokens)
            labels[i, :response_start_token_idx] = -100

    tokenized["labels"] = labels.tolist()
    return tokenized


def main(args):
    """Main training function."""
    # Setup
    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    model_name = get_model_name(args.model_size)
    print(f"🚀 Starting SFT training for {model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.use_lora:
        print("🔧 Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load and prepare dataset
    print("📊 Loading and preparing Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    if len(dataset) > 10000:
        dataset = dataset.select(range(10000))

    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Training arguments
    output_dir = f"./models/{model_name.split('/')[-1]}-{'LoRA' if args.use_lora else 'Full'}-SFT"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=max(100, int(0.05 * args.max_steps)),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.005,
        max_grad_norm=1.0,
        gradient_accumulation_steps=8,
        fp16=False,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=10000,
        save_total_limit=1,
        report_to="wandb" if not args.disable_wandb else "none",
        run_name=f"{model_name.split('/')[-1]}-SFT" if not args.disable_wandb else None,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Train
    print("🚀 Starting training...")
    trainer.train()

    # Save final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT Training Script for Instruction Following"
    )
    parser.add_argument(
        "--model-size", type=str, default="3B", choices=["0.5B", "1.5B", "3B", "8B"]
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()
    main(args)
