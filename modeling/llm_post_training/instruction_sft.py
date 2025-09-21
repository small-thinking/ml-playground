"""
SFT (Supervised Fine-Tuning) Training Script for Instruction Following

This script implements SFT training for improving instruction-following
capabilities. Supports both Alpaca dataset format and custom messages format.
Supports both initial training and continuing from a previous SFT model.

Usage:
    # Initial SFT training from base model with default Alpaca dataset
    python instruction_sft.py --model-size 3B --use-lora

    # SFT training with custom Alpaca format dataset from HF
    python instruction_sft.py --model-size 3B --dataset-format alpaca \
        --dataset-name microsoft/orca-math-word-problems-200k

    # SFT training with custom messages format dataset from HF
    python instruction_sft.py --model-size 3B --dataset-format messages \
        --dataset-name HuggingFaceH4/ultrachat_200k

    # Continue SFT from local checkpoint
    python instruction_sft.py --checkpoint-path ./models/Llama-3.2-3B-Full-SFT \
        --dataset-name tech-tao/gang-jing_contrarian_sft_data \
        --dataset-format messages

    # Continue SFT from Hugging Face model
    python instruction_sft.py --checkpoint-path microsoft/DialoGPT-medium
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
    """Format instruction following examples from Alpaca format."""
    # Create the instruction-following format, which is the
    # foundation of SFT. The model is trained on this structured format to
    # learn how to follow instructions.
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


def format_messages(example: Dict[str, Any]) -> Dict[str, str]:
    """Format conversation examples from messages format."""
    messages = example["messages"]

    # Build the conversation text
    conversation_parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            conversation_parts.append(f"### Instruction:\n{content}")
        elif role == "assistant":
            conversation_parts.append(f"### Response:\n{content}")
        # Skip other roles (system, etc.) for now

    return {"text": "\n\n".join(conversation_parts)}


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

    # Determine model path for loading
    if args.checkpoint_path:
        model_path = args.checkpoint_path
        print(f"🔄 Continuing SFT training from: {model_path}")
    else:
        if not args.model_size:
            raise ValueError(
                "Either --model-size or --checkpoint-path must be " "specified"
            )
        model_path = get_model_name(args.model_size)
        print(f"🚀 Starting SFT training for {model_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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
    if args.dataset_format == "alpaca":
        dataset_name = args.dataset_name or "tatsu-lab/alpaca"
        print(f"📊 Loading Alpaca format dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        format_function = format_instruction
    elif args.dataset_format == "messages":
        if not args.dataset_name:
            raise ValueError(
                "--dataset-name must be specified when using messages format"
            )
        print(f"📊 Loading messages format dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split="train")
        format_function = format_messages
    else:
        raise ValueError(f"Unsupported dataset format: {args.dataset_format}")
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"📊 Dataset columns: {dataset.column_names}")

    dataset = dataset.map(format_function, remove_columns=dataset.column_names)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Calculate max_steps based on dataset size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    max_possible_steps = len(dataset) // effective_batch_size
    max_steps = min(args.max_steps, max_possible_steps)

    print("📊 Training configuration:")
    print(f"   - Dataset size: {len(dataset)} samples")
    print(f"   - Effective batch size: {effective_batch_size}")
    print(f"   - Max possible steps: {max_possible_steps}")
    print(f"   - Using max_steps: {max_steps}")

    # Training arguments
    if args.checkpoint_path:
        # For continue training, use checkpoint name in output directory
        checkpoint_name = args.checkpoint_path.split("/")[-1]
        output_dir = (
            f"./models/{checkpoint_name}-"
            f"{'LoRA' if args.use_lora else 'Full'}-Continue-SFT"
        )
    else:
        # For initial training, use base model name
        output_dir = (
            f"./models/{model_path.split('/')[-1]}-"
            f"{'LoRA' if args.use_lora else 'Full'}-SFT"
        )
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=max(100, int(0.05 * max_steps)),
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.005,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=False,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=min(10000, max_steps),
        save_total_limit=1,
        report_to="wandb" if not args.disable_wandb else "none",
        run_name=(
            f"{model_path.split('/')[-1]}-SFT" if not args.disable_wandb else None
        ),
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
        "--model-size",
        type=str,
        default=None,
        choices=["0.5B", "1.5B", "3B", "8B"],
        help=(
            "Model size for initial training "
            "(not needed when using --checkpoint-path)"
        ),
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to previous SFT model to continue training from "
            "(HF model name or local folder)"
        ),
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="alpaca",
        choices=["alpaca", "messages"],
        help=(
            "Dataset format: 'alpaca' for Alpaca format or 'messages' "
            "for messages format"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset name (optional for alpaca format, "
            "required for messages format)"
        ),
    )
    args = parser.parse_args()

    # Validation
    if args.checkpoint_path and args.use_lora:
        print("⚠️  Warning: Continue training with LoRA is not fully " "supported yet.")
        print("   Consider using full fine-tuning for continue training.")
    if args.dataset_format == "messages" and not args.dataset_name:
        raise ValueError("--dataset-name is required when using messages format")

    main(args)
