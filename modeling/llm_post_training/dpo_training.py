#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training Script

This script implements DPO training for preference learning using configurable
datasets and base models. It supports both full fine-tuning and LoRA training.

Dataset Format:
The script expects datasets with 'chosen' and 'rejected' columns, where each
contains conversation format with 'role' and 'content' fields:
- chosen: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
- rejected: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

The script automatically converts this to DPO format (prompt/chosen/rejected).

Base Models:
- Hugging Face model names (e.g., meta-llama/Llama-3.2-3B)
- Local model directories (e.g., ./models/Llama-3.2-3B-LoRA-SFT)
- SFT-tuned models from previous training runs

Usage Examples:
    # Using Hugging Face model
    python dpo_training.py --dataset tech-tao/yizhipian_yizhipian_dpo_data \
        --base-model meta-llama/Llama-3.2-3B --use-lora
    
    # Using local SFT-tuned model
    python dpo_training.py --dataset tech-tao/gang-jing_contrarian_dpo_data \
        --base-model ./models/Llama-3.2-3B-Full-SFT --use-lora
"""

import os
import argparse
from typing import Optional
import json

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType


def get_model_name(model_size: str) -> str:
    """Get the model name based on size for predefined models."""
    model_mapping = {
        "8B": "meta-llama/Llama-3.1-8B",
        "3B": "meta-llama/Llama-3.2-3B",
        "1.5B": "Qwen/Qwen2-1.5B",
        "0.5B": "Qwen/Qwen2-0.5B",
    }
    if model_size not in model_mapping:
        raise ValueError(f"Invalid model size: {model_size}")
    return model_mapping[model_size]


def load_dpo_dataset(dataset_name: str, max_samples: Optional[int] = None) -> Dataset:
    """
    Load and validate DPO dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset on Hugging Face
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        Dataset object with DPO format
    """
    print(f"📊 Loading DPO dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

    # Validate dataset format - check for chosen/rejected columns
    required_columns = ["chosen", "rejected"]
    missing_columns = [
        col for col in required_columns if col not in dataset.column_names
    ]

    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")

    print("📊 Dataset loaded successfully:")
    print(f"   - Total samples: {len(dataset)}")
    print(f"   - Columns: {dataset.column_names}")

    # Sample a few examples to show format
    if len(dataset) > 0:
        example = dataset[0]
        print("📊 Example data format:")
        
        # Show chosen conversation format
        if isinstance(example['chosen'], list) and len(example['chosen']) > 0:
            chosen_user = next((msg for msg in example['chosen'] if msg.get('role') == 'user'), {})
            chosen_assistant = next((msg for msg in example['chosen'] if msg.get('role') == 'assistant'), {})
            print(f"   - Chosen User: {chosen_user.get('content', '')[:100]}...")
            print(f"   - Chosen Assistant: {chosen_assistant.get('content', '')[:100]}...")
        else:
            print(f"   - Chosen: {str(example['chosen'])[:100]}...")
        
        # Show rejected conversation format
        if isinstance(example['rejected'], list) and len(example['rejected']) > 0:
            rejected_user = next((msg for msg in example['rejected'] if msg.get('role') == 'user'), {})
            rejected_assistant = next((msg for msg in example['rejected'] if msg.get('role') == 'assistant'), {})
            print(f"   - Rejected User: {rejected_user.get('content', '')[:100]}...")
            print(f"   - Rejected Assistant: {rejected_assistant.get('content', '')[:100]}...")
        else:
            print(f"   - Rejected: {str(example['rejected'])[:100]}...")

    if max_samples and len(dataset) > max_samples:
        print(f"📊 Limiting dataset to {max_samples} samples for testing")
        dataset = dataset.select(range(max_samples))

    return dataset


def preprocess_dpo_dataset(dataset: Dataset) -> Dataset:
    """
    Preprocess the dataset to convert conversation format to DPO format.
    
    Args:
        dataset: Dataset with chosen/rejected conversation format
        
    Returns:
        Dataset with prompt/chosen/rejected format for DPO training
    """
    def convert_conversation_to_dpo(example):
        # Extract user message from chosen conversation (should be the same in both)
        chosen_user = next((msg for msg in example['chosen'] if msg.get('role') == 'user'), {})
        chosen_assistant = next((msg for msg in example['chosen'] if msg.get('role') == 'assistant'), {})
        rejected_assistant = next((msg for msg in example['rejected'] if msg.get('role') == 'assistant'), {})
        
        return {
            'prompt': chosen_user.get('content', ''),
            'chosen': chosen_assistant.get('content', ''),
            'rejected': rejected_assistant.get('content', '')
        }
    
    print("🔄 Converting conversation format to DPO format...")
    processed_dataset = dataset.map(convert_conversation_to_dpo, remove_columns=dataset.column_names)
    
    print(f"📊 Preprocessed dataset:")
    print(f"   - Total samples: {len(processed_dataset)}")
    print(f"   - Columns: {processed_dataset.column_names}")
    
    # Show example of processed format
    if len(processed_dataset) > 0:
        example = processed_dataset[0]
        print("📊 Processed example:")
        print(f"   - Prompt: {example['prompt'][:100]}...")
        print(f"   - Chosen: {example['chosen'][:100]}...")
        print(f"   - Rejected: {example['rejected'][:100]}...")
    
    return processed_dataset


def prepare_model_and_tokenizer(
    model_path: str, use_lora: bool = False, lora_config: Optional[LoraConfig] = None
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and prepare model and tokenizer for DPO training.

    Args:
        model_path: Path to the base model (local or Hugging Face)
        use_lora: Whether to apply LoRA configuration
        lora_config: LoRA configuration (if None, uses default)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"🤖 Loading model and tokenizer from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("🔧 Set pad_token to eos_token")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Apply LoRA if requested
    if use_lora:
        print("🔧 Applying LoRA configuration...")
        if lora_config is None:
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

    return model, tokenizer


def create_dpo_config(args: argparse.Namespace, dataset_size: int) -> DPOConfig:
    """
    Create DPO training configuration.

    Args:
        args: Command line arguments
        dataset_size: Size of the training dataset

    Returns:
        DPOConfig object
    """
    # Calculate training steps
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size
    max_steps = min(args.max_steps, steps_per_epoch * args.num_epochs)

    print("📊 Training configuration:")
    print(f"   - Dataset size: {dataset_size} samples")
    print(f"   - Effective batch size: {effective_batch_size}")
    print(f"   - Steps per epoch: {steps_per_epoch}")
    print(f"   - Max steps: {max_steps}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Beta (DPO): {args.beta}")

    return DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(100, int(0.05 * max_steps)),
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=min(1000, max_steps // 5),
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="wandb" if not args.disable_wandb else "none",
        run_name=(
            f"{os.path.basename(args.base_model)}-DPO"
            if not args.disable_wandb
            else None
        ),
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )


def main(args):
    """Main training function."""
    # Setup
    if args.hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    print("🚀 Starting DPO training")
    print(f"   - Dataset: {args.dataset}")
    print(f"   - Base model: {args.base_model}")
    print(f"   - Use LoRA: {args.use_lora}")

    # Determine model path
    if args.model_size:
        model_path = get_model_name(args.model_size)
    else:
        model_path = args.base_model

    # Load dataset
    dataset = load_dpo_dataset(args.dataset, args.max_samples)
    
    # Preprocess dataset to convert conversation format to DPO format
    dataset = preprocess_dpo_dataset(dataset)

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_path, args.use_lora)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create DPO config
    dpo_config = create_dpo_config(args, len(dataset))

    # Initialize DPO trainer
    print("🚀 Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("🚀 Starting DPO training...")
    trainer.train()

    # Save final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save training config
    config_info = {
        "dataset": args.dataset,
        "base_model": model_path,
        "use_lora": args.use_lora,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "max_steps": dpo_config.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config_info, f, indent=2)

    print("✅ DPO training completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DPO Training Script for Preference Learning"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Hugging Face dataset name " "(e.g., tech-tao/yizhipian_yizhipian_dpo_data)"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (for testing)",
    )

    # Model configuration
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=("Path to base model (local directory or " "Hugging Face model name)"),
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["0.5B", "1.5B", "3B", "8B"],
        help="Predefined model size (alternative to --base-model)",
    )
    parser.add_argument(
        "--use-lora", action="store_true", help="Use LoRA for efficient fine-tuning"
    )

    # Training configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/dpo-trained-model",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Per device batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter (temperature for preference learning)",
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max-prompt-length", type=int, default=512, help="Maximum prompt length"
    )

    # Other options
    parser.add_argument(
        "--disable-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for private datasets",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.base_model and not args.model_size:
        parser.error("Either --base-model or --model-size must be specified")

    if args.base_model and args.model_size:
        parser.error("Cannot specify both --base-model and --model-size")

    main(args)
