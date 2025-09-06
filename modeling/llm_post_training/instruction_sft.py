"""
SFT (Supervised Fine-Tuning) Training Script for Instruction Following

This script implements SFT training for improving instruction-following
capabilities
using the Alpaca dataset. It demonstrates how to fine-tune models to follow
instructions more effectively.

Usage:
    python instruction_sft.py --model-size 3B --use-lora
    python instruction_sft.py --model-size 8B --no-lora
    python instruction_sft.py --disable-wandb
    python instruction_sft.py --help

Arguments:
    --model-size: Model size to use ("0.5B", "1.5B", "3B", "8B") [default: 3B]
    --use-lora: Enable LoRA for efficient fine-tuning [default: False]
    --disable-wandb: Disable wandb logging [default: False]
    --max-steps: Maximum training steps [default: 1000]
    --batch-size: Training batch size [default: 4]
    --learning-rate: Learning rate [default: 2e-5]
    --hf-token: Hugging Face token for accessing gated repositories
    [default: None]

Examples:
    # Full fine-tuning with 3B model
    python instruction_sft.py --model-size 3B

    # LoRA fine-tuning with 8B model
    python instruction_sft.py --model-size 8B --use-lora

    # Custom training configuration
    python instruction_sft.py --model-size 1.5B --max-steps 2000 --batch-size 8

    # With Hugging Face token for gated models
    python instruction_sft.py --model-size 3B --hf-token your_token_here
"""

import os
import argparse
import math
from datetime import datetime
from typing import Optional, Dict, Any

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


class SFTMetricsTrainer(Trainer):
    """Custom trainer that computes additional SFT metrics."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute loss and additional metrics."""
        outputs = model(**inputs)
        loss = outputs.loss

        if return_outputs:
            return loss, outputs

        return loss

    def log(self, logs):
        """Log additional metrics including perplexity."""
        # Compute perplexity from loss
        if "train_loss" in logs:
            logs["train_perplexity"] = math.exp(logs["train_loss"])
        if "eval_loss" in logs:
            logs["eval_perplexity"] = math.exp(logs["eval_loss"])

        # Compute token accuracy if we have predictions
        if "eval_predictions" in logs and "eval_labels" in logs:
            predictions = logs["eval_predictions"]
            labels = logs["eval_labels"]

            # Flatten and compute accuracy
            predictions_flat = predictions.flatten()
            labels_flat = labels.flatten()

            # Mask out padding tokens (typically -100)
            mask = labels_flat != -100
            if mask.sum() > 0:
                correct = (predictions_flat[mask] == labels_flat[mask]).sum()
                total = mask.sum()
                logs["eval_token_accuracy"] = correct.float() / total

        super().log(logs)


class InstructionSFTTrainer:
    """Main trainer class for SFT instruction following training."""

    def __init__(
        self,
        model_size: str = "3B",
        use_lora: bool = False,
        wandb_enabled: bool = True,
        max_steps: int = 10000,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the trainer with model configuration.

        Args:
            model_size: Size of the model ("0.5B", "1.5B", "3B", "8B")
            use_lora: Whether to use LoRA for efficient fine-tuning
            wandb_enabled: Whether to enable wandb logging
            max_steps: Maximum training steps
            batch_size: Training batch size
            learning_rate: Learning rate for training
            hf_token: Hugging Face token for accessing gated repositories
        """
        self.model_size = model_size
        self.use_lora = use_lora
        self.wandb_enabled = wandb_enabled
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hf_token = hf_token
        self.model_name = self._get_model_name()

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.dataset = None

        # Setup workspace directories
        self.workspace_dir = os.environ.get("WORKSPACE_DIR", "/workspace")
        self.models_dir = os.path.join(self.workspace_dir, "models")
        self.data_dir = os.path.join(self.workspace_dir, "data")
        self.cache_dir = os.path.join(self.workspace_dir, "cache")

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set environment variables for HuggingFace
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HF_HUB_CACHE"] = self.models_dir
        os.environ["HF_DATASETS_CACHE"] = self.data_dir

        # Set Hugging Face token if provided
        if self.hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token

        # Setup logging
        self.log_dir = "debug_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"sft_debug_{timestamp}.txt")

    def _get_model_name(self) -> str:
        """Get the model name based on size."""
        model_mapping = {
            "8B": "meta-llama/Llama-3.1-8B",  # Base model, not instruct
            "3B": "meta-llama/Llama-3.2-3B",  # Base model, not instruct
            "1.5B": "Qwen/Qwen2-1.5B",  # Base model, not instruct
            "0.5B": "Qwen/Qwen2-0.5B",  # Base model, not instruct
        }

        if self.model_size not in model_mapping:
            raise ValueError(f"Invalid model size: {self.model_size}")

        return model_mapping[self.model_size]

    def load_model_and_tokenizer(self) -> None:
        """Load the model and tokenizer."""
        print(f"🔄 Loading model and tokenizer: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.models_dir,
            trust_remote_code=True,
        )

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.models_dir,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Apply LoRA if enabled
        if self.use_lora:
            self._apply_lora()

        # Ensure parameters are in correct precision for mixed precision
        self._ensure_correct_precision()

    def _apply_lora(self) -> None:
        """Apply LoRA configuration to the model."""
        print("🔧 Applying LoRA configuration...")

        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha parameter
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Target modules for different model architectures
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Fix for FP16 gradients error: ensure trainable parameters are in FP32
        # when using mixed precision training
        print(
            "🔧 Converting trainable parameters to FP32 for "
            "mixed precision compatibility..."
        )
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    def _ensure_correct_precision(self) -> None:
        """
        Ensure model parameters are in correct precision for mixed precision.

        This method fixes the 'Attempting to unscale FP16 gradients' error by
        ensuring that all trainable parameters are in FP32 when using mixed
        precision training. AMP expects trainable parameters to be in FP32 and
        handles casting to FP16 during the forward pass.
        """
        # Only apply when using GPU with mixed precision
        if torch.cuda.is_available():
            print("🔧 Ensuring correct precision for mixed precision training...")
            trainable_params = 0
            converted_params = 0

            for param in self.model.parameters():
                if param.requires_grad:
                    trainable_params += 1
                    if param.dtype == torch.float16:
                        param.data = param.data.float()
                        converted_params += 1

            if converted_params > 0:
                print(
                    f"   Converted {converted_params}/{trainable_params} "
                    f"trainable parameters from FP16 to FP32"
                )
            else:
                print(
                    f"   All {trainable_params} trainable parameters "
                    f"already in correct precision"
                )

    def load_and_prepare_dataset(self) -> None:
        """Load and prepare the Alpaca dataset."""
        print("📊 Loading Alpaca dataset...")

        # Load the Alpaca dataset
        self.dataset = load_dataset("tatsu-lab/alpaca", split="train")

        # Take a subset for faster training (optional)
        if len(self.dataset) > 10000:
            self.dataset = self.dataset.select(range(10000))
            print(
                f"📊 Using subset of {len(self.dataset)} examples for "
                f"faster training"
            )

        # Format the dataset
        self.dataset = self.dataset.map(
            self._format_instruction,
            remove_columns=self.dataset.column_names,
        )

        # Tokenize the dataset
        self.dataset = self.dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset",
        )

    def _format_instruction(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Format instruction following examples.

        Args:
            example: Single example from the dataset

        Returns:
            Formatted example with instruction and response
        """
        # Create instruction-following format
        if example.get("input", "").strip():
            # Has input field
            instruction = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            # No input field
            instruction = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )

        return {"text": instruction}

    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize the formatted examples.

        Args:
            examples: Batch of examples

        Returns:
            Tokenized examples
        """
        # Tokenize the text with proper configuration
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,  # Truncate to max_length
            padding="max_length",  # Pad to max_length
            max_length=512,
            return_tensors="pt",  # Return PyTorch tensors
            add_special_tokens=True,
        )

        # For SFT, we use the same input as labels (teacher forcing)
        # Since we're using return_tensors="pt", input_ids will be PyTorch
        # tensors
        # Use clone() for PyTorch tensors instead of copy()
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    def get_training_arguments(self) -> TrainingArguments:
        """Get training arguments configuration."""
        # Create output directory in models folder
        model_name_short = self.model_name.split("/")[-1]
        lora_suffix = "LoRA" if self.use_lora else "Full"
        model_output_name = f"{model_name_short}-Base-{lora_suffix}-SFT"
        output_dir = os.path.join(self.models_dir, model_output_name)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=100,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb" if self.wandb_enabled else "none",
            run_name=(
                f"{model_name_short}-Base-{lora_suffix}-SFT"
                if self.wandb_enabled
                else None
            ),
            dataloader_pin_memory=False,
        )

        return training_args

    def print_directory_info(self) -> None:
        """Print information about workspace directories."""
        print("📁 Workspace Configuration:")
        print(f"   Workspace Directory: {self.workspace_dir}")
        print(f"   Models Directory: {self.models_dir}")
        print(f"   Data Directory: {self.data_dir}")
        print(f"   Cache Directory: {self.cache_dir}")
        print(f"   Model: {self.model_name}")
        dataset_size = len(self.dataset) if self.dataset else "Not loaded"
        print(f"   Dataset Size: {dataset_size}")
        print("-" * 50)

    def train(self) -> None:
        """Execute the training process."""
        # Print directory information
        self.print_directory_info()

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Load and prepare dataset
        self.load_and_prepare_dataset()

        # Get training arguments
        training_args = self.get_training_arguments()

        # Create data collator with proper padding configuration
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        )

        # Initialize trainer with proper configuration
        trainer = SFTMetricsTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator,
            # Use processing_class instead of deprecated tokenizer parameter
            processing_class=self.tokenizer,
        )

        # Validate dataset before training
        print("🔍 Validating dataset...")
        sample_batch = next(iter(trainer.get_train_dataloader()))
        print(f"   Batch keys: {sample_batch.keys()}")
        print(f"   Input IDs shape: {sample_batch['input_ids'].shape}")
        print(f"   Labels shape: {sample_batch['labels'].shape}")
        print("✅ Dataset validation passed")

        # Start training
        print("🚀 Starting SFT training...")
        trainer.train()

        # Save the final model
        print("💾 Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)

        print("✅ Training completed successfully!")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SFT Training Script for Instruction Following",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["0.5B", "1.5B", "3B", "8B"],
        default="3B",
        help="Model size to use for training",
    )

    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA for efficient fine-tuning",
    )

    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Maximum training steps"
    )

    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for training",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for accessing gated repositories",
    )

    return parser.parse_args()


def main():
    """Main entry point for the training script."""
    # Parse command-line arguments
    args = parse_arguments()

    print("🚀 Starting SFT training with:")
    print(f"   Model: {args.model_size}")
    lora_status = "Enabled" if args.use_lora else "Disabled"
    wandb_status = "Disabled" if args.disable_wandb else "Enabled"
    hf_token_status = "Provided" if args.hf_token else "Not provided"
    print(f"   LoRA: {lora_status}")
    print(f"   Wandb: {wandb_status}")
    print(f"   HF Token: {hf_token_status}")
    print(f"   Max Steps: {args.max_steps}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    print("-" * 50)

    # Create and run trainer
    trainer = InstructionSFTTrainer(
        model_size=args.model_size,
        use_lora=args.use_lora,
        wandb_enabled=not args.disable_wandb,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hf_token=args.hf_token,
    )

    trainer.train()


if __name__ == "__main__":
    main()
