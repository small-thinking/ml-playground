#!/usr/bin/env python3
"""
Comparison script to demonstrate the difference between base and SFT models.

This script loads both a base model and its SFT-trained version to show
how supervised fine-tuning transforms model behavior.

Usage:
    python compare_base_vs_sft.py \
        --base-model meta-llama/Llama-3.2-3B \
        --sft-model /workspace/models/Llama-3.2-3B-Full-SFT
"""

import argparse
import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelComparator:
    """Compare base model vs SFT model behavior."""

    def __init__(
        self,
        base_model_name: str,
        sft_model_path: str,
        device: Optional[str] = None,
    ):
        """
        Initialize the model comparator.

        Args:
            base_model_name: Name of the base model
            (e.g., "meta-llama/Llama-3.2-3B")
            sft_model_path: Path to the SFT-trained model
            device: Device to use for inference (auto-detect if None)
        """
        self.base_model_name = base_model_name
        self.sft_model_path = sft_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model components
        self.base_tokenizer = None
        self.base_model = None
        self.sft_tokenizer = None
        self.sft_model = None

    def load_models(self) -> None:
        """Load both base and SFT models."""
        print("🔄 Loading base model...")
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        print("🔄 Loading SFT model...")
        self.sft_tokenizer = AutoTokenizer.from_pretrained(
            self.sft_model_path,
            trust_remote_code=True,
        )
        self.sft_model = AutoModelForCausalLM.from_pretrained(
            self.sft_model_path,
            torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        print("✅ Both models loaded successfully!")

    def generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate response from a model."""
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length // 2,
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt) :].strip()

    def compare_instruction_following(
        self,
        instruction: str,
        input_text: str = "",
        max_length: int = 256,
    ) -> None:
        """
        Compare how base and SFT models handle instruction following.

        Args:
            instruction: The instruction to test
            input_text: Optional input context
            max_length: Maximum length of generated text
        """
        print("\n" + "=" * 80)
        print("🧪 INSTRUCTION FOLLOWING COMPARISON")
        print("=" * 80)
        print(f"Instruction: {instruction}")
        if input_text:
            print(f"Input: {input_text}")
        print("\n" + "-" * 80)

        # Format prompt for SFT model
        if input_text.strip():
            sft_prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            sft_prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Simple prompt for base model
        base_prompt = f"{instruction}\n\n"

        # Generate responses
        print("🤖 BASE MODEL RESPONSE:")
        print("-" * 40)
        try:
            base_response = self.generate_response(
                self.base_model, self.base_tokenizer, base_prompt, max_length
            )
            print(base_response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n🎯 SFT MODEL RESPONSE:")
        print("-" * 40)
        try:
            sft_response = self.generate_response(
                self.sft_model, self.sft_tokenizer, sft_prompt, max_length
            )
            print(sft_response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print("=" * 80)

    def run_demo_comparisons(self) -> None:
        """Run a series of demonstration comparisons."""
        demo_instructions = [
            "Write a haiku about artificial intelligence",
            "Explain the concept of machine learning in simple terms",
            "Write a short story about a robot learning to paint",
            "List 5 benefits of renewable energy",
            "Write a Python function to calculate fibonacci numbers",
        ]

        print("🎬 Running demonstration comparisons...")
        print("This will show how SFT transforms base model behavior")
        print("-" * 80)

        for i, instruction in enumerate(demo_instructions, 1):
            print(f"\n📝 Demo {i}/{len(demo_instructions)}")
            self.compare_instruction_following(instruction)

    def run_interactive_mode(self) -> None:
        """Run interactive comparison mode."""
        print("\n🎮 Interactive Comparison Mode")
        print("Type 'quit' to exit, 'demo' for demonstration comparisons")
        print("-" * 60)

        while True:
            try:
                instruction = input("\n📝 Enter instruction: ").strip()

                if instruction.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                elif instruction.lower() == "demo":
                    self.run_demo_comparisons()
                    continue
                elif not instruction:
                    continue

                # Ask for optional input
                input_text = input(
                    "📥 Enter input (optional, press Enter to skip): "
                ).strip()

                self.compare_instruction_following(instruction, input_text)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare base model vs SFT model behavior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Name of the base model (e.g., meta-llama/Llama-3.2-3B)",
    )

    parser.add_argument(
        "--sft-model",
        type=str,
        required=True,
        help="Path to the SFT-trained model directory",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for inference",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Single instruction to test (runs interactive mode if not provided)",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input context for the instruction",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration comparisons",
    )

    return parser.parse_args()


def main():
    """Main entry point for the comparison script."""
    args = parse_arguments()

    # Validate SFT model path
    if not os.path.exists(args.sft_model):
        print(f"❌ Error: SFT model path does not exist: {args.sft_model}")
        return

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚀 Base Model vs SFT Model Comparison")
    print(f"   Base Model: {args.base_model}")
    print(f"   SFT Model: {args.sft_model}")
    print(f"   Device: {device}")
    print("-" * 60)

    # Create comparator and load models
    comparator = ModelComparator(args.base_model, args.sft_model, device)
    comparator.load_models()

    # Run comparison
    if args.demo:
        # Run demonstration comparisons
        comparator.run_demo_comparisons()
    elif args.instruction:
        # Single instruction comparison
        comparator.compare_instruction_following(args.instruction, args.input)
    else:
        # Interactive mode
        comparator.run_interactive_mode()


if __name__ == "__main__":
    main()
