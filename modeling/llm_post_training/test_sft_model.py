#!/usr/bin/env python3
"""
Test script for trained SFT models.

This script demonstrates how to load and test a trained SFT model
for instruction following tasks.

Usage:
    python test_sft_model.py --model-path /path/to/trained/model
    python test_sft_model.py --model-path /workspace/models/Llama-3.2-3B-Base-LoRA-SFT
"""

import argparse
import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SFTModelTester:
    """Test class for trained SFT models."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the model tester.

        Args:
            model_path: Path to the trained model directory
            device: Device to use for inference (auto-detect if None)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load the trained model and tokenizer."""
        print(f"🔄 Loading model from: {self.model_path}")
        print(f"🖥️  Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        print("✅ Model loaded successfully!")

    def format_instruction(self, instruction: str, input_text: str = "") -> str:
        """
        Format instruction following the training format.

        Args:
            instruction: The instruction to follow
            input_text: Optional input context

        Returns:
            Formatted prompt
        """
        if input_text.strip():
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def generate_response(
        self,
        instruction: str,
        input_text: str = "",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response for the given instruction.

        Args:
            instruction: The instruction to follow
            input_text: Optional input context
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Format the prompt
        prompt = self.format_instruction(instruction, input_text)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length // 2,  # Leave room for response
        )

        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part
        if "### Response:\n" in full_response:
            response = full_response.split("### Response:\n")[-1].strip()
        else:
            response = full_response[len(prompt) :].strip()

        return response

    def test_instruction(self, instruction: str, input_text: str = "") -> None:
        """
        Test a single instruction and print the result.

        Args:
            instruction: The instruction to test
            input_text: Optional input context
        """
        print("\n" + "=" * 60)
        print("🧪 TESTING INSTRUCTION")
        print("=" * 60)
        print(f"Instruction: {instruction}")
        if input_text:
            print(f"Input: {input_text}")
        print("\n🤖 Generated Response:")
        print("-" * 40)

        try:
            response = self.generate_response(instruction, input_text)
            print(response)
        except Exception as e:
            print(f"❌ Error generating response: {e}")

        print("=" * 60)

    def run_interactive_mode(self) -> None:
        """Run interactive testing mode."""
        print("\n🎮 Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 40)

        while True:
            try:
                instruction = input("\n📝 Enter instruction: ").strip()

                if instruction.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break
                elif instruction.lower() == "help":
                    print("\n📖 Available commands:")
                    print("  - Type any instruction to test it")
                    print("  - Use 'quit' or 'exit' to stop")
                    print("  - Example: 'Write a haiku about AI'")
                    continue
                elif not instruction:
                    continue

                # Ask for optional input
                input_text = input(
                    "📥 Enter input (optional, press Enter to skip): "
                ).strip()

                self.test_instruction(instruction, input_text)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test trained SFT models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory",
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
        help="Single instruction to test (if not provided, runs interactive mode)",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input context for the instruction",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum length of generated text",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    return parser.parse_args()


def main():
    """Main entry point for the test script."""
    args = parse_arguments()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}")
        return

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚀 SFT Model Tester")
    print(f"   Model Path: {args.model_path}")
    print(f"   Device: {device}")
    print(f"   Max Length: {args.max_length}")
    print(f"   Temperature: {args.temperature}")
    print("-" * 50)

    # Create tester and load model
    tester = SFTModelTester(args.model_path, device)
    tester.load_model()

    # Run test
    if args.instruction:
        # Single instruction test
        tester.test_instruction(args.instruction, args.input)
    else:
        # Interactive mode
        tester.run_interactive_mode()


if __name__ == "__main__":
    main()
