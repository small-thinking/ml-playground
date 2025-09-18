#!/usr/bin/env python3
"""
Interactive Chat Mode for LLM Post-Training Models

This script provides an interactive chat interface for loading and testing
arbitrary models, including base models, SFT models, and LoRA fine-tuned models.

Usage:
    # Load a base model
    python chat_mode.py --model-path meta-llama/Llama-3.2-3B

    # Load an SFT model
    python chat_mode.py --model-path /workspace/models/Llama-3.2-3B-LoRA-SFT

    # Load with custom generation parameters
    python chat_mode.py --model-path meta-llama/Llama-3.2-3B --temperature 0.8 --max-length 512
"""

import argparse
import os
import json
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


class ChatInterface:
    """Interactive chat interface for LLM models."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        use_4bit: bool = False,
        use_8bit: bool = False,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the chat interface.

        Args:
            model_path: Path to model (HuggingFace ID or local path)
            device: Device to use ('cuda', 'cpu', or 'auto')
            use_4bit: Use 4-bit quantization
            use_8bit: Use 8-bit quantization
            trust_remote_code: Trust remote code execution
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trust_remote_code = trust_remote_code

        # Model components
        self.tokenizer = None
        self.model = None
        self.conversation_history: List[Dict[str, str]] = []

        # Quantization config
        self.quantization_config = None
        if use_4bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif use_8bit:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    def detect_model_type(self) -> str:
        """Detect if the model is a base model, SFT model, or LoRA model."""
        if os.path.exists(self.model_path):
            # Local path - check for LoRA adapter
            if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
                return "lora"
            elif os.path.exists(os.path.join(self.model_path, "config.json")):
                return "sft"
            else:
                return "unknown"
        else:
            # HuggingFace model ID - assume base model
            return "base"

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        print(f"🔄 Loading model from: {self.model_path}")

        # Load tokenizer
        print("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine model type and load accordingly
        model_type = self.detect_model_type()
        print(f"🔍 Detected model type: {model_type}")

        if model_type == "lora":
            self._load_lora_model()
        else:
            self._load_standard_model()

        print("✅ Model loaded successfully!")

    def _load_standard_model(self) -> None:
        """Load a standard model (base or SFT)."""
        print("🤖 Loading standard model...")

        # Determine dtype based on device
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
        )

    def _load_lora_model(self) -> None:
        """Load a LoRA model with base model."""
        print("🔧 Loading LoRA model...")

        # For LoRA models, we need to load the base model first
        # Check if there's a base model specified in adapter config
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")
        else:
            # Try to infer base model from path or use a default
            base_model_name = "meta-llama/Llama-3.2-3B"  # Default fallback

        print(f"📦 Loading base model: {base_model_name}")

        # Load base model
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
        )

        # Load LoRA adapter
        print("🔗 Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)

    def format_prompt(self, user_input: str, include_history: bool = True) -> str:
        """Format the user input into a proper prompt."""
        if not include_history or not self.conversation_history:
            # Simple prompt for first interaction
            return f"### Instruction:\n{user_input}\n\n### Response:\n"

        # Build conversation context
        context = "### Conversation History:\n"
        for entry in self.conversation_history[-3:]:  # Last 3 exchanges
            context += f"Human: {entry['user']}\n"
            context += f"Assistant: {entry['assistant']}\n\n"

        context += f"### Current Instruction:\n{user_input}\n\n### Response:\n"
        return context

    def generate_response(
        self,
        user_input: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        include_history: bool = True,
    ) -> str:
        """Generate a response from the model."""
        # Format prompt
        prompt = self.format_prompt(user_input, include_history)

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # Leave room for response
        )

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt) :].strip()

        return response

    def add_to_history(self, user_input: str, assistant_response: str) -> None:
        """Add interaction to conversation history."""
        self.conversation_history.append(
            {"user": user_input, "assistant": assistant_response}
        )

        # Keep only last 10 exchanges to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️  Conversation history cleared!")

    def show_history(self) -> None:
        """Display conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history.")
            return

        print("\n📚 Conversation History:")
        print("-" * 50)
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"{i}. Human: {entry['user']}")
            print(f"   Assistant: {entry['assistant']}")
            print()

    def run_interactive_chat(self, **generation_kwargs) -> None:
        """Run the interactive chat interface."""
        print("\n🎮 Interactive Chat Mode")
        print("=" * 50)
        print("Commands:")
        print("  /clear  - Clear conversation history")
        print("  /history - Show conversation history")
        print("  /quit   - Exit chat")
        print("  /help   - Show this help")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if user_input == "/quit":
                        print("👋 Goodbye!")
                        break
                    elif user_input == "/clear":
                        self.clear_history()
                        continue
                    elif user_input == "/history":
                        self.show_history()
                        continue
                    elif user_input == "/help":
                        print("\n🎮 Interactive Chat Mode")
                        print("=" * 50)
                        print("Commands:")
                        print("  /clear  - Clear conversation history")
                        print("  /history - Show conversation history")
                        print("  /quit   - Exit chat")
                        print("  /help   - Show this help")
                        print("=" * 50)
                        continue
                    else:
                        print("❓ Unknown command. Type /help for available commands.")
                        continue

                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.generate_response(user_input, **generation_kwargs)
                print(response)

                # Add to history
                self.add_to_history(user_input, response)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Chat Mode for LLM Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (HuggingFace ID or local path)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for inference",
    )

    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )

    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum length of generated response",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )

    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code execution",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test (runs interactive mode if not provided)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the chat script."""
    args = parse_arguments()

    # Validate model path
    if not args.model_path:
        print("❌ Error: Model path is required")
        return

    # Check if local path exists
    if not args.model_path.startswith(("http://", "https://")) and not os.path.exists(
        args.model_path
    ):
        print(f"⚠️  Warning: Local path does not exist: {args.model_path}")
        print("   Assuming it's a HuggingFace model ID...")

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🚀 LLM Chat Mode")
    print(f"   Model: {args.model_path}")
    print(f"   Device: {device}")
    print(
        f"   Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'None'}"
    )
    print("-" * 60)

    # Create chat interface
    chat = ChatInterface(
        model_path=args.model_path,
        device=device,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        trust_remote_code=args.trust_remote_code,
    )

    # Load model
    chat.load_model()

    # Generation parameters
    generation_kwargs = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": not args.no_sample,
    }

    # Run chat
    if args.prompt:
        # Single prompt mode
        print(f"\n💬 Prompt: {args.prompt}")
        print("🤖 Assistant: ", end="", flush=True)
        response = chat.generate_response(args.prompt, **generation_kwargs)
        print(response)
    else:
        # Interactive mode
        chat.run_interactive_chat(**generation_kwargs)


if __name__ == "__main__":
    main()
