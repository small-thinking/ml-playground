#!/usr/bin/env python3
"""
Example script demonstrating vocabulary dump functionality.

This script shows how to use the vocab_inspect.py tool to dump
a model's vocabulary to a JSON file.
"""

import subprocess
import sys
import os


def dump_vocabulary_example():
    """Example of dumping vocabulary from a model."""

    # Example model paths (adjust based on your setup)
    model_paths = [
        "Qwen/Qwen2-0.5B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "distilbert-base-uncased",
    ]

    print("🔍 Vocabulary Dump Examples")
    print("=" * 50)

    for model_path in model_paths:
        output_file = f"vocab_{model_path.replace('/', '_')}.json"

        print(f"\n📝 Dumping vocabulary for: {model_path}")
        print(f"📁 Output file: {output_file}")

        # Build command
        cmd = [
            "python",
            "vocab_inspect.py",
            "--model-path",
            model_path,
            "--dump-vocab",
            output_file,
            "--no-embeddings",  # Don't load full model for faster processing
        ]

        try:
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Successfully dumped vocabulary")
                print(f"📊 Output: {result.stdout}")
            else:
                print(f"❌ Error: {result.stderr}")

        except Exception as e:
            print(f"❌ Exception: {e}")

        print("-" * 30)


def show_vocab_structure():
    """Show the structure of the dumped vocabulary JSON."""

    example_structure = {
        "model_path": "Qwen/Qwen2-0.5B-Instruct",
        "vocab_size": 152064,
        "special_tokens": {
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
            "sep_token": "<|eot_id|>",
        },
        "vocabulary": [
            {
                "token_id": 0,
                "token": "<|begin_of_text|>",
                "length": 15,
                "is_special": True,
                "is_punctuation": False,
                "is_digit": False,
                "is_alpha": False,
                "is_whitespace": False,
                "is_subword": False,
            },
            {
                "token_id": 1,
                "token": "<|end_of_text|>",
                "length": 13,
                "is_special": True,
                "is_punctuation": False,
                "is_digit": False,
                "is_alpha": False,
                "is_whitespace": False,
                "is_subword": False,
            },
            # ... more tokens
        ],
    }

    print("\n📋 Vocabulary JSON Structure:")
    print("=" * 50)
    print("The dumped vocabulary JSON contains:")
    print("• model_path: Original model path")
    print("• vocab_size: Total number of tokens")
    print("• special_tokens: Special token mappings")
    print("• vocabulary: Array of token objects with:")
    print("  - token_id: Numeric ID of the token")
    print("  - token: The actual token string")
    print("  - length: Character length of token")
    print("  - is_special: Whether it's a special token")
    print("  - is_punctuation: Whether it's punctuation")
    print("  - is_digit: Whether it's a digit")
    print("  - is_alpha: Whether it's alphabetic")
    print("  - is_whitespace: Whether it's whitespace")
    print("  - is_subword: Whether it's a subword token")


if __name__ == "__main__":
    print("🚀 Vocabulary Inspector - Dump Examples")
    print("=" * 60)

    # Show structure first
    show_vocab_structure()

    # Ask user if they want to run examples
    response = input("\n❓ Run vocabulary dump examples? (y/n): ").lower().strip()

    if response in ["y", "yes"]:
        dump_vocabulary_example()
    else:
        print("👋 Skipping examples. Use the following command to dump vocabularies:")
        print("\nExample commands:")
        print(
            "python vocab_inspect.py --model-path Qwen/Qwen2-0.5B-Instruct --dump-vocab qwen_0.5b_vocab.json"
        )
        print(
            "python vocab_inspect.py --model-path meta-llama/Llama-3.2-3B-Instruct --dump-vocab llama_vocab.json"
        )
        print(
            "python vocab_inspect.py --model-path distilbert-base-uncased --dump-vocab distilbert_vocab.json"
        )
