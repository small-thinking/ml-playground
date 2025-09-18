#!/usr/bin/env python3
"""
Utility functions for model management and discovery.

This module provides helper functions for working with trained models,
including model discovery, validation, and metadata extraction.
"""

import os
import json
from typing import List, Dict, Any, Optional


def find_models_in_directory(directory: str = "./models") -> List[Dict[str, Any]]:
    """
    Find all trained models in a directory.

    Args:
        directory: Directory to search for models

    Returns:
        List of model information dictionaries
    """
    models = []

    if not os.path.exists(directory):
        return models

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            model_info = get_model_info(item_path)
            if model_info:
                models.append(model_info)

    return models


def get_model_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a model.

    Args:
        model_path: Path to the model directory

    Returns:
        Dictionary with model information or None if invalid
    """
    if not os.path.exists(model_path):
        return None

    info = {
        "path": model_path,
        "name": os.path.basename(model_path),
        "type": "unknown",
        "base_model": None,
        "training_type": None,
        "config": None,
    }

    # Check for config.json
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                info["config"] = config
                info["type"] = "sft"
        except Exception:
            pass

    # Check for LoRA adapter
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
                info["type"] = "lora"
                info["base_model"] = adapter_config.get("base_model_name_or_path")
                info["training_type"] = adapter_config.get("task_type", "unknown")
        except Exception:
            pass

    # Determine training type from name
    name_lower = info["name"].lower()
    if "sft" in name_lower:
        info["training_type"] = "sft"
    elif "grpo" in name_lower:
        info["training_type"] = "grpo"
    elif "lora" in name_lower:
        info["training_type"] = "lora"

    return info


def list_available_models(directory: str = "./models") -> None:
    """
    List all available models in a directory.

    Args:
        directory: Directory to search for models
    """
    models = find_models_in_directory(directory)

    if not models:
        print(f"📁 No models found in {directory}")
        return

    print(f"📁 Available models in {directory}:")
    print("-" * 80)

    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Type: {model['type'].upper()}")
        print(f"   Training: {model['training_type'] or 'Unknown'}")
        if model["base_model"]:
            print(f"   Base Model: {model['base_model']}")
        print(f"   Path: {model['path']}")
        print()


def validate_model_path(model_path: str) -> bool:
    """
    Validate that a model path contains the necessary files.

    Args:
        model_path: Path to validate

    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(model_path):
        return False

    # Check for essential files
    essential_files = ["config.json"]
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]

    has_essential = any(
        os.path.exists(os.path.join(model_path, f)) for f in essential_files
    )
    has_tokenizer = any(
        os.path.exists(os.path.join(model_path, f)) for f in tokenizer_files
    )

    return has_essential and has_tokenizer


def get_model_size_info(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Get size information about a model.

    Args:
        model_path: Path to the model

    Returns:
        Dictionary with size information or None if unavailable
    """
    if not os.path.exists(model_path):
        return None

    size_info = {
        "total_size_mb": 0,
        "file_count": 0,
        "largest_file": None,
        "largest_file_size_mb": 0,
    }

    total_size = 0
    file_count = 0
    largest_file = None
    largest_size = 0

    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1

                if file_size > largest_size:
                    largest_size = file_size
                    largest_file = file
            except OSError:
                continue

    size_info["total_size_mb"] = total_size / (1024 * 1024)
    size_info["file_count"] = file_count
    size_info["largest_file"] = largest_file
    size_info["largest_file_size_mb"] = largest_size / (1024 * 1024)

    return size_info


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Model management utilities")
    parser.add_argument(
        "--directory",
        type=str,
        default="./models",
        help="Directory to search for models",
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Validate a specific model path",
    )
    parser.add_argument(
        "--size-info",
        type=str,
        help="Get size information for a specific model",
    )

    args = parser.parse_args()

    if args.validate:
        is_valid = validate_model_path(args.validate)
        print(f"Model validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        return

    if args.size_info:
        size_info = get_model_size_info(args.size_info)
        if size_info:
            print(f"Model size information for {args.size_info}:")
            print(f"  Total size: {size_info['total_size_mb']:.2f} MB")
            print(f"  File count: {size_info['file_count']}")
            print(
                f"  Largest file: {size_info['largest_file']} ({size_info['largest_file_size_mb']:.2f} MB)"
            )
        else:
            print("❌ Could not get size information")
        return

    # Default: list available models
    list_available_models(args.directory)


if __name__ == "__main__":
    main()
