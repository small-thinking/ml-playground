#!/usr/bin/env python3
"""
Convert silk-road/alpaca-data-gpt4-chinese dataset to standard alpaca format.

Original format (6 columns):
- instruction_zh, input_zh, output_zh (Chinese)
- instruction, input, output (English)

Output format (3 columns):
- instruction, input, output

Each original row becomes 2 rows: one English, one Chinese.

Usage:
    python convert_silk_road_to_alpaca.py
"""

import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from dotenv import load_dotenv


def load_hf_token() -> str:
    """Load Hugging Face token from environment variables."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please set it in .env file.")
    return hf_token


def convert_to_alpaca_format(dataset: Dataset) -> Dataset:
    """
    Convert 6-column dataset to 3-column alpaca format.
    Each original row becomes 2 rows (English + Chinese).
    """
    print("Converting dataset to alpaca format...")

    # Convert to pandas
    df = pd.DataFrame(dataset)

    # Create English rows
    english_df = pd.DataFrame(
        {"instruction": df["instruction"], "input": df["input"], "output": df["output"]}
    )

    # Create Chinese rows
    chinese_df = pd.DataFrame(
        {
            "instruction": df["instruction_zh"],
            "input": df["input_zh"],
            "output": df["output_zh"],
        }
    )

    # Combine both datasets
    alpaca_df = pd.concat([english_df, chinese_df], ignore_index=True)

    # Remove empty rows
    alpaca_df = alpaca_df[
        (alpaca_df["instruction"].str.strip() != "")
        & (alpaca_df["output"].str.strip() != "")
    ].reset_index(drop=True)

    print(f"Converted {len(df)} rows to {len(alpaca_df)} rows")
    return Dataset.from_pandas(alpaca_df)


def upload_to_huggingface(dataset: Dataset, repo_name: str, hf_token: str) -> str:
    """Upload dataset to Hugging Face Hub."""
    print(f"Uploading to Hugging Face: {repo_name}")

    login(token=hf_token)
    api = HfApi()

    # Get username
    user_info = api.whoami(token=hf_token)
    username = user_info["name"]
    full_repo_id = f"{username}/{repo_name}"

    # Create and upload
    api.create_repo(
        repo_id=full_repo_id, repo_type="dataset", token=hf_token, exist_ok=True
    )

    dataset.push_to_hub(repo_id=full_repo_id, token=hf_token)
    print(f"Uploaded to: https://huggingface.co/datasets/{full_repo_id}")

    return full_repo_id


def main():
    """Main conversion function."""
    try:
        # Load token and dataset
        hf_token = load_hf_token()
        original_dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese")["train"]

        print(f"Original dataset: {len(original_dataset)} rows")

        # Convert to alpaca format
        alpaca_dataset = convert_to_alpaca_format(original_dataset)

        # Upload to Hugging Face
        repo_id = upload_to_huggingface(alpaca_dataset, "alpaca-bilingual", hf_token)

        print(f"\n✅ Success!")
        print(f"📊 Original: {len(original_dataset)} rows")
        print(f"📊 Converted: {len(alpaca_dataset)} rows")
        print(f"🔗 Dataset: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
