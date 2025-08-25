#!/usr/bin/env python3
"""
Data transformation and dataset utilities for SimCSE.
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
try:
    from .config import DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT, MAX_SAMPLES, MAX_LEN, BATCH_SIZE
except ImportError:
    from config import DATASET_NAME, DATASET_SUBSET, DATASET_SPLIT, MAX_SAMPLES, MAX_LEN, BATCH_SIZE


class TextDataset(Dataset):
    """Simple dataset for text data."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class AllNLIDataset(Dataset):
    """Dataset for sentence-transformers/all-nli with anchor-positive pairs."""

    def __init__(self, anchor_texts: List[str], positive_texts: List[str]):
        assert len(anchor_texts) == len(positive_texts)
        self.anchor_texts = anchor_texts
        self.positive_texts = positive_texts

    def __len__(self) -> int:
        return len(self.anchor_texts)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.anchor_texts[idx], self.positive_texts[idx]


def load_training_data() -> Tuple[List[str], List[str]]:
    """Load and prepare training data from all-nli dataset."""
    print("\n📚 Loading dataset...")
    raw_dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)
    
    actual_size = len(raw_dataset)
    samples_to_use = min(actual_size, MAX_SAMPLES)
    
    print(f"📊 Dataset has {actual_size} total samples")
    print(f"📊 Using {samples_to_use} samples (requested: {MAX_SAMPLES})")
    
    raw_dataset = raw_dataset.select(range(samples_to_use))
    
    anchor_texts = [r["anchor"] for r in raw_dataset]
    positive_texts = [r["positive"] for r in raw_dataset]
    
    print(f"📊 Loaded {len(anchor_texts)} anchor-positive pairs")
    print(f"📝 Sample anchor: '{anchor_texts[0][:50]}...'")
    print(f"📝 Sample positive: '{positive_texts[0][:50]}...'")
    
    return anchor_texts, positive_texts


def create_dataloader(anchor_texts: List[str], positive_texts: List[str], model) -> DataLoader:
    """Create DataLoader for training with anchor-positive pairs."""
    print("\n🔄 Setting up data loader...")
    dataset = AllNLIDataset(anchor_texts, positive_texts)

    def collate_fn(batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        """Collate function for anchor-positive pairs."""
        anchors, positives = zip(*batch)

        anchor_tokens = model.tokenizer(
            anchors,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        positive_tokens = model.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            "anchor_input_ids": anchor_tokens["input_ids"],
            "anchor_attention_mask": anchor_tokens["attention_mask"],
            "positive_input_ids": positive_tokens["input_ids"],
            "positive_attention_mask": positive_tokens["attention_mask"],
        }

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )