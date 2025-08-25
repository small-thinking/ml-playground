#!/usr/bin/env python3
"""
Configuration for SimCSE embedding model.
"""

import torch

# Model configuration
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64

# Training configuration
BATCH_SIZE = 512
LEARNING_RATE = 1e-5
EPOCHS = 1
DROPOUT_RATE = 0.1

# Dataset configuration
DATASET_NAME = "sentence-transformers/all-nli"
DATASET_SUBSET = "pair"
DATASET_SPLIT = "train"
MAX_SAMPLES = 1000000

# SimCSE specific parameters
TEMPERATURE = 0.05


# Device detection
def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


DEVICE = get_device()
