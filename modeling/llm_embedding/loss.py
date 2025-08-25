#!/usr/bin/env python3
"""
Loss functions for SimCSE contrastive learning.
"""

import torch
import torch.nn.functional as F
from .config import TEMPERATURE


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    Compute InfoNCE loss for contrastive learning using in-batch negatives.

    Args:
        z1: First set of embeddings [batch_size, embedding_dim]
        z2: Second set of embeddings [batch_size, embedding_dim]
        temperature: Temperature parameter for softmax

    Returns:
        Scalar InfoNCE loss value
    """
    assert z1.dim() == 2 and z2.dim() == 2, "Embeddings must be 2D (batch, dim)."
    assert z1.size(0) == z2.size(0), "z1 and z2 must have the same batch size."

    # L2-normalize to use cosine similarity and stabilize training
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Similarity matrix: [batch_size, batch_size]
    logits = (z1 @ z2.t()) / temperature

    # Positive pairs are on the diagonal
    labels = torch.arange(z1.size(0), device=z1.device)

    # Cross-entropy over logits
    loss = F.cross_entropy(logits, labels)
    return loss