#!/usr/bin/env python3
"""
Utility functions for training and evaluation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional


def compute_similarity_metrics(z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, float]:
    """
    Compute similarity metrics for training monitoring.

    Args:
        z1: First set of embeddings [batch_size, embedding_dim]
        z2: Second set of embeddings [batch_size, embedding_dim]

    Returns:
        Dictionary containing similarity metrics
    """
    with torch.no_grad():
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        similarity_matrix = z1_norm @ z2_norm.t()

        batch_size = z1.size(0)
        positive_similarities = torch.diag(similarity_matrix)
        mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        negative_similarities = similarity_matrix[~mask]

        metrics = {
            "positive_sim_mean": positive_similarities.mean().item(),
            "negative_sim_mean": negative_similarities.mean().item(),
            "pos_neg_margin": (
                positive_similarities.mean() - negative_similarities.mean()
            ).item(),
        }

        return metrics


def log_batch_metrics(
    step: int,
    loss: float,
    metrics: Dict[str, float],
    optimizer: torch.optim.Optimizer,
    total_batches: int,
    progress_bar: tqdm,
    log_interval: int = 10,
) -> None:
    """Log batch-level metrics to console."""
    progress_bar.set_postfix(
        {
            "loss": f"{loss:.4f}",
            "pos_sim": f"{metrics['positive_sim_mean']:.3f}",
            "neg_sim": f"{metrics['negative_sim_mean']:.3f}",
            "margin": f"{metrics['pos_neg_margin']:.3f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "batch": f"{step + 1}/{total_batches}",
        }
    )

    if step % log_interval == 0:
        print(
            f"\n🔍 Batch {step} - Loss: {loss:.6f}, "
            f"Pos: {metrics['positive_sim_mean']:.3f}, "
            f"Neg: {metrics['negative_sim_mean']:.3f}, "
            f"Margin: {metrics['pos_neg_margin']:.3f}"
        )


def log_epoch_metrics(
    epoch: int,
    avg_loss: float,
    epoch_metrics: Dict[str, float],
    wandb_run: Optional = None,
    dataloader_len: int = 0,
) -> None:
    """Log epoch-level metrics to console."""
    print(f"\n📊 Epoch {epoch + 1} Summary:")
    print(f"  Loss: {avg_loss:.6f}")
    print(f"  Positive Sim: {epoch_metrics['positive_sim_mean']:.3f}")
    print(f"  Negative Sim: {epoch_metrics['negative_sim_mean']:.3f}")
    print(f"  Margin: {epoch_metrics['pos_neg_margin']:.3f}")


def save_model(model, filepath: str = "models/simcse_model.pt") -> None:
    """Save the trained model."""
    print("💾 Saving model...")
    torch.save(model.state_dict(), filepath)
    print(f"✅ Model saved to {filepath}")


def evaluate_similarity_matrix(embeddings: torch.Tensor, test_sentences: list) -> None:
    """Display similarity matrix and find most similar pairs."""
    similarity_matrix = embeddings @ embeddings.t()
    
    print("\n📊 Similarity Matrix:")
    print("=" * 80)
    for i, sent1 in enumerate(test_sentences):
        for j, sent2 in enumerate(test_sentences):
            sim = similarity_matrix[i, j].item()
            print(f"{sim:.3f}", end="\t")
        print(f" | {sent1[:30]}...")

    print("\n🔗 Most Similar Pairs:")
    print("=" * 50)

    pairs = []
    for i in range(len(test_sentences)):
        for j in range(i + 1, len(test_sentences)):
            sim = similarity_matrix[i, j].item()
            pairs.append((sim, i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)

    for idx, (sim, i, j) in enumerate(tqdm(pairs, desc="Analyzing pairs", unit="pair")):
        print(f"#{idx + 1} Similarity {sim:.3f}:")
        print(f"  '{test_sentences[i]}'")
        print(f"  '{test_sentences[j]}'")
        print()