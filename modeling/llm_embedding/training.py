#!/usr/bin/env python3
"""
SimCSE Embedding Demo - Complete Implementation

This script demonstrates how to fine-tune a BERT model using
SimCSE (https://arxiv.org/abs/2104.08821) approach
to generate high-quality sentence embeddings.

Dataset: Uses sentence-transformers/all-nli dataset with pair subset
         - Loads anchor-positive pairs for natural contrastive learning
         - Each batch contains multiple anchor-positive pairs
         - In-batch negatives are used for contrastive learning
         - More natural than original SimCSE approach

Usage:
    uv run python modeling/llm_embedding/training.py
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import numpy as np

# Import shared model components
try:
    from .model import SimCSEModel
except ImportError:
    # Fallback for direct script execution
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from model import SimCSEModel

# -------- Configuration --------
MODEL_NAME = "bert-base-uncased"  # Use "bert-base-chinese" for Chinese
BATCH_SIZE = 128  # Reduced for better contrastive learning
LEARNING_RATE = 1e-5  # Slightly lower for more stable training
EPOCHS = 1  # More epochs to see actual learning
MAX_LEN = 64

# Dataset configuration
DATASET_NAME = "sentence-transformers/all-nli"
DATASET_SUBSET = "pair"  # Use pair subset with anchor/positive columns
DATASET_SPLIT = "train"
MAX_SAMPLES = 1000000  # Reduced for faster debugging

# SimCSE specific parameters
TEMPERATURE = 0.05  # Temperature for contrastive learning
DROPOUT_RATE = 0.1  # Dropout rate for creating different views

# Device detection for Apple Silicon, CUDA, and CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon GPU
else:
    DEVICE = "cpu"

print("🚀 SimCSE Embedding Demo")
print(f"📱 Using device: {DEVICE}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"📚 Dataset: {DATASET_NAME} ({DATASET_SUBSET}) ({MAX_SAMPLES} samples)")
print("=" * 50)


def info_nce_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = TEMPERATURE
) -> torch.Tensor:
    """
    Compute InfoNCE loss for contrastive learning using in-batch negatives.

    This version L2-normalizes embeddings so that (z1 @ z2.T) equals cosine
    similarity in [-1, 1]. Cross-entropy over the similarity logits encourages
    the i-th row to place maximum probability on the i-th column (the positive pair).

    Args:
        z1: First set of embeddings [batch_size, embedding_dim]
        z2: Second set of embeddings [batch_size, embedding_dim]
        temperature: Temperature parameter for softmax (typical values 0.03–0.1)

    Returns:
        Scalar InfoNCE loss value (>= 0)
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

    # Cross-entropy over logits (internally applies log-softmax)
    loss = F.cross_entropy(logits, labels)
    return loss


def compute_similarity_metrics(z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, float]:
    """
    Compute essential similarity metrics for training monitoring.

    Args:
        z1: First set of embeddings [batch_size, embedding_dim]
        z2: Second set of embeddings [batch_size, embedding_dim]

    Returns:
        Dictionary containing similarity metrics
    """
    with torch.no_grad():
        # L2-normalize embeddings
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = z1_norm @ z2_norm.t()

        # Extract diagonal (positive pairs) and off-diagonal (negative pairs)
        batch_size = z1.size(0)
        positive_similarities = torch.diag(similarity_matrix)
        mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        negative_similarities = similarity_matrix[~mask]

        # Compute essential metrics
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
    """
    Log batch-level metrics to console.

    Args:
        step: Current batch step
        loss: Current loss value
        metrics: Dictionary of similarity metrics
        optimizer: Optimizer for learning rate
        total_batches: Total number of batches
        progress_bar: Progress bar to update
        log_interval: How often to print detailed metrics
    """
    # Always update progress bar
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

    # Print detailed metrics every log_interval batches
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
    """
    Log epoch-level metrics to console.

    Args:
        epoch: Current epoch number
        avg_loss: Average loss for the epoch
        epoch_metrics: Aggregated metrics for the epoch
        wandb_run: Optional wandb run object (unused, kept for compatibility)
        dataloader_len: Total number of batches (unused, kept for compatibility)
    """
    print(f"\n📊 Epoch {epoch + 1} Summary:")
    print(f"  Loss: {avg_loss:.6f}")
    print(f"  Positive Sim: {epoch_metrics['positive_sim_mean']:.3f}")
    print(f"  Negative Sim: {epoch_metrics['negative_sim_mean']:.3f}")
    print(f"  Margin: {epoch_metrics['pos_neg_margin']:.3f}")


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
        """
        Initialize dataset with anchor and positive text pairs.

        Args:
            anchor_texts: List of anchor sentences
            positive_texts: List of positive (similar) sentences
        """
        assert len(anchor_texts) == len(
            positive_texts
        ), "Anchor and positive texts must have the same length"
        self.anchor_texts = anchor_texts
        self.positive_texts = positive_texts

    def __len__(self) -> int:
        return len(self.anchor_texts)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.anchor_texts[idx], self.positive_texts[idx]


def load_training_data() -> Tuple[List[str], List[str]]:
    """
    Load and prepare training data from all-nli dataset.

    Returns:
        Tuple of (anchor_texts, positive_texts) for contrastive learning
    """
    print("\n📚 Loading dataset...")
    # Load all-nli dataset with pair subset (anchor/positive pairs)
    raw_dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

    # Select minimum of actual size and MAX_SAMPLES
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


def setup_model_and_optimizer() -> Tuple[SimCSEModel, torch.optim.Optimizer]:
    """
    Initialize model and optimizer.

    Returns:
        Tuple of (model, optimizer)
    """
    print("\n🤖 Initializing model...")
    model = SimCSEModel(MODEL_NAME, MAX_LEN).to(DEVICE)

    # Enable dropout for SimCSE (critical for creating different views)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = DROPOUT_RATE  # Set dropout rate for SimCSE

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer


def create_dataloader(
    anchor_texts: List[str], positive_texts: List[str], model: SimCSEModel
) -> DataLoader:
    """
    Create DataLoader for training with anchor-positive pairs.

    Args:
        anchor_texts: List of anchor sentences
        positive_texts: List of positive sentences
        model: Model instance for tokenization

    Returns:
        Configured DataLoader
    """
    print("\n🔄 Setting up data loader...")
    dataset = AllNLIDataset(anchor_texts, positive_texts)

    def collate_fn(batch: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        """Collate function for anchor-positive pairs."""
        anchors, positives = zip(*batch)

        # Tokenize both anchor and positive texts
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

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return dataloader


def train_epoch(
    model: SimCSEModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    wandb_run: Optional = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        epoch: Current epoch number
        wandb_run: Optional wandb run object for logging

    Returns:
        Tuple of (average_loss, epoch_metrics)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Track metrics for epoch summary
    all_metrics = []

    # Progress bar
    progress_bar = tqdm(
        dataloader,
        desc=f"Training Epoch {epoch + 1}/{EPOCHS}",
        unit="batch",
        leave=True,
    )

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Forward pass - only pass required parameters
        model_inputs = {
            k: v
            for k, v in batch.items()
            if k
            in [
                "anchor_input_ids",
                "anchor_attention_mask",
                "positive_input_ids",
                "positive_attention_mask",
            ]
        }
        z1, z2 = model(**model_inputs)

        # Compute symmetric InfoNCE loss
        loss = (info_nce_loss(z1, z2) + info_nce_loss(z2, z1)) / 2.0

        # Compute similarity metrics
        metrics = compute_similarity_metrics(z1, z2)
        all_metrics.append(metrics)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        current_loss = total_loss / num_batches

        # Log batch metrics
        log_batch_metrics(
            step, current_loss, metrics, optimizer, len(dataloader), progress_bar
        )

        # Log to wandb if enabled (batch-level only)
        if wandb_run is not None:
            global_step = step + epoch * len(dataloader)
            wandb_run.log(
                {
                    "loss": loss.item(),
                    "avg_loss": current_loss,
                    "positive_sim": metrics["positive_sim_mean"],
                    "negative_sim": metrics["negative_sim_mean"],
                    "margin": metrics["pos_neg_margin"],
                },
                step=global_step,
            )

    # Compute epoch averages
    avg_loss = total_loss / num_batches
    epoch_metrics = {}
    for key in all_metrics[0].keys():
        epoch_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, epoch_metrics


def validate_model(model: SimCSEModel, dataloader: DataLoader, epoch: int) -> dict:
    """
    Validate the model on a subset of data.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        epoch: Current epoch number

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_debug_metrics = []

    print(f"\n🔍 Validating epoch {epoch + 1}...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", unit="batch"):
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward pass
            model_inputs = {
                k: v
                for k, v in batch.items()
                if k
                in [
                    "anchor_input_ids",
                    "anchor_attention_mask",
                    "positive_input_ids",
                    "positive_attention_mask",
                ]
            }
            z1, z2 = model(**model_inputs)

            # Compute loss
            loss = (info_nce_loss(z1, z2) + info_nce_loss(z2, z1)) / 2.0

            # Compute debug metrics
            debug_metrics = compute_similarity_metrics(z1, z2)
            all_debug_metrics.append(debug_metrics)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    # Aggregate debug metrics
    avg_debug = {}
    for key in all_debug_metrics[0].keys():
        if key != "loss":
            avg_debug[key] = np.mean([m[key] for m in all_debug_metrics])

    print("📊 Validation Results:")
    print(f"  Loss: {avg_loss:.6f}")
    print(f"  Positive Sim: {avg_debug['positive_sim_mean']:.3f}")
    print(f"  Negative Sim: {avg_debug['negative_sim_mean']:.3f}")
    print(f"  Margin: {avg_debug['pos_neg_margin']:.3f}")

    return {"loss": avg_loss, **avg_debug}


def compute_evaluation_metrics(
    model: SimCSEModel, wandb_run: Optional = None, epoch: int = -1
) -> dict:
    """
    Compute similarity metrics for evaluation.

    Args:
        model: Model to evaluate
        wandb_run: Optional wandb run object for logging
        epoch: Current epoch number (-1 for final evaluation)

    Returns:
        Dictionary containing similarity statistics
    """
    model.eval()

    # Test sentences
    test_sentences = [
        "What can machine learning do?",
        "How to make pizza?",
        "Machine learning applications",
        "Pizza making instructions",
        "Artificial intelligence capabilities",
        "Cooking recipes",
    ]

    with torch.no_grad():
        # Generate embeddings
        embeddings = model.encode(test_sentences, DEVICE)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = embeddings @ embeddings.t()

        # Calculate statistics
        min_sim = similarity_matrix.min().item()

        # Log to wandb if enabled (final evaluation only)
        if wandb_run is not None and epoch < 0:
            print(f"🔍 Final - Similarity: min={min_sim:.3f}")
            wandb_run.log({})

        return {
            "min_similarity": min_sim,
            "similarity_matrix": similarity_matrix,
            "test_sentences": test_sentences,
        }


def evaluate_model(model: SimCSEModel, wandb_run: Optional = None) -> None:
    """
    Evaluate the trained model on test sentences.

    Args:
        model: Trained model to evaluate
        wandb_run: Optional wandb run object for logging
    """
    print("\n🔍 Testing embeddings...")

    # Get similarity metrics
    metrics = compute_evaluation_metrics(model, wandb_run, epoch=-1)
    similarity_matrix = metrics["similarity_matrix"]
    test_sentences = metrics["test_sentences"]

    # Log detailed similarity matrix to wandb if enabled
    if wandb_run is not None:
        import wandb as wandb_pkg

        # Create a table for similarity matrix
        similarity_data = []
        for i, sent1 in enumerate(test_sentences):
            for j, sent2 in enumerate(test_sentences):
                similarity_data.append(
                    [sent1[:50], sent2[:50], similarity_matrix[i, j].item()]
                )

        wandb_run.log(
            {
                "similarity_matrix": wandb_pkg.Table(
                    columns=["Sentence 1", "Sentence 2", "Similarity"],
                    data=similarity_data,
                ),
            }
        )

    # Display similarity matrix with better formatting
    print("\n📊 Similarity Matrix:")
    print("=" * 80)
    for i, sent1 in enumerate(test_sentences):
        for j, sent2 in enumerate(test_sentences):
            sim = similarity_matrix[i, j].item()
            print(f"{sim:.3f}", end="\t")
        print(f" | {sent1[:30]}...")

    # Find most similar pairs with progress tracking
    print("\n🔗 Most Similar Pairs:")
    print("=" * 50)

    # Create pairs and sort by similarity
    pairs = []
    for i in range(len(test_sentences)):
        for j in range(i + 1, len(test_sentences)):
            sim = similarity_matrix[i, j].item()
            pairs.append((sim, i, j))

    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Display top pairs with progress bar
    for idx, (sim, i, j) in enumerate(tqdm(pairs, desc="Analyzing pairs", unit="pair")):
        print(f"#{idx + 1} Similarity {sim:.3f}:")
        print(f"  '{test_sentences[i]}'")
        print(f"  '{test_sentences[j]}'")
        print()


def save_model(model: SimCSEModel, filepath: str = "models/simcse_model.pt") -> None:
    """
    Save the trained model.

    Args:
        model: Model to save
        filepath: Path where to save the model
    """
    print("💾 Saving model...")
    torch.save(model.state_dict(), filepath)
    print(f"✅ Model saved to {filepath}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Train SimCSE embedding model")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick verification run with minimal training",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main training and demo function."""
    args = parse_args()

    # Initialize wandb if enabled
    wandb_run = None
    if args.wandb:
        try:
            import sys

            # Remove current directory from path to avoid local wandb conflict
            if "." in sys.path:
                sys.path.remove(".")

            import wandb as wandb_pkg

            print("📊 Weights & Biases logging enabled")

            # Create meaningful run name
            model_short = MODEL_NAME.split("/")[-1]
            run_name = (
                f"simcse-{model_short}-lr{LEARNING_RATE}-" f"bs{BATCH_SIZE}-ep{EPOCHS}"
            )

            wandb_run = wandb_pkg.init(
                project="simcse-embedding",
                name=run_name,
                config={
                    "model_name": MODEL_NAME,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "max_len": MAX_LEN,
                    "dataset": DATASET_NAME,
                    "max_samples": MAX_SAMPLES,
                    "device": DEVICE,
                },
            )

            # Configure wandb to show time series charts
            wandb_run.define_metric("loss", summary="min")
            wandb_run.define_metric("avg_loss", summary="min")
        except ImportError:
            print("⚠️  wandb not installed. Install with: pip install wandb")
            print("📊 Continuing without wandb logging...")
        except Exception as e:
            print(f"⚠️  Failed to initialize wandb: {e}")
            print("📊 Continuing without wandb logging...")

    # Load training data
    anchor_texts, positive_texts = load_training_data()

    # Setup model and optimizer
    model, optimizer = setup_model_and_optimizer()

    # Create data loader
    dataloader = create_dataloader(anchor_texts, positive_texts, model)

    # Training loop with enhanced progress tracking
    if args.dry_run:
        print("\n🧪 DRY RUN MODE - Quick verification")
        print("📊 Will train for 1 batch only")
        # Limit to 1 batch for dry run
        dataloader = [next(iter(dataloader))]
        epochs = 1
    else:
        epochs = EPOCHS

    print(f"\n🎯 Starting training for {epochs} epochs...")
    print(f"📊 Total batches per epoch: {len(dataloader)}")
    print(f"📈 Total training steps: {len(dataloader) * epochs}")

    # Track training history
    training_history = []

    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        avg_loss, epoch_metrics = train_epoch(
            model, dataloader, optimizer, epoch, wandb_run
        )
        training_history.append(avg_loss)
        log_epoch_metrics(epoch, avg_loss, epoch_metrics, wandb_run, len(dataloader))
        print(f"✅ Epoch {epoch + 1} completed! Average loss: {avg_loss:.4f}")

        # Show training progress
        if epoch > 0:
            loss_change = training_history[-1] - training_history[-2]
            print(f"📊 Loss change: {loss_change:+.4f}")

    print("\n✅ Training completed!")
    print(f"📊 Final average loss: {avg_loss:.4f}")
    print(f"📈 Loss history: {[f'{loss:.4f}' for loss in training_history]}")

    # Evaluate model
    evaluate_model(model, wandb_run)

    # Create models directory and save model (skip in dry run)
    if not args.dry_run:
        import os

        os.makedirs("models", exist_ok=True)
        save_model(model)

        # Model saved locally (not uploaded to wandb)
    else:
        print("\n🧪 Dry run completed - no model saved")

    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()

    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    main()
