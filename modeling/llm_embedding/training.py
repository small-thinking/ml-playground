#!/usr/bin/env python3
"""
SimCSE Training Script

Core training functionality for SimCSE embedding model.
Usage:
    uv run python modeling/llm_embedding/training.py
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, Optional

# Import components
try:
    from .model import SimCSEModel
    from .config import *
    from .data import load_training_data, create_dataloader
    from .loss import info_nce_loss
    from .utils import compute_similarity_metrics, log_batch_metrics, log_epoch_metrics, save_model
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from model import SimCSEModel
    from config import *
    from data import load_training_data, create_dataloader
    from loss import info_nce_loss
    from utils import compute_similarity_metrics, log_batch_metrics, log_epoch_metrics, save_model

print("🚀 SimCSE Embedding Demo")
print(f"📱 Using device: {DEVICE}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"📚 Dataset: {DATASET_NAME} ({DATASET_SUBSET}) ({MAX_SAMPLES} samples)")
print("=" * 50)












def setup_model_and_optimizer() -> Tuple[SimCSEModel, torch.optim.Optimizer]:
    """Initialize model and optimizer."""
    print("\n🤖 Initializing model...")
    model = SimCSEModel(MODEL_NAME, MAX_LEN).to(DEVICE)

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = DROPOUT_RATE

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer


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






def evaluate_model(model: SimCSEModel, wandb_run: Optional[object] = None) -> None:
    """Evaluate the trained model on test sentences."""
    print("\n🔍 Testing embeddings...")
    
    test_sentences = [
        "What can machine learning do?",
        "How to make pizza?", 
        "Machine learning applications",
        "Pizza making instructions",
        "Artificial intelligence capabilities",
        "Cooking recipes",
    ]
    
    with torch.no_grad():
        embeddings = model.encode(test_sentences, DEVICE)
        
    try:
        from .utils import evaluate_similarity_matrix
    except ImportError:
        from utils import evaluate_similarity_matrix
    evaluate_similarity_matrix(embeddings, test_sentences)




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
