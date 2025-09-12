"""Training script for auto-encoder model.

Can be run from repo root:
    python -m modeling.generation.ae.train_autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from typing import List, Optional
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Get the current working directory (should be repo root when run as module)
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import the modules
from modeling.generation.ae.autoencoder import create_autoencoder
from modeling.generation.celeba_dataloader import create_celeba_dataloader

logger = logging.getLogger(__name__)


class AutoEncoderTrainer:
    """Trainer class for auto-encoder model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = "checkpoints",
        wandb_enabled: bool = True,
        project_name: str = "autoencoder-training",
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Auto-encoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            save_dir: Directory to save checkpoints
            wandb_enabled: Whether to enable wandb logging
            project_name: Name of the wandb project
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.wandb_enabled = wandb_enabled
        self.project_name = project_name

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.current_epoch = 0

        # Initialize wandb if enabled
        if self.wandb_enabled:
            # Create meaningful run name with key hyperparameters
            run_name = f"ae_latent{model.latent_dim}_lr{learning_rate}_bs{train_loader.batch_size}"
            logger.info(
                f"Initializing wandb with project: {self.project_name}, run: {run_name}"
            )
            wandb.init(
                project=self.project_name,
                name=run_name,
                mode="online",  # Explicitly set to online mode
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": train_loader.batch_size,
                    "device": str(device),
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "latent_dim": model.latent_dim,
                },
            )
            logger.info("Wandb initialized successfully")

            # Define metrics for wandb
            wandb.define_metric("epoch")
            wandb.define_metric("train_loss", step_metric="epoch")
            wandb.define_metric("val_loss", step_metric="epoch")
            wandb.define_metric("test_init", step_metric="epoch")

            # Test wandb logging
            try:
                wandb.log({"test_init": 1.0, "epoch": 0}, commit=True)
                logger.info("✅ Test wandb log successful")
            except Exception as e:
                logger.error(f"❌ Test wandb log failed: {e}")

        logger.info(
            f"Initialized trainer with "
            f"{sum(p.numel() for p in model.parameters())} parameters"
        )

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, mu, logvar = self.model(batch)

            # Compute loss (MSE between input and reconstructed)
            loss = self.criterion(reconstructed, batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / num_batches

        # Log to wandb if enabled
        if self.wandb_enabled:
            try:
                wandb.log(
                    {"train_loss": avg_loss, "epoch": self.current_epoch}, commit=True
                )
                logger.info(
                    f"✅ Logged train_loss {avg_loss:.6f} to wandb (epoch {self.current_epoch})"
                )
            except Exception as e:
                logger.error(f"❌ Failed to log train_loss to wandb: {e}")

        return avg_loss

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = batch.to(self.device)
                reconstructed, mu, logvar = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Log to wandb if enabled
        if self.wandb_enabled:
            try:
                wandb.log(
                    {"val_loss": avg_loss, "epoch": self.current_epoch}, commit=True
                )
                logger.info(
                    f"✅ Logged val_loss {avg_loss:.6f} to wandb (epoch {self.current_epoch})"
                )
            except Exception as e:
                logger.error(f"❌ Failed to log val_loss to wandb: {e}")

        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number of loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch

    def log_training_history(self) -> None:
        """
        Log training history to wandb.
        """
        if self.wandb_enabled:
            # Create a custom chart for training history
            data = []
            for i, (train_loss, val_loss) in enumerate(
                zip(self.train_losses, self.val_losses)
            ):
                data.append([i, train_loss, val_loss])

            table = wandb.Table(data=data, columns=["epoch", "train_loss", "val_loss"])
            wandb.log(
                {
                    "training_history": wandb.plot.line(
                        table,
                        "epoch",
                        ["train_loss", "val_loss"],
                        title="Training History",
                    )
                }
            )

    def log_reconstructions(
        self,
        num_samples: int = 8,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Log input and reconstructed images to wandb.

        Args:
            num_samples: Number of samples to visualize
            epoch: Current epoch number for logging
        """
        if not self.wandb_enabled:
            return
        self.model.eval()
        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(self.val_loader))
            batch = batch.to(self.device)[:num_samples]

            # Get reconstructions
            reconstructed, _, _ = self.model(batch)

            # Move to CPU and denormalize
            batch = batch.cpu()
            reconstructed = reconstructed.cpu()

            # Denormalize from [-1, 1] to [0, 1]
            batch = (batch + 1) / 2
            reconstructed = (reconstructed + 1) / 2
            batch = torch.clamp(batch, 0, 1)
            reconstructed = torch.clamp(reconstructed, 0, 1)

            # Log images to wandb
            images = []
            for i in range(num_samples):
                # Original image
                img_orig = batch[i].permute(1, 2, 0)
                images.append(wandb.Image(img_orig, caption=f"Original {i+1}"))

                # Reconstructed image
                img_recon = reconstructed[i].permute(1, 2, 0)
                images.append(wandb.Image(img_recon, caption=f"Reconstructed {i+1}"))

            # Log to wandb
            log_dict = {"reconstructions": images}
            if epoch is not None:
                log_dict["epoch"] = epoch

            wandb.log(log_dict)

    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        validate_every: int = 1,
        log_reconstructions_every: int = 5,
    ) -> None:
        """
        Train the auto-encoder.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
            log_reconstructions_every: Log reconstructions every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validation
            if (epoch + 1) % validate_every == 0:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                logger.info(
                    f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}, "
                    f"Val Loss = {val_loss:.6f}"
                )

                # Check if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch + 1, is_best=True)
            else:
                logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)

            # Log reconstructions to wandb
            if (epoch + 1) % log_reconstructions_every == 0:
                self.log_reconstructions(epoch=epoch + 1)

        # Log final training history
        self.log_training_history()

        # Finish wandb run
        if self.wandb_enabled:
            wandb.finish()

        logger.info("Training completed!")


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Configuration
    config = {
        "batch_size": 64,
        "image_size": 64,
        "latent_dim": 128,
        "hidden_dims": [32, 64, 128, 256],
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "num_epochs": 10,
        "save_dir": "autoencoder_checkpoints",
    }

    # Device - Support Mac Silicon MPS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_celeba_dataloader(
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        split="train",
        num_workers=4,
    )

    val_loader = create_celeba_dataloader(
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        split="valid",
        num_workers=4,
    )

    # Create model
    logger.info("Creating auto-encoder model...")
    model = create_autoencoder(
        input_channels=3,
        latent_dim=config["latent_dim"],
        hidden_dims=config["hidden_dims"],
        output_size=config["image_size"],
        device=device,
    )

    # Create trainer
    trainer = AutoEncoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        save_dir=config["save_dir"],
        wandb_enabled=True,
        project_name="autoencoder-celeba",
    )

    # Train
    trainer.train(
        num_epochs=config["num_epochs"],
        save_every=10,
        validate_every=1,
        log_reconstructions_every=5,
    )


if __name__ == "__main__":
    main()
