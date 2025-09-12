"""Training script for auto-encoder model.

Can be run from repo root:
    python -m modeling.generation.ae.train_autoencoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import wandb
import numpy as np
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
        use_cosine_scheduler: bool = True,
        scheduler_t_max: int = 100,
        scheduler_eta_min: float = 0.0,
        step_level_scheduling: bool = True,
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
            use_cosine_scheduler: Whether to use cosine annealing scheduler
            scheduler_t_max: Max steps/epochs for cosine scheduler
            scheduler_eta_min: Minimum learning rate for cosine scheduler
            step_level_scheduling: Step scheduler per batch (True) or per epoch
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.wandb_enabled = wandb_enabled
        self.project_name = project_name
        self.use_cosine_scheduler = use_cosine_scheduler
        self.scheduler_t_max = scheduler_t_max
        self.scheduler_eta_min = scheduler_eta_min
        self.step_level_scheduling = step_level_scheduling

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.MSELoss()

        # Setup learning rate scheduler
        if self.use_cosine_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_t_max,
                eta_min=self.scheduler_eta_min,
            )
        else:
            self.scheduler = None

        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.current_epoch = 0

        # Initialize wandb if enabled
        if self.wandb_enabled:
            # Create meaningful run name with key hyperparameters
            run_name = (
                f"ae_latent{model.latent_dim}_lr{learning_rate}_"
                f"bs{train_loader.batch_size}"
            )
            logger.info(
                f"Initializing wandb with project: {self.project_name}, "
                f"run: {run_name}"
            )
            # Prepare config for wandb
            wandb_config = {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": train_loader.batch_size,
                "device": str(device),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "latent_dim": model.latent_dim,
            }

            # Add scheduler config
            wandb_config["use_cosine_scheduler"] = self.use_cosine_scheduler
            wandb_config["scheduler_t_max"] = self.scheduler_t_max
            wandb_config["scheduler_eta_min"] = self.scheduler_eta_min
            wandb_config["step_level_scheduling"] = self.step_level_scheduling

            wandb.init(
                project=self.project_name,
                name=run_name,
                mode="online",  # Explicitly set to online mode
                config=wandb_config,
            )
            logger.info("Wandb initialized successfully")

            # # Define metrics for wandb
            # wandb.define_metric("epoch")
            # wandb.define_metric("train_loss", step_metric="epoch")
            # wandb.define_metric("val_loss", step_metric="epoch")
            # wandb.define_metric("learning_rate", step_metric="epoch")

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

            # Step scheduler after each batch if step-level scheduling enabled
            if self.scheduler is not None and self.step_level_scheduling:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / num_batches

        # Log to wandb if enabled
        if self.wandb_enabled:
            try:
                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "learning_rate": current_lr,
                        "epoch": self.current_epoch,
                    },
                )
                logger.info(
                    f"✅ Logged train_loss {avg_loss:.6f}, "
                    f"lr {current_lr:.2e} to wandb "
                    f"(epoch {self.current_epoch})"
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
                wandb.log({"val_loss": avg_loss, "epoch": self.current_epoch})
                logger.info(
                    f"✅ Logged val_loss {avg_loss:.6f} to wandb "
                    f"(epoch {self.current_epoch})"
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
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
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

        # Load scheduler state if available
        if (
            self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

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
                # Convert to numpy and scale to uint8
                img_orig_np = (img_orig.numpy() * 255).astype(np.uint8)
                images.append(wandb.Image(img_orig_np, caption=f"Original {i+1}"))

                # Reconstructed image
                img_recon = reconstructed[i].permute(1, 2, 0)
                # Convert to numpy and scale to uint8
                img_recon_np = (img_recon.numpy() * 255).astype(np.uint8)
                images.append(wandb.Image(img_recon_np, caption=f"Reconstructed {i+1}"))

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

                # Step scheduler at epoch level if not using step-level
                if self.scheduler is not None and not self.step_level_scheduling:
                    self.scheduler.step()
            else:
                logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}")
                # Step scheduler even without validation
                if self.scheduler is not None and not self.step_level_scheduling:
                    self.scheduler.step()

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
        "num_epochs": 5,
        "save_dir": "autoencoder_checkpoints",
        # Learning rate scheduler config
        "use_cosine_scheduler": True,
        # Updated to total steps if step_level_scheduling=True
        "scheduler_t_max": 5,
        "scheduler_eta_min": 1e-5,  # Non-zero minimum
        "step_level_scheduling": True,  # Step per batch for finer granularity
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

    # Calculate total steps for step-level scheduling
    if config["step_level_scheduling"]:
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config["num_epochs"]
        config["scheduler_t_max"] = total_steps
        logger.info(
            f"Step-level scheduling: {steps_per_epoch} steps/epoch, "
            f"{total_steps} total steps"
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
        use_cosine_scheduler=config["use_cosine_scheduler"],
        scheduler_t_max=config["scheduler_t_max"],
        scheduler_eta_min=config["scheduler_eta_min"],
        step_level_scheduling=config["step_level_scheduling"],
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
