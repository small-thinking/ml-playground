"""Training script for Variational Auto-Encoder (VAE) model.

Can be run from repo root:
    python -m modeling.generation.ae.train_vae
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
from modeling.generation.ae.vae import create_vae, vae_loss
from modeling.generation.image_dataloader import create_image_dataloader

logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer class for Variational Auto-Encoder model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = "vae_checkpoints",
        wandb_enabled: bool = True,
        project_name: str = "vae-training",
        use_cosine_scheduler: bool = True,
        scheduler_t_max: int = 100,
        scheduler_eta_min: float = 0.0,
        step_level_scheduling: bool = True,
        beta: float = 1.0,
    ) -> None:
        """
        Initialize VAE trainer.

        Args:
            model: VAE model
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
            beta: Beta parameter for beta-VAE
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
        self.beta = beta

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

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
        self.train_recon_losses: List[float] = []
        self.val_recon_losses: List[float] = []
        self.train_kl_losses: List[float] = []
        self.val_kl_losses: List[float] = []
        self.current_epoch = 0

        # Initialize wandb if enabled
        if self.wandb_enabled:
            # Create meaningful run name with key hyperparameters
            run_name = (
                f"vae_512x512_latent{model.latent_dim}_beta{beta}_lr{learning_rate}_"
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
                "beta": beta,
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

        logger.info(
            f"Initialized VAE trainer with "
            f"{sum(p.numel() for p in model.parameters())} parameters, "
            f"beta={beta}"
        )

    def train_epoch(self) -> tuple[float, float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_total_loss, average_recon_loss, average_kl_loss)
        """
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, mu, logvar, z = self.model(batch)

            # Debug: Check for NaN values in model outputs
            if torch.isnan(reconstructed).any():
                logger.error(
                    f"❌ NaN detected in reconstructed output at batch {batch_idx}"
                )
                logger.error(
                    f"Reconstructed stats: min={reconstructed.min():.4f}, max={reconstructed.max():.4f}"
                )
            if torch.isnan(mu).any():
                logger.error(f"❌ NaN detected in mu at batch {batch_idx}")
                logger.error(f"Mu stats: min={mu.min():.4f}, max={mu.max():.4f}")
            if torch.isnan(logvar).any():
                logger.error(f"❌ NaN detected in logvar at batch {batch_idx}")
                logger.error(
                    f"Logvar stats: min={logvar.min():.4f}, max={logvar.max():.4f}"
                )

            # Compute VAE loss
            total_loss_batch, recon_loss_batch, kl_loss_batch = vae_loss(
                reconstructed, batch, mu, logvar, self.beta
            )

            # Debug: Check for NaN values in loss components
            if torch.isnan(total_loss_batch):
                logger.error(f"❌ NaN detected in total_loss at batch {batch_idx}")
                logger.error(
                    f"Recon loss: {recon_loss_batch.item():.4f}, KL loss: {kl_loss_batch.item():.4f}"
                )
                logger.error(
                    f"Batch stats: min={batch.min():.4f}, max={batch.max():.4f}"
                )
                logger.error(
                    f"Reconstructed stats: min={reconstructed.min():.4f}, max={reconstructed.max():.4f}"
                )
                # Skip this batch to prevent training from crashing
                continue

            # Normalize by batch size
            batch_size = batch.size(0)
            total_loss_batch = total_loss_batch / batch_size
            recon_loss_batch = recon_loss_batch / batch_size
            kl_loss_batch = kl_loss_batch / batch_size

            # Backward pass
            total_loss_batch.backward()
            # Gradient clipping - configurable for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Step scheduler after each batch if step-level scheduling enabled
            if self.scheduler is not None and self.step_level_scheduling:
                self.scheduler.step()

            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss_batch.item()
            total_kl_loss += kl_loss_batch.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "total": f"{total_loss_batch.item():.4f}",
                    "recon": f"{recon_loss_batch.item():.4f}",
                    "kl": f"{kl_loss_batch.item():.4f}",
                }
            )

        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        # Log to wandb if enabled
        if self.wandb_enabled:
            try:
                current_lr = self.optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "train_total_loss": avg_total_loss,
                        "train_recon_loss": avg_recon_loss,
                        "train_kl_loss": avg_kl_loss,
                        "learning_rate": current_lr,
                        "epoch": self.current_epoch,
                    },
                )
                logger.info(
                    f"✅ Logged train losses to wandb "
                    f"(epoch {self.current_epoch}): "
                    f"total={avg_total_loss:.4f}, "
                    f"recon={avg_recon_loss:.4f}, "
                    f"kl={avg_kl_loss:.4f}, "
                    f"lr={current_lr:.2e}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to log train losses to wandb: {e}")

        return avg_total_loss, avg_recon_loss, avg_kl_loss

    def validate(self) -> tuple[float, float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average_total_loss, average_recon_loss, average_kl_loss)
        """
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = batch.to(self.device)
                reconstructed, mu, logvar, z = self.model(batch)

                # Compute VAE loss
                total_loss_batch, recon_loss_batch, kl_loss_batch = vae_loss(
                    reconstructed, batch, mu, logvar, self.beta
                )

                # Normalize by batch size
                batch_size = batch.size(0)
                total_loss_batch = total_loss_batch / batch_size
                recon_loss_batch = recon_loss_batch / batch_size
                kl_loss_batch = kl_loss_batch / batch_size

                total_loss += total_loss_batch.item()
                total_recon_loss += recon_loss_batch.item()
                total_kl_loss += kl_loss_batch.item()
                num_batches += 1

        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        # Log to wandb if enabled
        if self.wandb_enabled:
            try:
                wandb.log(
                    {
                        "val_total_loss": avg_total_loss,
                        "val_recon_loss": avg_recon_loss,
                        "val_kl_loss": avg_kl_loss,
                        "epoch": self.current_epoch,
                    }
                )
                logger.info(
                    f"✅ Logged val losses to wandb "
                    f"(epoch {self.current_epoch}): "
                    f"total={avg_total_loss:.4f}, "
                    f"recon={avg_recon_loss:.4f}, "
                    f"kl={avg_kl_loss:.4f}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to log val losses to wandb: {e}")

        return avg_total_loss, avg_recon_loss, avg_kl_loss

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
            "train_recon_losses": self.train_recon_losses,
            "val_recon_losses": self.val_recon_losses,
            "train_kl_losses": self.train_kl_losses,
            "val_kl_losses": self.val_kl_losses,
            "beta": self.beta,
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
        self.train_recon_losses = checkpoint.get("train_recon_losses", [])
        self.val_recon_losses = checkpoint.get("val_recon_losses", [])
        self.train_kl_losses = checkpoint.get("train_kl_losses", [])
        self.val_kl_losses = checkpoint.get("val_kl_losses", [])

        epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch

    def log_training_history(self) -> None:
        """
        Log training history to wandb.
        """
        if self.wandb_enabled:
            # Create custom charts for training history
            data = []
            for i, (
                total_loss,
                val_total_loss,
                recon_loss,
                val_recon_loss,
                kl_loss,
                val_kl_loss,
            ) in enumerate(
                zip(
                    self.train_losses,
                    self.val_losses,
                    self.train_recon_losses,
                    self.val_recon_losses,
                    self.train_kl_losses,
                    self.val_kl_losses,
                )
            ):
                data.append(
                    [
                        i,
                        total_loss,
                        val_total_loss,
                        recon_loss,
                        val_recon_loss,
                        kl_loss,
                        val_kl_loss,
                    ]
                )

            table = wandb.Table(
                data=data,
                columns=[
                    "epoch",
                    "train_total_loss",
                    "val_total_loss",
                    "train_recon_loss",
                    "val_recon_loss",
                    "train_kl_loss",
                    "val_kl_loss",
                ],
            )

            # Log total loss chart
            wandb.log(
                {
                    "training_history_total": wandb.plot.line(
                        table,
                        "epoch",
                        ["train_total_loss", "val_total_loss"],
                        title="Total Loss History",
                    )
                }
            )

            # Log reconstruction loss chart
            wandb.log(
                {
                    "training_history_recon": wandb.plot.line(
                        table,
                        "epoch",
                        ["train_recon_loss", "val_recon_loss"],
                        title="Reconstruction Loss History",
                    )
                }
            )

            # Log KL loss chart
            wandb.log(
                {
                    "training_history_kl": wandb.plot.line(
                        table,
                        "epoch",
                        ["train_kl_loss", "val_kl_loss"],
                        title="KL Loss History",
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
            reconstructed, mu, logvar, z = self.model(batch)

            # Move to CPU and handle different ranges for visualization
            batch = batch.cpu()
            reconstructed = reconstructed.cpu()

            # Denormalize input from [-1, 1] to [0, 1] for visualization
            batch = (batch + 1) / 2
            batch = torch.clamp(batch, 0, 1)
            # VAE output is already in [0, 1] range from sigmoid
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
                images.append(
                    wandb.Image(img_recon_np, caption=f"VAE Reconstructed {i+1}")
                )

            # Log to wandb
            log_dict = {"reconstructions": images}
            if epoch is not None:
                log_dict["epoch"] = epoch

            wandb.log(log_dict)

    def log_generated_samples(
        self,
        num_samples: int = 16,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Log generated samples to wandb.

        Args:
            num_samples: Number of samples to generate
            epoch: Current epoch number for logging
        """
        if not self.wandb_enabled:
            return
        self.model.eval()
        with torch.no_grad():
            # Generate samples
            generated = self.model.generate(num_samples, self.device)
            generated = generated.cpu()

            # Ensure output is in [0, 1] range
            generated = torch.clamp(generated, 0, 1)

            # Log images to wandb
            images = []
            for i in range(num_samples):
                img_gen = generated[i].permute(1, 2, 0)
                # Convert to numpy and scale to uint8
                img_gen_np = (img_gen.numpy() * 255).astype(np.uint8)
                images.append(wandb.Image(img_gen_np, caption=f"Generated {i+1}"))

            # Log to wandb
            log_dict = {"generated_samples": images}
            if epoch is not None:
                log_dict["epoch"] = epoch

            wandb.log(log_dict)

    def train(
        self,
        num_epochs: int,
        validate_every: int = 1,
        log_reconstructions_every: int = 5,
        log_generated_every: int = 10,
    ) -> None:
        """
        Train the VAE.

        Args:
            num_epochs: Number of epochs to train
            validate_every: Validate every N epochs
            log_reconstructions_every: Log reconstructions every N epochs
            log_generated_every: Log generated samples every N epochs
        """
        logger.info(f"Starting VAE training for {num_epochs} epochs")
        best_val_loss = float("inf")
        patience_counter = 0
        patience = getattr(self, "patience", 3)
        min_delta = getattr(self, "min_delta", 0.001)

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training
            (
                train_total_loss,
                train_recon_loss,
                train_kl_loss,
            ) = self.train_epoch()
            self.train_losses.append(train_total_loss)
            self.train_recon_losses.append(train_recon_loss)
            self.train_kl_losses.append(train_kl_loss)

            # Validation
            if (epoch + 1) % validate_every == 0:
                val_total_loss, val_recon_loss, val_kl_loss = self.validate()
                self.val_losses.append(val_total_loss)
                self.val_recon_losses.append(val_recon_loss)
                self.val_kl_losses.append(val_kl_loss)
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"Train Total = {train_total_loss:.4f}, "
                    f"Train Recon = {train_recon_loss:.4f}, "
                    f"Train KL = {train_kl_loss:.4f} | "
                    f"Val Total = {val_total_loss:.4f}, "
                    f"Val Recon = {val_recon_loss:.4f}, "
                    f"Val KL = {val_kl_loss:.4f}"
                )

                # Check if best model and early stopping
                if val_total_loss < best_val_loss - min_delta:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                    self.save_checkpoint(epoch + 1, is_best=True)
                    logger.info(f"✅ New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(
                        f"⏳ No improvement for {patience_counter} epochs (patience: {patience})"
                    )

                    # Early stopping
                    if patience_counter >= patience:
                        logger.info(
                            f"🛑 Early stopping triggered after {epoch + 1} epochs"
                        )
                        break

                # Step scheduler at epoch level if not using step-level
                if self.scheduler is not None and not self.step_level_scheduling:
                    self.scheduler.step()
            else:
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"Train Total = {train_total_loss:.4f}, "
                    f"Train Recon = {train_recon_loss:.4f}, "
                    f"Train KL = {train_kl_loss:.4f}"
                )
                # Step scheduler even without validation
                if self.scheduler is not None and not self.step_level_scheduling:
                    self.scheduler.step()

            # Note: Checkpoint saving removed - model will only be saved after training completion

            # Log reconstructions to wandb
            if (epoch + 1) % log_reconstructions_every == 0:
                self.log_reconstructions(epoch=epoch + 1)

            # Log generated samples to wandb
            if (epoch + 1) % log_generated_every == 0:
                self.log_generated_samples(epoch=epoch + 1)

        # Save final model after training completion
        self.save_checkpoint(self.current_epoch, is_best=False)
        logger.info(f"Saved final model after training completion (epoch {self.current_epoch})")

        # Log final training history
        self.log_training_history()

        # Finish wandb run
        if self.wandb_enabled:
            wandb.finish()

        logger.info("VAE training completed!")


def main():
    """Main training function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Configuration - Anti-overfitting setup for 512x512 resolution
    config = {
        "batch_size": 32,
        "image_size": 512,
        "latent_dim": 512,
        "hidden_dims": [64, 128, 256, 512],  # Keep capacity but add regularization
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,  # Increased weight decay for regularization
        "num_epochs": 5,
        "save_dir": "vae_checkpoints",
        "beta": 1.0,  # Increased beta for stronger KL regularization
        # Learning rate scheduler config
        "use_cosine_scheduler": True,
        "scheduler_t_max": 20,
        "scheduler_eta_min": 1e-6,  # Lower minimum
        "step_level_scheduling": False,  # Use epoch-level scheduling for stability
        # Regularization config
        "dropout_rate": 0.1,  # Add dropout for regularization
        "gradient_clip_norm": 1.0,  # More conservative gradient clipping
        # Early stopping config
        "patience": 3,  # Stop if no improvement for 3 epochs
        "min_delta": 0.001,  # Minimum change to qualify as improvement
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
    train_loader = create_image_dataloader(
        dataset_type="afhq",
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        split="train",
        num_workers=4,
    )

    # For AFHQv2, we create a validation split from the training data
    val_loader = create_image_dataloader(
        dataset_type="afhq",
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        split="validation",  # This will create a validation split from training data
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
    logger.info("Creating VAE model...")
    model = create_vae(
        input_channels=3,
        latent_dim=config["latent_dim"],
        hidden_dims=config["hidden_dims"],
        output_size=config["image_size"],
        beta=config["beta"],
        device=device,
        dropout_rate=config["dropout_rate"],
    )

    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        save_dir=config["save_dir"],
        wandb_enabled=True,
        project_name="vae-afhq",
        use_cosine_scheduler=config["use_cosine_scheduler"],
        scheduler_t_max=config["scheduler_t_max"],
        scheduler_eta_min=config["scheduler_eta_min"],
        step_level_scheduling=config["step_level_scheduling"],
        beta=config["beta"],
    )

    # Add early stopping parameters to trainer
    trainer.patience = config["patience"]
    trainer.min_delta = config["min_delta"]

    # Train
    trainer.train(
        num_epochs=config["num_epochs"],
        validate_every=1,
        log_reconstructions_every=5,
        log_generated_every=10,
    )


if __name__ == "__main__":
    main()
