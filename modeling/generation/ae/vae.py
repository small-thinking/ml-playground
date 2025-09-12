"""Variational Auto-Encoder (VAE) model implementation for image generation and reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VAEEncoder(nn.Module):
    """Convolutional encoder for Variational Auto-Encoder."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list = [32, 64, 128, 256],
        input_size: int = 64,
    ) -> None:
        """
        Initialize VAE encoder.

        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for each layer
            input_size: Input image size (assumes square images)
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        in_channels = input_channels

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Calculate the size after convolutions
        self.flatten_size = self._get_flatten_size(input_channels, input_size)

        # Latent layers for mean and log variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_flatten_size(self, input_channels: int, input_size: int = 64) -> int:
        """Calculate the flattened size after convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_size, input_size)
            dummy_output = self.encoder(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class VAEDecoder(nn.Module):
    """Convolutional decoder for Variational Auto-Encoder."""

    def __init__(
        self,
        latent_dim: int = 128,
        output_channels: int = 3,
        hidden_dims: list = [256, 128, 64, 32],
        output_size: int = 64,
    ) -> None:
        """
        Initialize VAE decoder.

        Args:
            latent_dim: Dimension of latent space
            output_channels: Number of output channels (3 for RGB)
            hidden_dims: List of hidden dimensions for each layer
            output_size: Target output image size
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_dims = hidden_dims

        # Calculate the size needed for the first linear layer
        self.flatten_size = self._get_flatten_size(
            latent_dim, hidden_dims[0], hidden_dims
        )

        # Initial linear layer
        self.fc = nn.Linear(latent_dim, self.flatten_size)

        # Build decoder layers (reverse order of encoder)
        layers = []
        # Start with the last hidden dimension and work backwards
        reversed_hidden_dims = list(reversed(hidden_dims))
        in_channels = reversed_hidden_dims[0]

        for hidden_dim in reversed_hidden_dims[1:]:
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = hidden_dim

        # Final layer
        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )

        self.decoder = nn.Sequential(*layers)

    def _get_flatten_size(
        self, latent_dim: int, first_hidden_dim: int, hidden_dims: list
    ) -> int:
        """Calculate the flattened size for the first linear layer."""
        # Calculate the size after all transpose convolutions
        # The decoder should start with the same spatial size that the encoder ends with
        size = self.output_size
        for _ in range(len(hidden_dims)):  # Number of conv layers in encoder
            size = (size - 1) // 2 + 1  # Same as encoder stride=2

        return hidden_dims[-1] * size * size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, channels, height, width)
        """
        x = self.fc(z)
        # Reshape to start decoder with the last hidden dimension
        batch_size = x.size(0)
        # Calculate the spatial size after encoder convolutions
        spatial_size = int((self.flatten_size // self.hidden_dims[-1]) ** 0.5)
        x = x.view(batch_size, self.hidden_dims[-1], spatial_size, spatial_size)
        x = self.decoder(x)

        # Ensure output is the correct size
        if x.size(-1) != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )

        return torch.sigmoid(x)  # Output in [0, 1] range for VAE


class VariationalAutoEncoder(nn.Module):
    """Complete Variational Auto-Encoder model."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list = [32, 64, 128, 256],
        output_size: int = 64,
        beta: float = 1.0,
    ) -> None:
        """
        Initialize Variational Auto-Encoder.

        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for encoder/decoder
            output_size: Target output image size
            beta: Beta parameter for beta-VAE (controls KL divergence weight)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Reverse hidden_dims for decoder
        decoder_hidden_dims = hidden_dims[::-1]

        self.encoder = VAEEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            input_size=output_size,
        )

        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            hidden_dims=decoder_hidden_dims,
            output_size=output_size,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed output
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstructed, mu, logvar, z)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar, z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input using mean of latent distribution (deterministic).

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor
        """
        mu, logvar = self.encode(x)
        reconstructed = self.decode(mu)
        return reconstructed

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate new samples from prior distribution.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate on

        Returns:
            Generated samples
        """
        with torch.no_grad():
            # Sample from prior N(0, I)
            z = torch.randn(num_samples, self.latent_dim, device=device)
            generated = self.decode(z)
        return generated

    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.

        Args:
            x1: First input image
            x2: Second input image
            num_steps: Number of interpolation steps

        Returns:
            Interpolated images
        """
        with torch.no_grad():
            # Encode both images
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)

            # Interpolate in latent space
            interpolated_images = []
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                img_interp = self.decode(z_interp)
                interpolated_images.append(img_interp)

        return torch.stack(interpolated_images, dim=0)


def vae_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss (ELBO).

    Args:
        reconstructed: Reconstructed images
        target: Target images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Beta parameter for beta-VAE

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss)
    """
    # Reconstruction loss (MSE for consistency with AE, handles range mismatch)
    # Convert target from [-1, 1] to [0, 1] for comparison with sigmoid output
    target_normalized = (target + 1) / 2
    recon_loss = F.mse_loss(reconstructed, target_normalized, reduction="sum")

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def create_vae(
    input_channels: int = 3,
    latent_dim: int = 128,
    hidden_dims: list = [32, 64, 128, 256],
    output_size: int = 64,
    beta: float = 1.0,
    device: Optional[torch.device] = None,
) -> VariationalAutoEncoder:
    """
    Create and initialize VAE model.

    Args:
        input_channels: Number of input channels
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden dimensions
        output_size: Target output image size
        beta: Beta parameter for beta-VAE
        device: Device to move model to

    Returns:
        Initialized VAE model
    """
    model = VariationalAutoEncoder(
        input_channels=input_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        output_size=output_size,
        beta=beta,
    )

    if device is not None:
        model = model.to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    logger.info(
        f"Created VAE with {sum(p.numel() for p in model.parameters())} parameters, "
        f"beta={beta}"
    )
    return model


if __name__ == "__main__":
    """Example usage of the VAE model."""
    import argparse

    parser = argparse.ArgumentParser(description="VAE Model Example")
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="Number of input channels (default: 3 for RGB)",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=64,
        help="Output image size (default: 64x64)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Hidden dimensions for encoder/decoder",
    )
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for KL divergence",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu/cuda)"
    )

    args = parser.parse_args()

    # Create VAE model
    model = create_vae(
        input_channels=args.input_channels,
        output_size=args.output_size,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        beta=args.beta,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("VAE model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Test forward pass with random input (normalized to [0,1] for BCE loss)
    batch_size = 4
    x = torch.rand(
        batch_size, args.input_channels, args.output_size, args.output_size
    ).to(device)

    with torch.no_grad():
        recon_x, mu, logvar, z = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Reconstructed shape: {recon_x.shape}")
        print(f"Mean shape: {mu.shape}")
        print(f"Log variance shape: {logvar.shape}")
        print(f"Latent shape: {z.shape}")

        # Calculate loss
        total_loss, recon_loss, kl_loss = vae_loss(
            recon_x, x, mu, logvar, beta=args.beta
        )
        print(f"Total VAE Loss: {total_loss.item():.4f}")
        print(f"Reconstruction Loss: {recon_loss.item():.4f}")
        print(f"KL Loss: {kl_loss.item():.4f}")
