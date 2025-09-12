"""Auto-encoder model implementation for image reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConvEncoder(nn.Module):
    """Convolutional encoder for auto-encoder."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list = [32, 64, 128, 256],
    ) -> None:
        """
        Initialize convolutional encoder.

        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for each layer
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
                        in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Calculate the size after convolutions
        self.flatten_size = self._get_flatten_size(input_channels)

        # Latent layer
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _get_flatten_size(self, input_channels: int) -> int:
        """Calculate the flattened size after convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 64, 64)
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


class ConvDecoder(nn.Module):
    """Convolutional decoder for auto-encoder."""

    def __init__(
        self,
        latent_dim: int = 128,
        output_channels: int = 3,
        hidden_dims: list = [256, 128, 64, 32],
        output_size: int = 64,
    ) -> None:
        """
        Initialize convolutional decoder.

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

        return torch.tanh(x)  # Output in [-1, 1] range


class AutoEncoder(nn.Module):
    """Complete auto-encoder model."""

    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: list = [32, 64, 128, 256],
        output_size: int = 64,
    ) -> None:
        """
        Initialize auto-encoder.

        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden dimensions for encoder/decoder
            output_size: Target output image size
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Reverse hidden_dims for decoder
        decoder_hidden_dims = hidden_dims[::-1]

        self.encoder = ConvEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )

        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            output_channels=input_channels,
            hidden_dims=decoder_hidden_dims,
            output_size=output_size,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: Input tensor

        Returns:
            Latent representation
        """
        mu, logvar = self.encoder(x)
        return mu  # For auto-encoder, we just use the mean

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through auto-encoder.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstructed, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = mu  # For auto-encoder, we use deterministic encoding
        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input without returning latent parameters.

        Args:
            x: Input tensor

        Returns:
            Reconstructed tensor
        """
        reconstructed, _, _ = self.forward(x)
        return reconstructed


def create_autoencoder(
    input_channels: int = 3,
    latent_dim: int = 128,
    hidden_dims: list = [32, 64, 128, 256],
    output_size: int = 64,
    device: Optional[torch.device] = None,
) -> AutoEncoder:
    """
    Create and initialize auto-encoder model.

    Args:
        input_channels: Number of input channels
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden dimensions
        output_size: Target output image size
        device: Device to move model to

    Returns:
        Initialized auto-encoder model
    """
    model = AutoEncoder(
        input_channels=input_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        output_size=output_size,
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
        f"Created auto-encoder with {sum(p.numel() for p in model.parameters())} parameters"
    )
    return model
