"""Test script for auto-encoder implementation using pytest."""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import pytest

# Add the modeling directory to the Python path
modeling_path = Path(__file__).parent.parent.parent.parent.parent / "modeling"
sys.path.insert(0, str(modeling_path))

from generation.ae.autoencoder import create_autoencoder
from generation.celeba_dataloader import create_celeba_dataloader


class TestAutoencoder:
    """Test class for auto-encoder implementation."""

    def test_model_creation(self):
        """Test auto-encoder model creation."""
        # Create model
        model = create_autoencoder(
            input_channels=3,
            latent_dim=64,
            hidden_dims=[32, 64, 128],
            output_size=64,
        )

        # Test forward pass
        dummy_input = torch.randn(4, 3, 64, 64)
        reconstructed, mu, logvar = model(dummy_input)

        # Verify shapes
        assert (
            reconstructed.shape == dummy_input.shape
        ), f"Expected {dummy_input.shape}, got {reconstructed.shape}"
        assert mu.shape == (4, 64), f"Expected (4, 64), got {mu.shape}"
        assert logvar.shape == (4, 64), f"Expected (4, 64), got {logvar.shape}"

    def test_encoding_decoding(self):
        """Test encoding and decoding separately."""
        model = create_autoencoder(
            input_channels=3,
            latent_dim=128,
            hidden_dims=[32, 64, 128, 256],
            output_size=64,
        )

        dummy_input = torch.randn(2, 3, 64, 64)

        # Test encoding
        latent = model.encode(dummy_input)
        assert latent.shape == (2, 128), f"Expected (2, 128), got {latent.shape}"

        # Test decoding
        reconstructed = model.decode(latent)
        assert (
            reconstructed.shape == dummy_input.shape
        ), f"Expected {dummy_input.shape}, got {reconstructed.shape}"

        # Test reconstruction method
        reconstructed2 = model.reconstruct(dummy_input)
        assert (
            reconstructed2.shape == dummy_input.shape
        ), f"Expected {dummy_input.shape}, got {reconstructed2.shape}"

    def test_with_mock_data(self):
        """Test with mock data (simulating CelebA data)."""
        # Create model
        model = create_autoencoder(
            input_channels=3,
            latent_dim=64,
            hidden_dims=[32, 64, 128],
            output_size=64,
        )

        # Create mock batch (simulating CelebA data in [-1, 1] range)
        batch = torch.randn(4, 3, 64, 64) * 0.5  # Scale to roughly [-1, 1] range

        # Test forward pass
        reconstructed, mu, logvar = model(batch)

        # Verify output is in correct range
        assert (
            -1.0 <= reconstructed.min() <= 1.0
        ), f"Expected [-1, 1], got min {reconstructed.min()}"
        assert (
            -1.0 <= reconstructed.max() <= 1.0
        ), f"Expected [-1, 1], got max {reconstructed.max()}"

    def test_loss_computation(self):
        """Test loss computation."""
        model = create_autoencoder(
            input_channels=3,
            latent_dim=64,
            hidden_dims=[32, 64, 128],
            output_size=64,
        )

        # Create dummy data
        input_data = torch.randn(4, 3, 64, 64)

        # Forward pass
        reconstructed, mu, logvar = model(input_data)

        # Compute MSE loss
        mse_loss = torch.nn.MSELoss()(reconstructed, input_data)

        assert mse_loss.item() > 0, "Loss should be positive"

    @pytest.mark.skip(reason="Visualization test - run manually if needed")
    def test_visualize_reconstruction(self):
        """Visualize a reconstruction example."""
        # Create model
        model = create_autoencoder(
            input_channels=3,
            latent_dim=64,
            hidden_dims=[32, 64, 128],
            output_size=64,
        )

        # Create mock batch
        batch = torch.randn(4, 3, 64, 64) * 0.5

        # Get reconstruction
        with torch.no_grad():
            reconstructed = model.reconstruct(batch)

        # Denormalize for visualization
        batch_vis = (batch + 1) / 2
        reconstructed_vis = (reconstructed + 1) / 2
        batch_vis = torch.clamp(batch_vis, 0, 1)
        reconstructed_vis = torch.clamp(reconstructed_vis, 0, 1)

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))

        for i in range(4):
            # Original images (top row)
            img_orig = batch_vis[i].permute(1, 2, 0)
            axes[0, i].imshow(img_orig)
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            # Reconstructed images (bottom row)
            img_recon = reconstructed_vis[i].permute(1, 2, 0)
            axes[1, i].imshow(img_recon)
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig("autoencoder_test_reconstruction.png", dpi=150, bbox_inches="tight")
        plt.close()


# Standalone test functions for backward compatibility
def test_model_creation_standalone():
    """Standalone test for model creation."""
    test_instance = TestAutoencoder()
    test_instance.test_model_creation()


def test_encoding_decoding_standalone():
    """Standalone test for encoding/decoding."""
    test_instance = TestAutoencoder()
    test_instance.test_encoding_decoding()


def test_with_mock_data_standalone():
    """Standalone test with mock data."""
    test_instance = TestAutoencoder()
    test_instance.test_with_mock_data()


def test_loss_computation_standalone():
    """Standalone test for loss computation."""
    test_instance = TestAutoencoder()
    test_instance.test_loss_computation()


if __name__ == "__main__":
    # Run tests manually for backward compatibility
    print("Testing Auto-encoder Implementation...")

    try:
        test_instance = TestAutoencoder()
        test_instance.test_model_creation()
        print("✓ Model creation test passed!")

        test_instance.test_encoding_decoding()
        print("✓ Encoding/decoding test passed!")

        test_instance.test_with_mock_data()
        print("✓ Mock data test passed!")

        test_instance.test_loss_computation()
        print("✓ Loss computation test passed!")

        print(
            "\n🎉 All tests passed! Auto-encoder implementation is working correctly."
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise
