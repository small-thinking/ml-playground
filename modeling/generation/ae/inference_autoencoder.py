"""Autoencoder inference script for image compression and recovery testing.

Can be run from repo root:
    python -m modeling.generation.ae.inference_autoencoder \
        --checkpoint_path autoencoder_checkpoints/best_model.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import logging
import os
import sys
import argparse
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity

# Get the current working directory (should be repo root when run as module)
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import the modules
from modeling.generation.ae.autoencoder import create_autoencoder
from modeling.generation.celeba_dataloader import create_celeba_dataloader

logger = logging.getLogger(__name__)


class AutoEncoderInference:
    """Autoencoder inference class for testing compression and recovery."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_size: int = 64,
    ) -> None:
        """
        Initialize autoencoder inference.

        Args:
            model: Trained autoencoder model
            device: Device to run inference on
            image_size: Expected image size
        """
        self.model = model
        self.device = device
        self.image_size = image_size
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}"
        )
        return checkpoint

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: Input image as numpy array (H, W, C) in [0, 255] range

        Returns:
            Preprocessed tensor (1, C, H, W) in [-1, 1] range
        """
        # Convert to PIL Image for consistent processing
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Resize and convert to tensor
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0

        # Normalize to [-1, 1]
        image = (image - 0.5) / 0.5

        # Convert to tensor and add batch dimension
        if len(image.shape) == 2:  # Grayscale
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        else:  # RGB
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)

        return image.unsqueeze(0).to(self.device)  # (1, C, H, W)

    def postprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess tensor to image.

        Args:
            tensor: Output tensor (1, C, H, W) in [-1, 1] range

        Returns:
            Image as numpy array (H, W, C) in [0, 255] range
        """
        # Move to CPU and remove batch dimension
        image = tensor.squeeze(0).cpu()

        # Denormalize from [-1, 1] to [0, 1]
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)

        # Convert to numpy
        if image.shape[0] == 1:  # Grayscale
            image = image.squeeze(0).numpy()  # (H, W)
        else:  # RGB
            image = image.permute(1, 2, 0).numpy()  # (H, W, C)

        return (image * 255).astype(np.uint8)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent representation.

        Args:
            image: Input image tensor

        Returns:
            Latent representation
        """
        with torch.no_grad():
            latent = self.model.encode(image)
        return latent

    def decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.

        Args:
            latent: Latent representation

        Returns:
            Reconstructed image tensor
        """
        with torch.no_grad():
            reconstructed = self.model.decode(latent)
        return reconstructed

    def reconstruct_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image through encode-decode process.

        Args:
            image: Input image tensor

        Returns:
            Reconstructed image tensor
        """
        with torch.no_grad():
            reconstructed = self.model.reconstruct(image)
        return reconstructed

    def calculate_metrics(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate reconstruction quality metrics.

        Args:
            original: Original image (H, W, C) in [0, 255] range
            reconstructed: Reconstructed image (H, W, C) in [0, 255] range

        Returns:
            Dictionary of metrics
        """
        # Ensure both images are in the same format
        if len(original.shape) == 2:  # Grayscale
            original = np.stack([original] * 3, axis=-1)
        if len(reconstructed.shape) == 2:  # Grayscale
            reconstructed = np.stack([reconstructed] * 3, axis=-1)

        # Convert to float for calculations
        orig_float = original.astype(np.float64)
        recon_float = reconstructed.astype(np.float64)

        # Calculate PSNR
        mse = np.mean((orig_float - recon_float) ** 2)
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        # Calculate SSIM
        ssim = structural_similarity(
            orig_float, recon_float, multichannel=True, channel_axis=2
        )

        # Calculate compression ratio
        original_size = original.nbytes
        # For autoencoder, we approximate the latent size
        latent_size = self.model.latent_dim * 4  # 4 bytes per float32
        compression_ratio = original_size / latent_size

        return {
            "psnr": psnr,
            "ssim": ssim,
            "mse": mse,
            "compression_ratio": compression_ratio,
            "original_size_bytes": original_size,
            "latent_size_bytes": latent_size,
        }

    def process_single_image(
        self,
        image_path: str,
        save_results: bool = True,
        output_dir: str = "inference_results",
    ) -> Dict[str, Any]:
        """
        Process a single image through the autoencoder.

        Args:
            image_path: Path to input image
            save_results: Whether to save results
            output_dir: Directory to save results

        Returns:
            Dictionary with results and metrics
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        image_tensor = self.preprocess_image(image_array)

        # Reconstruct image
        reconstructed_tensor = self.reconstruct_image(image_tensor)
        reconstructed_array = self.postprocess_image(reconstructed_tensor)

        # Calculate metrics
        metrics = self.calculate_metrics(image_array, reconstructed_array)

        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(image_path).stem

            # Save original and reconstructed images
            orig_path = os.path.join(output_dir, f"{base_name}_original.png")
            recon_path = os.path.join(output_dir, f"{base_name}_reconstructed.png")
            Image.fromarray(image_array).save(orig_path)
            Image.fromarray(reconstructed_array).save(recon_path)

            # Create comparison image
            comp_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            self.create_comparison_image(
                image_array, reconstructed_array, comp_path, metrics
            )

        return {
            "original": image_array,
            "reconstructed": reconstructed_array,
            "metrics": metrics,
        }

    def create_comparison_image(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        save_path: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Create a side-by-side comparison image.

        Args:
            original: Original image
            reconstructed: Reconstructed image
            save_path: Path to save comparison image
            metrics: Quality metrics
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        axes[0].imshow(original)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")

        # Reconstructed image
        axes[1].imshow(reconstructed)
        axes[1].set_title("Reconstructed Image", fontsize=14)
        axes[1].axis("off")

        # Add metrics as text
        metrics_text = f"PSNR: {metrics['psnr']:.2f} dB\n"
        metrics_text += f"SSIM: {metrics['ssim']:.4f}\n"
        metrics_text += f"MSE: {metrics['mse']:.2f}\n"
        metrics_text += f"Compression Ratio: {metrics['compression_ratio']:.1f}x"

        fig.text(
            0.5,
            0.02,
            metrics_text,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def process_batch(
        self,
        dataloader: DataLoader,
        num_samples: int = 8,
        save_results: bool = True,
        output_dir: str = "inference_results",
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images from dataloader.

        Args:
            dataloader: DataLoader with images
            num_samples: Number of samples to process
            save_results: Whether to save results
            output_dir: Directory to save results

        Returns:
            List of results for each sample
        """
        results = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx * dataloader.batch_size >= num_samples:
                    break

                batch = batch.to(self.device)
                batch_size = min(
                    batch.size(0), num_samples - batch_idx * dataloader.batch_size
                )

                # Reconstruct batch
                reconstructed = self.reconstruct_image(batch[:batch_size])

                # Process each image in the batch
                for i in range(batch_size):
                    original_tensor = batch[i : i + 1]
                    reconstructed_tensor = reconstructed[i : i + 1]

                    # Convert to numpy arrays
                    original_array = self.postprocess_image(original_tensor)
                    reconstructed_array = self.postprocess_image(reconstructed_tensor)

                    # Calculate metrics
                    metrics = self.calculate_metrics(
                        original_array, reconstructed_array
                    )

                    results.append(
                        {
                            "original": original_array,
                            "reconstructed": reconstructed_array,
                            "metrics": metrics,
                        }
                    )

        # Save batch results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            self.create_batch_comparison(results, output_dir)

        return results

    def create_batch_comparison(
        self, results: List[Dict[str, Any]], output_dir: str, max_images: int = 8
    ) -> None:
        """
        Create a grid comparison of multiple images.

        Args:
            results: List of results from batch processing
            output_dir: Directory to save results
            max_images: Maximum number of images to display
        """
        num_images = min(len(results), max_images)
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))

        if num_images == 1:
            axes = axes.reshape(2, 1)

        for i in range(num_images):
            result = results[i]

            # Original images (top row)
            axes[0, i].imshow(result["original"])
            axes[0, i].set_title(f"Original {i+1}", fontsize=10)
            axes[0, i].axis("off")

            # Reconstructed images (bottom row)
            axes[1, i].imshow(result["reconstructed"])
            axes[1, i].set_title(f"Reconstructed {i+1}", fontsize=10)
            axes[1, i].axis("off")

        # Add average metrics
        avg_psnr = np.mean([r["metrics"]["psnr"] for r in results[:num_images]])
        avg_ssim = np.mean([r["metrics"]["ssim"] for r in results[:num_images]])
        avg_compression = np.mean(
            [r["metrics"]["compression_ratio"] for r in results[:num_images]]
        )

        metrics_text = f"Average PSNR: {avg_psnr:.2f} dB | "
        metrics_text += f"Average SSIM: {avg_ssim:.4f} | "
        metrics_text += f"Average Compression: {avg_compression:.1f}x"

        fig.suptitle(metrics_text, fontsize=12)
        plt.tight_layout()
        batch_path = os.path.join(output_dir, "batch_comparison.png")
        plt.savefig(batch_path, dpi=150, bbox_inches="tight")
        plt.close()

    def analyze_latent_space(
        self, dataloader: DataLoader, num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze the latent space properties.

        Args:
            dataloader: DataLoader with images
            num_samples: Number of samples to analyze

        Returns:
            Dictionary with latent space statistics
        """
        latents = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(latents) >= num_samples:
                    break

                batch = batch.to(self.device)
                batch_latents = self.encode_image(batch)
                latents.append(batch_latents.cpu())

        # Concatenate all latents
        all_latents = torch.cat(latents, dim=0)[:num_samples]

        # Calculate statistics
        latent_stats = {
            "mean": all_latents.mean(dim=0).numpy(),
            "std": all_latents.std(dim=0).numpy(),
            "min": all_latents.min(dim=0)[0].numpy(),
            "max": all_latents.max(dim=0)[0].numpy(),
            "shape": all_latents.shape,
        }

        return latent_stats


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Autoencoder inference for image compression testing"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="Path to a single image to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples to process from dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Image size for processing"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Latent dimension of the model"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Hidden dimensions of the model",
    )
    parser.add_argument(
        "--analyze_latent", action="store_true", help="Analyze latent space properties"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating autoencoder model...")
    model = create_autoencoder(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        output_size=args.image_size,
        device=device,
    )

    # Create inference object
    inference = AutoEncoderInference(model, device, args.image_size)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    inference.load_checkpoint(args.checkpoint_path)

    # Process single image if provided
    if args.image_path:
        logger.info(f"Processing single image: {args.image_path}")
        result = inference.process_single_image(
            args.image_path, save_results=True, output_dir=args.output_dir
        )

        # Print metrics
        metrics = result["metrics"]
        print("\nReconstruction Quality Metrics:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"Compression Ratio: {metrics['compression_ratio']:.1f}x")
        print(f"Original Size: {metrics['original_size_bytes']} bytes")
        print(f"Latent Size: {metrics['latent_size_bytes']} bytes")

    # Process batch from dataset
    logger.info("Creating dataloader for batch processing...")
    dataloader = create_celeba_dataloader(
        batch_size=8,
        image_size=args.image_size,
        split="test",
        num_workers=2,
    )

    logger.info(f"Processing {args.num_samples} samples from dataset...")
    batch_results = inference.process_batch(
        dataloader,
        num_samples=args.num_samples,
        save_results=True,
        output_dir=args.output_dir,
    )

    # Calculate average metrics
    avg_metrics = {
        "psnr": np.mean([r["metrics"]["psnr"] for r in batch_results]),
        "ssim": np.mean([r["metrics"]["ssim"] for r in batch_results]),
        "mse": np.mean([r["metrics"]["mse"] for r in batch_results]),
        "compression_ratio": np.mean(
            [r["metrics"]["compression_ratio"] for r in batch_results]
        ),
    }

    print("\nAverage Reconstruction Quality Metrics:")
    print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f}")
    print(f"MSE: {avg_metrics['mse']:.2f}")
    print(f"Compression Ratio: {avg_metrics['compression_ratio']:.1f}x")

    # Analyze latent space if requested
    if args.analyze_latent:
        logger.info("Analyzing latent space...")
        latent_stats = inference.analyze_latent_space(dataloader, num_samples=100)

        print("\nLatent Space Analysis:")
        print(f"Latent shape: {latent_stats['shape']}")
        mean_act = latent_stats["mean"].mean()
        std_act = latent_stats["std"].mean()
        print(f"Mean activation: {mean_act:.4f} ± {std_act:.4f}")
        min_act = latent_stats["min"].min()
        max_act = latent_stats["max"].max()
        print(f"Activation range: [{min_act:.4f}, {max_act:.4f}]")

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
