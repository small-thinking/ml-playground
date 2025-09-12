"""VAE inference script for image generation, reconstruction, and latent space exploration.

Can be run from repo root:
    python -m modeling.generation.ae.inference_vae \
        --checkpoint_path vae_checkpoints/best_model.pt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
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
from modeling.generation.ae.vae import create_vae
from modeling.generation.image_dataloader import create_image_dataloader

logger = logging.getLogger(__name__)


class VAEInference:
    """VAE inference class for generation, reconstruction, and latent space exploration."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        image_size: int = 64,
    ) -> None:
        """
        Initialize VAE inference.

        Args:
            model: Trained VAE model
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
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}, "
            f"beta={checkpoint.get('beta', 'unknown')}"
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
            tensor: Output tensor (1, C, H, W) in [0, 1] range

        Returns:
            Image as numpy array (H, W, C) in [0, 255] range
        """
        # Move to CPU and remove batch dimension
        image = tensor.squeeze(0).cpu()

        # VAE output is already in [0, 1] range
        image = torch.clamp(image, 0, 1)

        # Convert to numpy
        if image.shape[0] == 1:  # Grayscale
            image = image.squeeze(0).numpy()  # (H, W)
        else:  # RGB
            image = image.permute(1, 2, 0).numpy()  # (H, W, C)

        return (image * 255).astype(np.uint8)

    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters.

        Args:
            image: Input image tensor

        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        with torch.no_grad():
            mu, logvar = self.model.encode(image)
        return mu, logvar

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

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generate new samples from prior distribution.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Generated samples tensor
        """
        with torch.no_grad():
            generated = self.model.generate(num_samples, self.device)
        return generated

    def interpolate_images(
        self, image1: torch.Tensor, image2: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.

        Args:
            image1: First input image
            image2: Second input image
            num_steps: Number of interpolation steps

        Returns:
            Interpolated images tensor
        """
        with torch.no_grad():
            interpolated = self.model.interpolate(image1, image2, num_steps)
        return interpolated

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
        # For VAE, we approximate the latent size
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
        output_dir: str = "vae_inference_results",
    ) -> Dict[str, Any]:
        """
        Process a single image through the VAE.

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

    def generate_and_save_samples(
        self,
        num_samples: int = 16,
        save_results: bool = True,
        output_dir: str = "vae_inference_results",
    ) -> List[np.ndarray]:
        """
        Generate and save new samples.

        Args:
            num_samples: Number of samples to generate
            save_results: Whether to save results
            output_dir: Directory to save results

        Returns:
            List of generated images
        """
        # Generate samples
        generated_tensor = self.generate_samples(num_samples)
        generated_images = []

        for i in range(num_samples):
            img_tensor = generated_tensor[i : i + 1]
            img_array = self.postprocess_image(img_tensor)
            generated_images.append(img_array)

        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)

            # Create grid of generated images
            self.create_generation_grid(generated_images, output_dir)

            # Save individual images
            for i, img in enumerate(generated_images):
                img_path = os.path.join(output_dir, f"generated_{i+1:03d}.png")
                Image.fromarray(img).save(img_path)

        return generated_images

    def create_generation_grid(
        self,
        images: List[np.ndarray],
        output_dir: str,
        grid_size: Tuple[int, int] = None,
    ) -> None:
        """
        Create a grid of generated images.

        Args:
            images: List of generated images
            output_dir: Directory to save results
            grid_size: Grid size (rows, cols). If None, auto-calculate
        """
        num_images = len(images)
        if grid_size is None:
            # Calculate grid size
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_size

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(rows * cols):
            if i < num_images:
                axes[i].imshow(images[i])
                axes[i].set_title(f"Generated {i+1}", fontsize=10)
            axes[i].axis("off")

        plt.tight_layout()
        grid_path = os.path.join(output_dir, "generated_grid.png")
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close()

    def create_interpolation_grid(
        self,
        image1_path: str,
        image2_path: str,
        num_steps: int = 10,
        save_results: bool = True,
        output_dir: str = "vae_inference_results",
    ) -> List[np.ndarray]:
        """
        Create interpolation between two images.

        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            num_steps: Number of interpolation steps
            save_results: Whether to save results
            output_dir: Directory to save results

        Returns:
            List of interpolated images
        """
        # Load and preprocess images
        img1 = Image.open(image1_path).convert("RGB")
        img1_array = np.array(img1)
        img1_tensor = self.preprocess_image(img1_array)

        img2 = Image.open(image2_path).convert("RGB")
        img2_array = np.array(img2)
        img2_tensor = self.preprocess_image(img2_array)

        # Interpolate
        interpolated_tensor = self.interpolate_images(
            img1_tensor, img2_tensor, num_steps
        )

        # Convert to numpy arrays
        interpolated_images = []
        for i in range(num_steps):
            img_tensor = interpolated_tensor[i : i + 1]
            img_array = self.postprocess_image(img_tensor)
            interpolated_images.append(img_array)

        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)

            # Create interpolation grid
            fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
            if num_steps == 1:
                axes = [axes]

            for i, img in enumerate(interpolated_images):
                axes[i].imshow(img)
                axes[i].set_title(f"Step {i+1}", fontsize=8)
                axes[i].axis("off")

            plt.tight_layout()
            interp_path = os.path.join(output_dir, "interpolation.png")
            plt.savefig(interp_path, dpi=150, bbox_inches="tight")
            plt.close()

        return interpolated_images

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
        mus = []
        logvars = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(mus) * dataloader.batch_size >= num_samples:
                    break

                batch = batch.to(self.device)
                mu, logvar = self.encode_image(batch)
                mus.append(mu.cpu())
                logvars.append(logvar.cpu())

        # Concatenate all latents
        all_mus = torch.cat(mus, dim=0)[:num_samples]
        all_logvars = torch.cat(logvars, dim=0)[:num_samples]

        # Calculate statistics
        latent_stats = {
            "mu_mean": all_mus.mean(dim=0).numpy(),
            "mu_std": all_mus.std(dim=0).numpy(),
            "mu_min": all_mus.min(dim=0)[0].numpy(),
            "mu_max": all_mus.max(dim=0)[0].numpy(),
            "logvar_mean": all_logvars.mean(dim=0).numpy(),
            "logvar_std": all_logvars.std(dim=0).numpy(),
            "shape": all_mus.shape,
        }

        return latent_stats

    def process_batch(
        self,
        dataloader: DataLoader,
        num_samples: int = 8,
        save_results: bool = True,
        output_dir: str = "vae_inference_results",
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
                    batch.size(0),
                    num_samples - batch_idx * dataloader.batch_size,
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
        self,
        results: List[Dict[str, Any]],
        output_dir: str,
        max_images: int = 8,
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


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="VAE inference for image generation and reconstruction"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a single image to test",
    )
    parser.add_argument(
        "--image1_path",
        type=str,
        default=None,
        help="Path to first image for interpolation",
    )
    parser.add_argument(
        "--image2_path",
        type=str,
        default=None,
        help="Path to second image for interpolation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=8,
        help="Number of samples to process from dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vae_inference_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Image size for processing"
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent dimension of the model",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Hidden dimensions of the model",
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Beta parameter of the VAE"
    )
    parser.add_argument(
        "--analyze_latent",
        action="store_true",
        help="Analyze latent space properties",
    )
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only generate new samples",
    )
    parser.add_argument(
        "--interpolate_only",
        action="store_true",
        help="Only perform interpolation",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating VAE model...")
    model = create_vae(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        output_size=args.image_size,
        beta=args.beta,
        device=device,
    )

    # Create inference object
    inference = VAEInference(model, device, args.image_size)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    inference.load_checkpoint(args.checkpoint_path)

    # Generate samples
    if args.generate_only or not (
        args.image_path or args.image1_path or args.image2_path
    ):
        logger.info(f"Generating {args.num_samples} samples...")
        generated_images = inference.generate_and_save_samples(
            num_samples=args.num_samples,
            save_results=True,
            output_dir=args.output_dir,
        )
        print(f"Generated {len(generated_images)} samples")

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

    # Perform interpolation if both images provided
    if args.image1_path and args.image2_path:
        logger.info(f"Interpolating between {args.image1_path} and {args.image2_path}")
        interpolated_images = inference.create_interpolation_grid(
            args.image1_path,
            args.image2_path,
            num_steps=10,
            save_results=True,
            output_dir=args.output_dir,
        )
        print(f"Created {len(interpolated_images)} interpolation steps")

    # Process batch from dataset (unless only generating or interpolating)
    if not (args.generate_only or args.interpolate_only):
        logger.info("Creating dataloader for batch processing...")
        dataloader = create_image_dataloader(
            dataset_type="afhq",
            batch_size=8,
            image_size=args.image_size,
            split="test",
            num_workers=2,
        )

        logger.info(f"Processing {args.num_test_samples} samples from dataset...")
        batch_results = inference.process_batch(
            dataloader,
            num_samples=args.num_test_samples,
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
            mu_mean = latent_stats["mu_mean"].mean()
            mu_std = latent_stats["mu_std"].mean()
            print(f"Mean activation: {mu_mean:.4f} ± {mu_std:.4f}")
            mu_min = latent_stats["mu_min"].min()
            mu_max = latent_stats["mu_max"].max()
            print(f"Mean range: [{mu_min:.4f}, {mu_max:.4f}]")
            logvar_mean = latent_stats["logvar_mean"].mean()
            print(f"Average log variance: {logvar_mean:.4f}")

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
