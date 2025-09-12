"""Example script showing how to use the autoencoder inference.

This script demonstrates how to:
1. Load a trained autoencoder model
2. Test image compression and recovery
3. Visualize results with quality metrics
"""

import torch
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modeling.generation.ae.autoencoder import create_autoencoder
from modeling.generation.ae.inference_autoencoder import AutoEncoderInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example usage of autoencoder inference."""
    # Configuration
    config = {
        "checkpoint_path": "autoencoder_checkpoints/best_model.pt",
        "image_size": 64,
        "latent_dim": 128,
        "hidden_dims": [32, 64, 128, 256],
        "output_dir": "inference_results",
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    logger.info("Creating autoencoder model...")
    model = create_autoencoder(
        input_channels=3,
        latent_dim=config["latent_dim"],
        hidden_dims=config["hidden_dims"],
        output_size=config["image_size"],
        device=device,
    )

    # Create inference object
    inference = AutoEncoderInference(model, device, config["image_size"])

    # Check if checkpoint exists
    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.exists():
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}. "
            "Please train the model first or provide a valid checkpoint path."
        )
        return

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    inference.load_checkpoint(str(checkpoint_path))

    # Example: Process a single image (if you have one)
    # Uncomment and modify the path below to test with your own image
    # image_path = "path/to/your/image.jpg"
    # if Path(image_path).exists():
    #     logger.info(f"Processing single image: {image_path}")
    #     result = inference.process_single_image(
    #         image_path,
    #         save_results=True,
    #         output_dir=config["output_dir"]
    #     )
    #
    #     # Print metrics
    #     metrics = result["metrics"]
    #     print(f"\nReconstruction Quality Metrics:")
    #     print(f"PSNR: {metrics['psnr']:.2f} dB")
    #     print(f"SSIM: {metrics['ssim']:.4f}")
    #     print(f"MSE: {metrics['mse']:.2f}")
    #     print(f"Compression Ratio: {metrics['compression_ratio']:.1f}x")

    logger.info("Inference setup complete!")
    logger.info(f"To run full inference, use:")
    logger.info(f"python -m modeling.generation.ae.inference_autoencoder \\")
    logger.info(f"    --checkpoint_path {config['checkpoint_path']} \\")
    logger.info(f"    --num_samples 8 \\")
    logger.info(f"    --output_dir {config['output_dir']}")


if __name__ == "__main__":
    main()
