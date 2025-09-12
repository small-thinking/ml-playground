# Autoencoder (AE)

A complete autoencoder implementation with convolutional encoder-decoder architecture for image compression and reconstruction.

## Features

- **Convolutional Architecture**: Efficient encoder-decoder with configurable layers
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Inference Tools**: Image compression, reconstruction, and quality analysis
- **CelebA Integration**: Ready-to-use dataloader for face image training

## Quick Start

### Training

```bash
# From repo root
python -m modeling.generation.ae.train_autoencoder
```

This creates checkpoints in `autoencoder_checkpoints/` directory.

### Inference

```bash
# Basic inference with 8 samples
python -m modeling.generation.ae.inference_autoencoder \
    --checkpoint_path autoencoder_checkpoints/best_model.pt \
    --num_samples 8 \
    --output_dir inference_results

# Single image reconstruction
python -m modeling.generation.ae.inference_autoencoder \
    --checkpoint_path autoencoder_checkpoints/best_model.pt \
    --image_path path/to/image.jpg \
    --output_dir inference_results
```

## Model Architecture

- **Input**: 3-channel RGB images (64x64 by default)
- **Latent Space**: 128 dimensions (configurable)
- **Hidden Layers**: [32, 64, 128, 256] channels (configurable)
- **Output**: Reconstructed images matching input size

## Quality Metrics

The inference script provides:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (dB)
- **SSIM** (Structural Similarity Index): Closer to 1.0 is better
- **MSE** (Mean Squared Error): Lower is better
- **Compression Ratio**: Latent space compression factor

## Output

Inference generates:

- Original and reconstructed image comparisons
- Quality metrics in console output
- Batch visualization grids
- Individual image analysis (when using `--image_path`)
