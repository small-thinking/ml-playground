# Autoencoder & VAE

Convolutional autoencoder and variational autoencoder implementations for image compression, reconstruction, and generation.

## Autoencoder (AE)

### Quick Start

```bash
# Training
python -m modeling.generation.ae.train_autoencoder

# Inference - reconstruct 8 samples
python -m modeling.generation.ae.inference_autoencoder \
    --checkpoint_path autoencoder_checkpoints/best_model.pt \
    --num_samples 8

# Single image reconstruction
python -m modeling.generation.ae.inference_autoencoder \
    --checkpoint_path autoencoder_checkpoints/best_model.pt \
    --image_path path/to/image.jpg
```

### Architecture

- **Input**: 3-channel RGB images (64x64)
- **Latent**: 128 dimensions
- **Hidden**: [32, 64, 128, 256] channels
- **Output**: Reconstructed images

## Variational Autoencoder (VAE)

### Quick Start

```bash
# Training
python -m modeling.generation.ae.train_vae

# Generate new samples
python -m modeling.generation.ae.inference_vae \
    --checkpoint_path vae_checkpoints/best_model.pt \
    --generate_only --num_samples 16

# Reconstruct single image
python -m modeling.generation.ae.inference_vae \
    --checkpoint_path vae_checkpoints/best_model.pt \
    --image_path path/to/image.jpg

# Interpolate between images
python -m modeling.generation.ae.inference_vae \
    --checkpoint_path vae_checkpoints/best_model.pt \
    --image1_path path/to/image1.jpg \
    --image2_path path/to/image2.jpg \
    --interpolate_only
```

### Key Features

- **Probabilistic Encoding**: Learns distribution over latent space (mean + variance)
- **Generation**: Sample new images from prior distribution
- **Beta-VAE**: Configurable beta parameter for disentangled representations
- **ELBO Loss**: Combines reconstruction + KL divergence

## Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index (0-1)
- **MSE**: Mean Squared Error
- **Compression**: Latent space ratio

## Output

- **AE**: Checkpoints in `autoencoder_checkpoints/`
- **VAE**: Checkpoints in `vae_checkpoints/` + Wandb logs
