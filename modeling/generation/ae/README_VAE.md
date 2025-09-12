# Variational Auto-Encoder (VAE) Implementation

This directory contains a complete implementation of a Variational Auto-Encoder (VAE) for image generation and reconstruction.

## Files

- `vae.py` - Core VAE model implementation with encoder, decoder, and reparameterization trick
- `train_vae.py` - Training script with ELBO loss function and beta-VAE support
- `inference_vae.py` - Inference script for generation, reconstruction, and latent space exploration
- `README_VAE.md` - This documentation file

## Key Features

### VAE Model (`vae.py`)

- **VAEEncoder**: Convolutional encoder that outputs mean and log-variance of latent distribution
- **VAEDecoder**: Convolutional decoder that reconstructs images from latent samples
- **VariationalAutoEncoder**: Complete VAE with reparameterization trick
- **Beta-VAE support**: Configurable beta parameter for controlling KL divergence weight
- **Generation methods**: Sample from prior, interpolate in latent space

### Training (`train_vae.py`)

- **ELBO Loss**: Combines reconstruction loss (BCE) and KL divergence
- **Beta-VAE**: Configurable beta parameter for disentangled representation learning
- **Wandb Integration**: Comprehensive logging of losses, reconstructions, and generated samples
- **Learning Rate Scheduling**: Cosine annealing with step-level or epoch-level scheduling
- **Checkpointing**: Save/load model states with training history

### Inference (`inference_vae.py`)

- **Image Reconstruction**: Encode and decode images with quality metrics (PSNR, SSIM)
- **Sample Generation**: Generate new images from prior distribution
- **Latent Interpolation**: Smooth interpolation between images in latent space
- **Latent Space Analysis**: Statistical analysis of learned representations
- **Batch Processing**: Process multiple images with visualization

## Usage

### Training

```bash
# Train VAE on AFHQ dataset
python -m modeling.generation.ae.train_vae

# Training will create checkpoints in vae_checkpoints/
```

### Inference

```bash
# Generate new samples
python -m modeling.generation.ae.inference_vae \
    --checkpoint_path vae_checkpoints/best_model.pt \
    --generate_only \
    --num_samples 16

# Reconstruct a single image
python -m modeling.generation.ae.inference_vae \
    --checkpoint_path vae_checkpoints/best_model.pt \
    --image_path path/to/image.jpg

# Interpolate between two images
python -m modeling.generation.ae.inference_vae \
    --checkpoint_path vae_checkpoints/best_model.pt \
    --image1_path path/to/image1.jpg \
    --image2_path path/to/image2.jpg \
    --interpolate_only

# Analyze latent space
python -m modeling.generation.ae.inference_vae \
    --checkpoint_path vae_checkpoints/best_model.pt \
    --analyze_latent
```

## Key Differences from Autoencoder

1. **Probabilistic Encoding**: VAE learns a distribution over latent space (mean + variance) instead of deterministic encoding
2. **Reparameterization Trick**: Enables gradient flow through stochastic sampling
3. **ELBO Loss**: Combines reconstruction quality with regularization of latent distribution
4. **Generation Capability**: Can generate new samples by sampling from prior distribution
5. **Beta-VAE**: Supports disentangled representation learning with configurable beta parameter

## Configuration

### Model Parameters

- `latent_dim`: Dimension of latent space (default: 128)
- `hidden_dims`: Hidden dimensions for encoder/decoder (default: [32, 64, 128, 256])
- `beta`: Beta parameter for beta-VAE (default: 1.0)

### Training Parameters

- `learning_rate`: Learning rate (default: 1e-3)
- `batch_size`: Batch size (default: 64)
- `num_epochs`: Number of training epochs (default: 10)
- `use_cosine_scheduler`: Enable cosine annealing (default: True)

## Output

Training creates:

- Model checkpoints in `vae_checkpoints/`
- Wandb logs with loss curves, reconstructions, and generated samples

Inference creates:

- Reconstructed images
- Generated samples
- Interpolation sequences
- Quality metrics (PSNR, SSIM, compression ratio)
- Latent space statistics

## Dependencies

- PyTorch
- Wandb (for logging)
- PIL, matplotlib, numpy (for visualization)
- scikit-image (for SSIM calculation)
