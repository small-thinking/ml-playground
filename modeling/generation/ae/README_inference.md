# Autoencoder Inference

This directory contains scripts for testing autoencoder image compression and recovery capabilities.

## Files

- `inference_autoencoder.py` - Main inference script with comprehensive functionality
- `example_inference.py` - Simple example showing how to use the inference script
- `autoencoder.py` - Autoencoder model implementation
- `train_autoencoder.py` - Training script

## Quick Start

### 1. Train a Model First

Before running inference, you need a trained model:

```bash
# From the repo root
python -m modeling.generation.ae.train_autoencoder
```

This will create checkpoints in `autoencoder_checkpoints/` directory.

### 2. Run Inference

#### Basic Usage

```bash
# From the repo root
python -m modeling.generation.ae.inference_autoencoder \
    --checkpoint_path autoencoder_checkpoints/best_model.pt \
    --num_samples 8 \
    --output_dir inference_results
```

#### With Single Image

```bash
python -m modeling.generation.ae.inference_autoencoder \
    --checkpoint_path autoencoder_checkpoints/best_model.pt \
    --image_path path/to/your/image.jpg \
    --output_dir inference_results
```

#### With Latent Space Analysis

```bash
python -m modeling.generation.ae.inference_autoencoder \
    --checkpoint_path autoencoder_checkpoints/best_model.pt \
    --num_samples 100 \
    --analyze_latent \
    --output_dir inference_results
```

## Command Line Options

- `--checkpoint_path`: Path to trained model checkpoint (required)
- `--image_path`: Path to single image for testing (optional)
- `--num_samples`: Number of samples to process from dataset (default: 8)
- `--output_dir`: Directory to save results (default: "inference_results")
- `--image_size`: Image size for processing (default: 64)
- `--latent_dim`: Latent dimension of the model (default: 128)
- `--hidden_dims`: Hidden dimensions of the model (default: [32, 64, 128, 256])
- `--analyze_latent`: Analyze latent space properties (flag)

## Output

The inference script generates:

1. **Individual Results** (if `--image_path` provided):

   - `{image_name}_original.png` - Original image
   - `{image_name}_reconstructed.png` - Reconstructed image
   - `{image_name}_comparison.png` - Side-by-side comparison with metrics

2. **Batch Results**:

   - `batch_comparison.png` - Grid showing multiple original/reconstructed pairs
   - Console output with quality metrics

3. **Quality Metrics**:
   - **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (dB)
   - **SSIM** (Structural Similarity Index): Closer to 1.0 is better
   - **MSE** (Mean Squared Error): Lower is better
   - **Compression Ratio**: How much the image is compressed

## Example Output

```
Reconstruction Quality Metrics:
PSNR: 28.45 dB
SSIM: 0.8234
MSE: 92.34
Compression Ratio: 12.3x
Original Size: 12288 bytes
Latent Size: 1024 bytes

Average Reconstruction Quality Metrics:
PSNR: 26.78 dB
SSIM: 0.7891
MSE: 134.56
Compression Ratio: 12.3x
```

## Understanding the Results

- **PSNR > 30 dB**: Excellent reconstruction quality
- **PSNR 20-30 dB**: Good reconstruction quality
- **PSNR < 20 dB**: Poor reconstruction quality

- **SSIM > 0.9**: Excellent structural similarity
- **SSIM 0.7-0.9**: Good structural similarity
- **SSIM < 0.7**: Poor structural similarity

- **Compression Ratio**: Shows how much the image is compressed. Higher values mean more compression but potentially lower quality.

## Troubleshooting

1. **Checkpoint not found**: Make sure you've trained a model first
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Image loading errors**: Ensure image paths are correct and images are in supported formats (JPG, PNG, etc.)

## Customization

You can modify the model architecture by changing the `--hidden_dims` and `--latent_dim` parameters to match your trained model's configuration.
