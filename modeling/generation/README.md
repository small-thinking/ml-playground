# Generation Module

This module contains implementations for auto-encoder and VAE training, including image dataset dataloaders (CelebA and AFHQv2) and complete auto-encoder implementation.

## Auto-Encoder Implementation

A complete auto-encoder implementation with convolutional encoder-decoder architecture, training logic, and visualization tools.

### Features

- **Convolutional Architecture**: Efficient encoder-decoder with configurable layers
- **Flexible Configuration**: Customizable latent dimensions and hidden layers
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Visualization**: Reconstruction visualization and training metrics
- **Checkpointing**: Model saving and loading capabilities

### Quick Start

```python
from modeling.generation import create_autoencoder, AutoEncoderTrainer, create_image_dataloader

# Create model
model = create_autoencoder(
    input_channels=3,
    latent_dim=128,
    hidden_dims=[32, 64, 128, 256],
    output_size=64
)

# Create data loaders (supports both CelebA and AFHQv2)
train_loader = create_image_dataloader(dataset_type="afhq", batch_size=32, image_size=64, split="train")
val_loader = create_image_dataloader(dataset_type="afhq", batch_size=32, image_size=64, split="validation")

# Create trainer
trainer = AutoEncoderTrainer(model, train_loader, val_loader, device)

# Train
trainer.train(num_epochs=50)
```

### Training

Run the training script:

```bash
cd modeling/generation
python train_autoencoder.py
```

### Testing

Test the implementation:

```bash
python test_autoencoder.py
```

## Image DataLoaders

Simple and efficient PyTorch DataLoaders for image datasets (CelebA and AFHQv2) from Hugging Face, optimized for auto-encoder and VAE training.

### Features

- **Simple API**: Easy-to-use interface with sensible defaults
- **Flexible Configuration**: Customizable batch size, image size, and preprocessing
- **Multiple Datasets**: Support for CelebA and AFHQv2 datasets
- **Multiple Splits**: Support for train, validation, and test splits
- **Auto-encoder/VAE Ready**: Images normalized to [-1, 1] range by default
- **Efficient Loading**: Multi-worker support with proper memory pinning

### Quick Start

```python
from modeling.generation import create_image_dataloader

# Create a simple dataloader (CelebA)
dataloader = create_image_dataloader(
    dataset_type="celeba",
    batch_size=32,
    image_size=64,
    split="train"
)

# Or use AFHQv2 dataset
dataloader = create_image_dataloader(
    dataset_type="afhq",
    batch_size=32,
    image_size=64,
    split="train"
)

# Iterate through batches
for batch in dataloader:
    # batch shape: (32, 3, 64, 64)
    # batch values: [-1, 1] range
    pass
```

### Advanced Usage

```python
from modeling.generation import ImageDataLoader

# Create a factory for multiple dataloaders
factory = ImageDataLoader(
    dataset_type="afhq",  # or "celeba"
    batch_size=64,
    image_size=128,
    normalize=True,
    num_workers=4
)

# Get different splits
train_loader = factory.get_train_dataloader()
val_loader = factory.get_val_dataloader()
test_loader = factory.get_test_dataloader()
```

### Configuration Options

| Parameter      | Default | Description                           |
| -------------- | ------- | ------------------------------------- |
| `dataset_type` | "afhq"  | Dataset type ("celeba" or "afhq")     |
| `batch_size`   | 32      | Number of images per batch            |
| `image_size`   | 64      | Target image size (square)            |
| `split`        | "train" | Dataset split (train/validation/test) |
| `normalize`    | True    | Normalize to [-1, 1] range            |
| `num_workers`  | 4       | Number of worker processes            |
| `cache_dir`    | None    | Directory to cache dataset            |

### Testing

Run the test script to verify everything works:

```bash
cd modeling/generation
python test_image_dataloader.py
```

### Examples

See `example_usage.py` for complete examples including:

- Basic usage patterns
- Auto-encoder training loops
- VAE training loops
- Factory pattern usage

### Dependencies

- `torch` >= 2.6.0
- `torchvision` (for transforms)
- `datasets` >= 2.18.0
- `transformers` >= 4.40.0

All dependencies are already included in the project's `pyproject.toml`.
