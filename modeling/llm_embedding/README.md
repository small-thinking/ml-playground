# SimCSE Embedding Model

This module implements a SimCSE (Simple Contrastive Learning of Sentence Embeddings) model for generating high-quality sentence embeddings using contrastive learning.

## Quick Setup

### Using uv with virtual environment (recommended)

```bash
# From the repository root
uv sync --extra dev
```

## Architecture

The model uses a shared architecture defined in `model.py` that can be used by both training and inference scripts:

- **`model.py`**: Contains the `SimCSEModel` class and utility functions
- **`training.py`**: Training script using the shared model
- **`inference.py`**: Inference script using the shared model

## Usage

### 1. Train the Model

```bash
uv run -m modeling.llm_embedding.training

# Quick verification run (1 batch only)
uv run -m modeling.llm_embedding.training --dry-run
```

This will:

- Load the SNLI dataset
- Train the model using contrastive learning
- Save the model to `models/simcse_model.pt`

### 2. Use the Trained Model

```bash
uv run -m modeling.llm_embedding.inference "What can machine learning do?"

# Multiple texts from file
uv run -m modeling.llm_embedding.inference --file texts.txt

# Save embeddings
uv run -m modeling.llm_embedding.inference "Your text" --output embeddings.npy
```

## Model Architecture

The `SimCSEModel` class implements:

- **BERT backbone**: Uses pre-trained BERT for feature extraction
- **SimCSE approach**: Generates two views of the same input using dropout
- **Contrastive learning**: Uses InfoNCE loss to learn meaningful embeddings
- **Mean pooling**: Aggregates token embeddings to sentence embeddings
- **L2 normalization**: Ensures embeddings are unit vectors

## Key Features

- **Shared architecture**: Same model class used for training and inference
- **Type annotations**: Full type hints for better code quality
- **Error handling**: Graceful fallbacks if model loading fails
- **Batch processing**: Efficient handling of large datasets
- **Device support**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU
- **Similarity analysis**: Built-in similarity matrix computation

## Configuration

### Training Parameters

- `MODEL_NAME`: Base BERT model (default: "bert-base-uncased")
- `BATCH_SIZE`: Training batch size (default: 64)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `EPOCHS`: Number of training epochs (default: 1)
- `MAX_LEN`: Maximum sequence length (default: 64)

### Inference Parameters

- `--model`: Path to trained model (default: "models/simcse_model.pt")
- `--model-name`: Base model name (default: "bert-base-uncased")
- `--batch-size`: Inference batch size (default: 32)
- `--output`: Output file for embeddings

## Example Output

### Training

```
🚀 SimCSE Embedding Demo
📱 Using device: mps
🤖 Model: bert-base-uncased
📚 Loading dataset...
📊 Loaded 50000 sentences
🤖 Initializing model...
🔄 Setting up data loader...
🎯 Starting training for 1 epochs...
✅ Training completed! Final average loss: 0.1234
💾 Saving model...
✅ Model saved to models/simcse_model.pt
```

### Inference

```
✅ Successfully imported SimCSEModel from model.py
✅ Loading trained SimCSE model from models/simcse_model.pt
✅ Model loaded successfully!
🍎 Using Apple Silicon GPU (MPS)
🤖 Loading model: bert-base-uncased
🔍 Generating embeddings...

📊 Results:
   Embedding shape: torch.Size([1, 768])
   Embedding dimension: 768

📝 Text 1: What can machine learning do?
   Embedding norm: 1.0000
   First 5 values: [0.1234 0.5678 0.9012 0.3456 0.7890]
```

## Dependencies

- `torch`: PyTorch for deep learning
- `transformers`: Hugging Face transformers for BERT
- `datasets`: Hugging Face datasets for data loading
- `tqdm`: Progress bars
- `numpy`: Numerical computing

## Notes

- The model automatically detects and uses the best available device (CUDA > MPS > CPU)
- Training uses the SNLI dataset by default, but can be easily modified
- The inference script includes fallback to pre-trained BERT if no trained model is found
- All embeddings are L2-normalized for consistent similarity computations
