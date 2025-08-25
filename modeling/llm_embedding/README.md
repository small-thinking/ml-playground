# SimCSE Embedding Model

This module implements a SimCSE (Simple Contrastive Learning of Sentence Embeddings) model for generating high-quality sentence embeddings using contrastive learning.

## Quick Setup

### Using uv with virtual environment (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv && source .venv/bin/activate  # On macOS/Linux
# or on Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## Architecture

The model uses a shared architecture defined in `model.py` that can be used by both training and inference scripts:

- **`model.py`**: Contains the `SimCSEModel` class and utility functions
- **`training.py`**: Training script using the shared model
- **`inference.py`**: Inference script using the shared model

## Usage

### 1. Train the Model

```bash
# From repository root (recommended)
python modeling/llm_embedding/training.py

# Quick verification run (1 batch only)
python modeling/llm_embedding/training.py --dry-run

# Or navigate to the directory first
cd modeling/llm_embedding
python training.py
```

This will:

- Load the SNLI dataset
- Train the model using contrastive learning
- Save the model to `models/simcse_model.pt`

### 2. Use the Trained Model

```bash
# From repository root (recommended)
python modeling/llm_embedding/inference.py "What can machine learning do?"

# Or navigate to the directory first
cd modeling/llm_embedding
python inference.py "What can machine learning do?"

# Multiple texts from file
python modeling/llm_embedding/inference.py --file texts.txt

# Save embeddings
python modeling/llm_embedding/inference.py "Your text" --output embeddings.npy
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
