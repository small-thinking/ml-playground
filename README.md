# ML Playground

A machine learning playground for experimenting with various ML models and techniques.

## Project Structure

```
ml-playground/
├── modeling/                    # ML model implementations
│   └── llm_embedding/          # LLM embedding demos
│       ├── training.py                # Training script
│       ├── inference.py               # Inference script
│       ├── model.py                   # Model architecture
│       └── README.md                  # Detailed documentation
├── visualizations/             # Data visualization demos
│   └── pac-man/               # Pac-Man visualization
├── pyproject.toml             # Project dependencies
├── dockerfile                 # Docker configuration
└── README.md                  # This file
```

## Quick Start

### 1. Install Dependencies

```bash
uv pip install -e .
```

### 2. Run Demos

#### LLM Embedding Demo

```bash
cd modeling/llm_embedding
python training.py
python inference.py "Your text here"
```

See `modeling/llm_embedding/README.md` for detailed documentation.

## Features

- **LLM Embedding**: Unsupervised sentence embedding learning using SimCSE
- **GPU Support**: Automatic CUDA/MPS detection and utilization
- **Docker Ready**: Complete Docker setup with CUDA support
- **Clean Code**: Type annotations, proper error handling, modular design

## Configuration

### Environment

- Python 3.8+
- PyTorch 2.0+
- CUDA support (optional but recommended)

### Key Dependencies

- `torch`: PyTorch for deep learning
- `transformers`: HuggingFace transformers library
- `datasets`: HuggingFace datasets library
- `accelerate`: Distributed training support

## Docker Usage

```bash
docker build -t ml-playground .
docker run --gpus all -it ml-playground
```

## Development

### Code Style

- Follow PEP 8 guidelines
- Use type annotations
- Add English comments
- Keep functions concise and focused

### Adding New Models

1. Create a new directory in `modeling/`
2. Implement your model with proper documentation
3. Add tests if needed
4. Update this README

## License

MIT License - feel free to use this code for your projects!
