# LLM Post Training

Docker setup for LLM post-training experiments.

## Quick Start

```bash
# Build with your Git credentials (required)
docker build -f modeling/llm_post_training/Dockerfile \
  --build-arg GIT_USER_NAME="Your Name" \
  --build-arg GIT_USER_EMAIL="your.email@example.com" \
  -t llm-post-training .

# Run container
docker run -it --gpus all \
  -v $(pwd):/workspace \
  -v ~/.ssh:/root/.ssh:ro \
  -v ~/.gitconfig:/root/.gitconfig:ro \
  -p 8888:8888 \
  llm-post-training
```

## Build Args

- `GIT_USER_NAME`: Your Git username (required)
- `GIT_USER_EMAIL`: Your Git email (required)
- `INSTALL_OLLAMA`: Set to "true" to install Ollama

## Features

- CUDA-enabled PyTorch 2.1.0
- Development tools: vim, git, tmux, oh-my-bash
- Mounts project directory and SSH keys
- Port 8888 for Jupyter
