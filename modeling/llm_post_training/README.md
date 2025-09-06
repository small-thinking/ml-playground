# LLM Post Training

GRPO (Generative Reward-Powered Optimization) training for reasoning tasks.

## Quick Start

```bash
# Setup environment (with SSH key generation)
chmod +x setup_env.sh
./setup_env.sh --email your.email@example.com

# Activate environment
source ~/.bashrc

# Login to Hugging Face (required for gated models)
huggingface-cli login

# Run GRPO training
python reasoning_grpo.py --model-size 3B --use-lora
```

## Training Options

```bash
# Available model sizes
python reasoning_grpo.py --model-size {0.5B,1.5B,3B,8B}

# LoRA fine-tuning (recommended)
python reasoning_grpo.py --model-size 3B --use-lora

# Custom training parameters
python reasoning_grpo.py --model-size 3B --max-steps 1000 --batch-size 8
```

## Features

- **GRPO Training**: Reward-based optimization for reasoning tasks
- **Model Support**: Llama 3.1/3.2 and Qwen2 models (0.5B to 8B)
- **LoRA Support**: Efficient fine-tuning with PEFT
- **Workspace Management**: Organized storage in `/workspace/{models,data,cache}`
- **Automated Setup**: One-command environment setup with SSH key generation
- **Hugging Face Integration**: CLI tools for model access and authentication

## Prerequisites

- **Hugging Face Account**: Required for accessing gated models (Llama, etc.)
- **Hugging Face Token**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **GitHub/GitLab Account**: For SSH key setup (optional but recommended)

## Authentication

### Hugging Face Login

```bash
# Login with your Hugging Face token
huggingface-cli login

# Or set token as environment variable
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

### SSH Key Setup

The setup script automatically generates SSH keys. Add the public key to:

- **GitHub**: Settings → SSH and GPG keys → New SSH key
- **GitLab**: User Settings → SSH Keys → Add key
