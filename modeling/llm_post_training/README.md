# LLM Post Training

Examples of post-training techniques for Large Language Models:

- **GRPO**: Generative Reward-Powered Optimization for reasoning tasks
- **SFT**: Supervised Fine-Tuning for instruction following

## Quick Start

```bash
# Setup environment
chmod +x setup_env.sh
./setup_env.sh --email your.email@example.com
source ~/.bashrc

# Login to Hugging Face
huggingface-cli login

# Train models
python reasoning_grpo.py --model-size 3B --use-lora    # GRPO
python instruction_sft.py --model-size 3B --use-lora   # SFT
```

## Models & Datasets

| Training | Models                           | Dataset        | Purpose                            |
| -------- | -------------------------------- | -------------- | ---------------------------------- |
| **GRPO** | Llama 3.1/3.2, Qwen2 (0.5B-8B)   | Mini-reasoning | Reasoning with structured thinking |
| **SFT**  | Base models (not instruct-tuned) | Alpaca         | Instruction following              |

## Usage

### GRPO Training

```bash
python reasoning_grpo.py --model-size 3B --use-lora --max-steps 1000
```

### SFT Training

```bash
python instruction_sft.py --model-size 3B --use-lora --max-steps 2000
```

### Model Comparison (SFT)

```bash
# Compare base vs SFT models
python compare_base_vs_sft.py \
    --base-model meta-llama/Llama-3.2-3B \
    --sft-model /workspace/models/Llama-3.2-3B-Base-LoRA-SFT
```

## Key Features

- **LoRA Support**: Efficient fine-tuning with PEFT
- **Base Models**: SFT uses base models to show clear transformation
- **Workspace Management**: Organized storage in `/workspace/{models,data,cache}`
- **Comparison Tools**: Side-by-side base vs SFT model comparison

## Prerequisites

- **Hugging Face Account**: Required for gated models (Llama, etc.)
- **Hugging Face Token**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **CUDA GPU**: Recommended for training

## Authentication

```bash
# Login with your Hugging Face token
huggingface-cli login

# Or set token as environment variable
export HUGGINGFACE_HUB_TOKEN=your_token_here
```
