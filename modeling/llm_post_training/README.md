# LLM Post Training

Examples of post-training techniques for Large Language Models:

- **SFT**: Supervised Fine-Tuning for instruction following
- **DPO**: Direct Preference Optimization for preference learning
- **GRPO**: Generative Reward-Powered Optimization for reasoning tasks

## Quick Start

```bash
# From repository root
uv sync --extra dev

# Login to Hugging Face
huggingface-cli login

# Train models
uv run -m modeling.llm_post_training.instruction_sft --model-size 3B --use-lora
uv run -m modeling.llm_post_training.dpo_training --dataset tech-tao/yizhipian_yizhipian_dpo_data --model-size 3B --use-lora
uv run -m modeling.llm_post_training.reasoning_grpo --model-size 3B --use-lora
```

## Models & Datasets

| Training | Models                           | Dataset             | Purpose                            |
| -------- | -------------------------------- | ------------------- | ---------------------------------- |
| **SFT**  | Base models (not instruct-tuned) | Alpaca              | Instruction following              |
| **DPO**  | Any base/SFT model               | Preference datasets | Preference learning                |
| **GRPO** | Llama 3.1/3.2, Qwen2 (0.5B-8B)   | Mini-reasoning      | Reasoning with structured thinking |

## Usage

### SFT Training

```bash
uv run -m modeling.llm_post_training.instruction_sft --model-size 3B --use-lora --max-steps 2000
```

### DPO Training

```bash
# Train with predefined model size
uv run -m modeling.llm_post_training.dpo_training --dataset tech-tao/yizhipian_yizhipian_dpo_data --model-size 3B --use-lora

# Train with custom base model (including SFT-tuned models)
uv run -m modeling.llm_post_training.dpo_training --dataset tech-tao/gang-jing_contrarian_dpo_data --base-model ./models/Llama-3.2-3B-LoRA-SFT --use-lora

# Train with custom parameters
uv run -m modeling.llm_post_training.dpo_training \
    --dataset tech-tao/yizhipian_yizhipian_dpo_data \
    --base-model meta-llama/Llama-3.2-3B \
    --use-lora \
    --beta 0.1 \
    --learning-rate 5e-6 \
    --max-steps 1000
```

### GRPO Training

```bash
uv run -m modeling.llm_post_training.reasoning_grpo --model-size 3B --use-lora --max-steps 1000
```

### Model Comparison (SFT)

```bash
# Compare base vs SFT models
uv run -m modeling.llm_post_training.compare_base_vs_sft \
    --base-model meta-llama/Llama-3.2-3B \
    --sft-model /workspace/models/Llama-3.2-3B-Base-LoRA-SFT
```

### Interactive Chat Mode

```bash
# Chat with a base model
uv run -m modeling.llm_post_training.chat_mode --model-path meta-llama/Llama-3.2-3B

# Chat with an SFT model
uv run -m modeling.llm_post_training.chat_mode --model-path /workspace/models/Llama-3.2-3B-LoRA-SFT

# Chat with custom parameters
uv run -m modeling.llm_post_training.chat_mode \
    --model-path meta-llama/Llama-3.2-3B \
    --temperature 0.8 \
    --max-length 512 \
    --use-4bit

# Single prompt test
uv run -m modeling.llm_post_training.chat_mode \
    --model-path meta-llama/Llama-3.2-3B \
    --prompt "Write a haiku about machine learning"
```

### Model Management

```bash
# List available models
uv run -m modeling.llm_post_training.model_utils --directory ./models

# Validate a model
uv run -m modeling.llm_post_training.model_utils --validate /path/to/model

# Get model size info
uv run -m modeling.llm_post_training.model_utils --size-info /path/to/model
```

## Key Features

- **LoRA Support**: Efficient fine-tuning with PEFT
- **Base Models**: SFT uses base models to show clear transformation
- **Configurable Datasets**: DPO supports any Hugging Face preference dataset
- **Flexible Base Models**: DPO can use any base model or SFT-tuned model
- **Workspace Management**: Organized storage in `/workspace/{models,data,cache}`
- **Comparison Tools**: Side-by-side base vs SFT model comparison
- **Interactive Chat**: Real-time chat interface for any trained model
- **Model Management**: Utilities for discovering and validating models
- **Quantization Support**: 4-bit and 8-bit quantization for memory efficiency

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
