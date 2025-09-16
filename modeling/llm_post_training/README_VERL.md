# VERL-based GRPO Training

This directory now includes a VERL-based implementation of GRPO (Generative Reward-Powered Optimization) training for reasoning tasks.

## Files

- **`reasoning_grpo_verl.py`**: VERL-based GRPO training script (with built-in validation)
- **`reasoning_grpo.py`**: Original TRL-based GRPO training script (for comparison)

## Key Differences: VERL vs TRL

| Feature              | TRL Implementation          | VERL Implementation            |
| -------------------- | --------------------------- | ------------------------------ |
| **Library**          | `trl.GRPOTrainer`           | `verl.GRPOTrainer`             |
| **Configuration**    | `GRPOConfig` (TRL)          | `GRPOConfig` (VERL)            |
| **Reward Functions** | Multiple separate functions | Single unified reward function |
| **GRPO Parameters**  | Standard TRL parameters     | VERL-specific GRPO parameters  |
| **Model Wrapper**    | Direct model usage          | VERL model wrapper support     |

## Usage

### 1. Basic VERL GRPO Training

```bash
# Basic training with LoRA (recommended)
python reasoning_grpo_verl.py --model-size 3B --use-lora

# Full model training
python reasoning_grpo_verl.py --model-size 3B

# Custom configuration
python reasoning_grpo_verl.py \
    --model-size 1.5B \
    --use-lora \
    --max-steps 1000 \
    --batch-size 8 \
    --learning-rate 2e-5
```

### 2. Compare with TRL Implementation

```bash
# Run TRL version
python reasoning_grpo.py --model-size 3B --use-lora

# Run VERL version
python reasoning_grpo_verl.py --model-size 3B --use-lora

# Compare outputs in /workspace/models/
```

## VERL-Specific Features

### 1. Unified Reward Function

The VERL implementation uses a single reward function that combines:

- **Format compliance**: Ensures proper `<think></think><answer></answer>` structure
- **Thinking quality**: Rewards detailed reasoning (penalizes short responses)
- **Answer correctness**: Rewards correct answers with partial matching

### 2. VERL GRPO Configuration

```python
# VERL-specific GRPO parameters
actor_rollout={
    'ref': {
        'rollout': {'n': 4},  # Number of samples per prompt
        'actor': {
            'ppo_mini_batch_size': 16,
            'ppo_epochs': 4,
            'clip_ratio': 0.2,
            'use_kl_loss': True,
            'kl_loss_coef': 0.001,
            'kl_loss_type': 'k1',
            'loss_agg_mode': 'token-mean'
        }
    }
},
data={'train_batch_size': batch_size * gradient_accumulation_steps},
algorithm={'adv_estimator': 'grpo'}
```

### 3. Enhanced Logging

- **VERL-specific debug logs**: `debug_logs/verl_grpo_debug_*.txt`
- **Reward breakdown**: Shows format, thinking, and answer rewards separately
- **VERL version tracking**: Displays VERL version in logs

## Model Output Locations

- **VERL Models**: `/workspace/models/{model-name}-LoRA-VERL-GRPO/`
- **TRL Models**: `/workspace/models/{model-name}-LoRA-GRPO/`

## Prerequisites

1. **VERL Installation**: `pip install verl`
2. **Hugging Face Token**: For gated models (Llama, etc.)
3. **GPU Support**: CUDA-compatible GPU recommended

## Troubleshooting

### VERL Import Errors

```bash
# Check VERL installation
python -c "import verl; print(verl.__version__)"

# Reinstall if needed
pip install verl
```

### Configuration Issues

```bash
# Test with minimal configuration (built-in validation will catch issues)
python reasoning_grpo_verl.py --model-size 0.5B --use-lora --max-steps 1
```

### Memory Issues

```bash
# Use smaller model and LoRA
python reasoning_grpo_verl.py --model-size 0.5B --use-lora --batch-size 1
```

## Performance Comparison

| Metric             | TRL GRPO | VERL GRPO                   |
| ------------------ | -------- | --------------------------- |
| **Memory Usage**   | Standard | Optimized                   |
| **Training Speed** | Baseline | Potentially faster          |
| **Convergence**    | Good     | May differ                  |
| **Stability**      | Stable   | VERL-specific optimizations |

## Next Steps

1. **Run both implementations** on the same dataset
2. **Compare performance metrics** (convergence, final rewards)
3. **Benchmark memory usage** and training speed
4. **Evaluate output quality** on reasoning tasks

## Example Training Pipeline

```bash
# 1. Login to Hugging Face
huggingface-cli login

# 2. Run VERL GRPO training (built-in validation)
python reasoning_grpo_verl.py \
    --model-size 3B \
    --use-lora \
    --max-steps 500 \
    --batch-size 4

# 3. Compare with TRL version
python reasoning_grpo.py \
    --model-size 3B \
    --use-lora \
    --max-steps 500 \
    --batch-size 4

# 4. Analyze results in /workspace/models/
```
