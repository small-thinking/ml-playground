# VERL-based GRPO Training for Reasoning Tasks

This directory contains a VERL (Versatile Reinforcement Learning) implementation of GRPO (Group Relative Policy Optimization) for training language models on reasoning tasks.

## Overview

The VERL implementation provides several advantages over the traditional TRL-based approach:

- **Better Scalability**: VERL is designed for distributed training and can handle larger models more efficiently
- **Improved Performance**: Optimized for multi-GPU and multi-node training scenarios
- **Flexible Architecture**: Supports different backends (FSDP, Megatron-LM) and worker configurations
- **Advanced Features**: Built-in support for fault tolerance, checkpointing, and resource management

## Files

- `reasoning_grpo_verl_clean.py`: Main VERL GRPO training script
- `reasoning_grpo.py`: Original TRL-based implementation for comparison
- `requirements_verl.txt`: VERL-specific dependencies
- `README_VERL_GRPO.md`: This documentation

## Key Differences from TRL Implementation

### 1. Reward Function Architecture

**TRL Version:**

```python
# Multiple separate reward functions
reward_funcs = [
    self.match_format_func,
    self.penalize_short_think_func,
    self.check_answer_func,
]
```

**VERL Version:**

```python
# Single unified reward manager
class ReasoningRewardManager:
    def compute_reward(self, completions, ground_truth, **kwargs):
        # Combines all reward components internally
        format_score = self._compute_format_reward(completion)
        thinking_score = self._compute_thinking_reward(completion)
        answer_score = self._compute_answer_reward(completion, gt)
        return format_score + thinking_score + answer_score
```

### 2. Configuration Management

**TRL Version:**

```python
# Simple configuration object
config = GRPOConfig(
    output_dir=output_dir,
    learning_rate=self.learning_rate,
    # ... other parameters
)
```

**VERL Version:**

```python
# Comprehensive configuration with nested structure
config = {
    "trainer": {"type": "grpo", "n_gpus_per_node": 1, ...},
    "actor_rollout_ref": {
        "actor": {"strategy": "fsdp", "model": {...}, ...},
        "rollout": {"temperature": 1.0, "num_generations": 8, ...}
    },
    "critic": {"strategy": "fsdp", "model": {...}, ...},
    "data": {"train_path": dataset_path, "batch_size": 4, ...},
    "grpo": {"cliprange": 0.2, "gamma": 0.99, ...}
}
```

### 3. Worker and Resource Management

**TRL Version:**

```python
# Simple trainer initialization
trainer = GRPOTrainer(
    model=self.model_name,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=self.dataset,
    peft_config=lora_config,
)
```

**VERL Version:**

```python
# Complex worker setup with resource pools
role_worker_mapping = {
    Role.ActorRollout: ActorRolloutRefWorker,
    Role.Critic: CriticWorker,
    Role.RefPolicy: ActorRolloutRefWorker
}

resource_pool_spec = {
    'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes
}

trainer = RayGRPOTrainer(
    config=config,
    tokenizer=tokenizer,
    role_worker_mapping=role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    ray_worker_group_cls=ray_worker_group_cls,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn,
)
```

## Installation

1. Install VERL and dependencies:

```bash
pip install -r requirements_verl.txt
```

2. Install VERL framework:

```bash
pip install verl
```

## Usage

### Basic Training

```bash
python reasoning_grpo_verl_clean.py
```

### With LoRA

```bash
python reasoning_grpo_verl_clean.py --use-lora
```

### Custom Configuration

```bash
python reasoning_grpo_verl_clean.py \
    --model-size 1.5B \
    --max-steps 1000 \
    --batch-size 8 \
    --learning-rate 2e-5
```

### With Hugging Face Token

```bash
python reasoning_grpo_verl_clean.py --hf-token your_token_here
```

## Configuration Options

| Parameter                       | Description                             | Default |
| ------------------------------- | --------------------------------------- | ------- |
| `--model-size`                  | Model size ("0.5B", "1.5B", "3B", "4B") | "4B"    |
| `--use-lora`                    | Enable LoRA fine-tuning                 | False   |
| `--disable-wandb`               | Disable wandb logging                   | False   |
| `--max-steps`                   | Maximum training steps                  | 500     |
| `--batch-size`                  | Training batch size                     | 4       |
| `--learning-rate`               | Learning rate                           | 1e-5    |
| `--gradient-accumulation-steps` | Gradient accumulation steps             | 16      |
| `--hf-token`                    | Hugging Face token                      | None    |

## Architecture Components

### 1. ReasoningRewardManager

- **Purpose**: Computes reward scores for reasoning completions
- **Components**:
  - Format compliance checking
  - Thinking quality assessment
  - Answer correctness evaluation
- **Integration**: Compatible with VERL's reward system

### 2. ReasoningGRPOVERLTrainer

- **Purpose**: Main trainer class for VERL GRPO training
- **Features**:
  - Dataset preparation and preprocessing
  - VERL configuration management
  - Worker and resource setup
  - Training orchestration

### 3. VERL Configuration

- **Trainer**: Defines training parameters and resource allocation
- **Actor/Rollout**: Model configuration and generation parameters
- **Critic**: Value function model setup
- **Data**: Dataset paths and batch configuration
- **GRPO**: Algorithm-specific hyperparameters

## Performance Considerations

### Memory Optimization

- **FSDP Strategy**: Enables efficient memory usage across GPUs
- **LoRA Support**: Reduces memory footprint for fine-tuning
- **Gradient Accumulation**: Allows larger effective batch sizes

### Scalability

- **Multi-GPU Support**: Built-in support for distributed training
- **Resource Pools**: Flexible GPU allocation and management
- **Ray Integration**: Enables multi-node training scenarios

## Debugging and Monitoring

### Debug Logging

- **Location**: `debug_logs/verl_grpo_debug_YYYYMMDD_HHMMSS.txt`
- **Content**: Detailed reward breakdowns and sample completions
- **Frequency**: Configurable via `num_examine` parameter

### Wandb Integration

- **Project**: `verl-reasoning-grpo`
- **Metrics**: Training loss, reward scores, and model performance
- **Tags**: `["reasoning", "grpo", "verl"]`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure VERL is properly installed

   ```bash
   pip install verl
   ```

2. **CUDA Memory Issues**: Reduce batch size or enable LoRA

   ```bash
   python reasoning_grpo_verl_clean.py --batch-size 2 --use-lora
   ```

3. **Dataset Loading Issues**: Check internet connection and HF token
   ```bash
   python reasoning_grpo_verl_clean.py --hf-token your_token_here
   ```

### Performance Tuning

1. **Increase Batch Size**: For better GPU utilization
2. **Enable LoRA**: For memory-efficient fine-tuning
3. **Adjust Learning Rate**: Based on model size and dataset
4. **Optimize Gradient Accumulation**: Balance memory and training speed

## Comparison with TRL Implementation

| Aspect                  | TRL Version        | VERL Version           |
| ----------------------- | ------------------ | ---------------------- |
| **Scalability**         | Single GPU focused | Multi-GPU/Multi-node   |
| **Memory Usage**        | Higher             | Optimized with FSDP    |
| **Configuration**       | Simple             | Comprehensive          |
| **Worker Management**   | Automatic          | Explicit control       |
| **Resource Allocation** | Basic              | Advanced pooling       |
| **Fault Tolerance**     | Limited            | Built-in support       |
| **Performance**         | Good               | Better for large scale |

## Future Enhancements

1. **Multi-Modal Support**: Extend to image reasoning tasks
2. **Advanced Algorithms**: Implement other RL algorithms (PPO, DPO)
3. **Custom Reward Models**: Support for learned reward functions
4. **Hyperparameter Optimization**: Automated tuning capabilities
5. **Model Serving**: Integration with inference servers

## References

- [VERL Documentation](https://verl.readthedocs.io/)
- [GRPO Paper](https://arxiv.org/abs/2406.05930)
- [VERL GitHub Repository](https://github.com/volcengine/verl)
- [Original TRL Implementation](./reasoning_grpo.py)
