"""
VERL-based GRPO Training Script for Reasoning Tasks

This script implements GRPO (Group Relative Policy Optimization) training using VERL
for improving reasoning capabilities on the mini-reasoning-dataset.

VERL (Versatile Reinforcement Learning) is a framework for RLHF training that
provides better scalability and performance compared to traditional TRL
implementations.

Usage:
    # Single GPU training
    python reasoning_grpo_verl.py --model-size 3B --use-lora
    python reasoning_grpo_verl.py --model-size 4B --gradient-accumulation-steps 16
    python reasoning_grpo_verl.py --disable-wandb

Arguments:
    --model-size: Model size to use ("0.5B", "1.5B", "3B", "4B") [default: 4B]
    --use-lora: Enable LoRA for efficient fine-tuning [default: False]
    --disable-wandb: Disable wandb logging [default: False]
    --max-steps: Maximum training steps [default: 500]
    --batch-size: Training batch size [default: 4]
    --learning-rate: Learning rate [default: 1e-5]
    --hf-token: Hugging Face token for accessing gated repositories
    [default: None]

Examples:
    # Basic training
    python reasoning_grpo_verl.py
    python reasoning_grpo_verl.py --use-lora

    # Custom configuration
    python reasoning_grpo_verl.py --model-size 1.5B --max-steps 1000 --batch-size 8

    # With Hugging Face token
    python reasoning_grpo_verl.py --hf-token your_token_here
"""

import re
import random
import os
import argparse
from datetime import datetime
from typing import List, Optional

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# VERL imports
from verl.config import Config
from verl.trainer.grpo.ray_trainer import RayGRPOTrainer
from verl.trainer.grpo.ray_trainer import ResourcePoolManager, Role


class ReasoningRewardManager:
    """VERL-compatible reward manager for reasoning tasks."""

    def __init__(self, tokenizer: AutoTokenizer, num_examine: int = 0):
        """
        Initialize the reward manager.

        Args:
            tokenizer: Tokenizer for the model
            num_examine: Number of examples to examine for debugging
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.step_counter = 0

        # Tag constants
        self.reasoning_start = "<think>"
        self.reasoning_end = "</think>"
        self.answer_start = "<answer>"
        self.answer_end = "</answer>"

        # Setup logging
        self.log_dir = "debug_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"verl_grpo_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

    def compute_reward(
        self, completions: List[str], ground_truth: List[str], **kwargs
    ) -> List[float]:
        """
        Compute reward scores for reasoning completions.

        Args:
            completions: List of model completions
            ground_truth: List of ground truth answers
            **kwargs: Additional arguments

        Returns:
            List of reward scores
        """
        self.step_counter += 1
        scores = []

        # Debug logging for first completion
        if completions and self.num_examine > 0:
            self._log_debug_info(completions[0], ground_truth[0])

        # Score each completion
        for completion, gt in zip(completions, ground_truth):
            # Calculate individual reward components
            format_score = self._compute_format_reward(completion)
            thinking_score = self._compute_thinking_reward(completion)
            answer_score = self._compute_answer_reward(completion, gt)

            # Combine scores
            total_score = format_score + thinking_score + answer_score
            scores.append(total_score)

        return scores

    def _compute_format_reward(self, completion: str) -> float:
        """Compute format compliance reward."""
        # Regex for matching the format: <think>content</think><answer>content</answer>
        match_format = re.compile(
            rf"^[\s]{{0,}}"
            rf"{self.reasoning_start}.*?{self.reasoning_end}"
            rf"{self.answer_start}.*?{self.answer_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )

        penalty = 0

        # Format compliance checking
        if match_format.search(completion) is not None:
            # Format is perfect - no penalty
            return penalty

        # Missing or incorrect tags
        penalty -= 1.0 if completion.count(self.reasoning_start) != 1 else 0
        penalty -= 1.0 if completion.count(self.reasoning_end) != 1 else 0
        penalty -= 1.0 if completion.count(self.answer_start) != 1 else 0
        penalty -= 1.0 if completion.count(self.answer_end) != 1 else 0

        # Content structure penalties
        penalty += self._check_content_structure(completion)

        return penalty

    def _check_content_structure(self, completion: str) -> float:
        """Check content structure and return penalty score."""
        penalty = 0

        # Unwrapped content (content not in tags)
        content_without_tags = re.sub(
            rf"{self.reasoning_start}.*?{self.reasoning_end}",
            "",
            completion,
            flags=re.DOTALL,
        )
        content_without_tags = re.sub(
            rf"{self.answer_start}.*?{self.answer_end}",
            "",
            content_without_tags,
            flags=re.DOTALL,
        )
        content_without_tags = content_without_tags.strip()

        if content_without_tags:
            penalty -= 5.0  # Penalty for unwrapped content

        # Wrong order (answer before thinking)
        think_pos = completion.find(self.reasoning_start)
        answer_pos = completion.find(self.answer_start)

        if think_pos != -1 and answer_pos != -1:
            if answer_pos < think_pos:  # Answer comes before thinking
                penalty -= 1.0

        # Multiple sections (should be exactly one of each)
        think_count = completion.count(self.reasoning_start)
        answer_count = completion.count(self.answer_start)

        if think_count > 1:
            penalty -= 2.0
        if answer_count > 1:
            penalty -= 2.0

        return penalty

    def _compute_thinking_reward(self, completion: str) -> float:
        """Compute thinking quality reward."""
        # Extract thinking content
        think_match = re.search(
            rf"{self.reasoning_start}(.+?){self.reasoning_end}",
            completion,
            flags=re.DOTALL,
        )

        if think_match:
            think_content = think_match.group(1).strip()
        else:
            think_content = completion

        content_length = len(think_content)

        # Gradual penalty for short thinking (under 200 chars)
        if content_length < 200:
            penalty_ratio = (200 - content_length) / 200
            # Gradual penalty from 0 to -10.0
            return -10.0 * penalty_ratio

        return 0

    def _compute_answer_reward(self, completion: str, ground_truth: str) -> float:
        """Compute answer correctness reward."""
        # Extract answer from completion
        answer_match = re.search(
            rf"{self.answer_start}\s*(.+?)\s*{self.answer_end}",
            completion,
            flags=re.DOTALL,
        )

        if answer_match is None:
            # No answer tags found - treat as wrong answer
            return -1.0

        answer = answer_match.group(1).strip()

        # Exact match gets full score
        if answer.lower() == ground_truth.lower():
            return 8.0
        # Partial match if answer contains ground truth
        elif ground_truth.lower() in answer.lower():
            return 3.0
        else:
            return -1.0  # Penalty for wrong answers

    def _log_debug_info(self, completion: str, ground_truth: str) -> None:
        """Log debug information for monitoring training progress."""
        # Always print when there's a full score, occasionally print other cases
        should_print = False
        print_reason = ""

        answer_match = re.search(
            rf"{self.answer_start}\s*(.+?)\s*{self.answer_end}",
            completion,
            flags=re.DOTALL,
        )

        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            if extracted_answer.lower() == ground_truth.lower():
                should_print = True
                print_reason = "🎯 FULL SCORE (8.0) - Exact match!"
            elif random.random() < 0.1:  # 10% chance for other cases
                should_print = True
                if ground_truth.lower() in extracted_answer.lower():
                    print_reason = "✅ PARTIAL SCORE (3.0) - Contains ground truth"
                else:
                    print_reason = "❌ WRONG ANSWER (-1.0) - No match"
        elif random.random() < 0.1:  # 10% chance for no tags case
            should_print = True
            print_reason = "❌ No answer tags found (-1.0 penalty)"

        if should_print:
            self._write_debug_output(completion, ground_truth, print_reason)

    def _write_debug_output(
        self, completion: str, ground_truth: str, print_reason: str
    ) -> None:
        """Write debug output to console and file."""
        # Calculate individual function scores for debugging
        format_reward = self._compute_format_reward(completion)
        think_reward = self._compute_thinking_reward(completion)
        answer_reward = self._compute_answer_reward(completion, ground_truth)
        total_reward = format_reward + think_reward + answer_reward

        # Prepare debug output
        debug_output = self._format_debug_output(
            completion,
            ground_truth,
            print_reason,
            format_reward,
            think_reward,
            answer_reward,
            total_reward,
        )

        # Print to console
        for line in debug_output:
            print(line)

        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(debug_output))
            f.write("\n")

    def _format_debug_output(
        self,
        completion: str,
        ground_truth: str,
        print_reason: str,
        format_reward: float,
        think_reward: float,
        answer_reward: float,
        total_reward: float,
    ) -> List[str]:
        """Format debug output for logging."""
        debug_output = []
        debug_output.append("\n" + "=" * 60)
        debug_output.append(
            f"VERL GRPO SPOT CHECK: PROMPT AND COMPLETIONS "
            f"(Step: {self.step_counter})"
        )
        debug_output.append("=" * 60)
        debug_output.append(f"==Completion:==\n {completion}\n")
        debug_output.append(f"==Ground Truth:==\n {ground_truth}")

        # Extract answer for display
        answer_match = re.search(
            rf"{self.answer_start}\s*(.+?)\s*{self.answer_end}",
            completion,
            flags=re.DOTALL,
        )
        if answer_match:
            extracted_answer = answer_match.group(1).strip()
            debug_output.append(f"==Extracted Answer: '{extracted_answer}'")

        debug_output.append(print_reason)
        debug_output.append("==SCORE BREAKDOWN==")
        debug_output.append(f"  Format reward: {format_reward}")
        debug_output.append(f"  Think reward: {think_reward}")
        debug_output.append(f"  Answer reward: {answer_reward}")
        debug_output.append(f"  TOTAL REWARD: {total_reward}")
        debug_output.append("=" * 60)

        return debug_output


class ReasoningGRPOVERLTrainer:
    """VERL-based GRPO trainer for reasoning tasks."""

    def __init__(
        self,
        model_size: str = "4B",
        use_lora: bool = False,
        wandb_enabled: bool = True,
        max_steps: int = 500,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 16,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the VERL GRPO trainer.

        Args:
            model_size: Size of the model ("0.5B", "1.5B", "3B", "4B")
            use_lora: Whether to use LoRA for efficient fine-tuning
            wandb_enabled: Whether to enable wandb logging
            max_steps: Maximum training steps
            batch_size: Training batch size
            learning_rate: Learning rate for training
            gradient_accumulation_steps: Number of gradient accumulation steps
            hf_token: Hugging Face token for accessing gated repositories
        """
        self.model_size = model_size
        self.use_lora = use_lora
        self.wandb_enabled = wandb_enabled
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.hf_token = hf_token
        self.model_name = self._get_model_name()

        # Setup workspace directories
        self.workspace_dir = os.environ.get(
            "WORKSPACE_DIR", os.path.expanduser("~/workspace")
        )
        if not os.access(self.workspace_dir, os.W_OK):
            self.workspace_dir = os.path.expanduser("~/workspace")

        self.models_dir = os.path.join(self.workspace_dir, "models")
        self.data_dir = os.path.join(self.workspace_dir, "data")
        self.cache_dir = os.path.join(self.workspace_dir, "cache")

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Set environment variables for HuggingFace
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["HF_HUB_CACHE"] = self.models_dir
        os.environ["HF_DATASETS_CACHE"] = self.data_dir

        # Set Hugging Face token if provided
        if self.hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self.hf_token

    def _get_model_name(self) -> str:
        """Get the model name based on size."""
        model_mapping = {
            "4B": "Qwen/Qwen3-4B-Instruct-2507",
            "3B": "meta-llama/Llama-3.2-3B-Instruct",
            "1.5B": "Qwen/Qwen2-1.5B-Instruct",
            "0.5B": "Qwen/Qwen2-0.5B-Instruct",
        }

        if self.model_size not in model_mapping:
            raise ValueError(f"Invalid model size: {self.model_size}")

        return model_mapping[self.model_size]

    def _create_reasoning_prompt(self, question: str) -> str:
        """Create the reasoning prompt template."""
        return f"""
        The following question requires reasoning.
        In addition to provide your answer, you should also provide your
        DETAILED thought process about how you arrive at your answer.
        Put your thought process between <think></think> tags and then put
        your answer between <answer></answer> tags.

        The question is:
        {question}
        """

    def prepare_dataset_for_verl(self) -> str:
        """
        Prepare dataset in VERL-compatible format and save as parquet.

        Returns:
            Path to the prepared parquet file
        """
        # Load dataset from HuggingFace
        dataset = load_dataset("tech-tao/mini-reasoning-dataset", split="train")

        # Transform dataset with reasoning prompt template
        processed_data = []
        for item in dataset:
            processed_item = {
                "prompt": self._create_reasoning_prompt(item["prompt"]),
                "ground_truth": item["completion"],
            }
            processed_data.append(processed_item)

        # Convert to DataFrame and save as parquet
        df = pd.DataFrame(processed_data)
        parquet_path = os.path.join(self.data_dir, "reasoning_dataset.parquet")
        df.to_parquet(parquet_path, index=False)

        print(f"📊 Dataset prepared and saved to: {parquet_path}")
        print(f"   Total samples: {len(processed_data)}")

        return parquet_path

    def create_verl_config(self) -> Config:
        """Create VERL configuration for GRPO training."""
        # Create output directory
        model_name_short = self.model_name.split("/")[-1]
        lora_suffix = "LoRA" if self.use_lora else "Full"
        model_output_name = f"{model_name_short}-{lora_suffix}-VERL-GRPO"
        output_dir = os.path.join(self.models_dir, model_output_name)

        # Prepare dataset
        dataset_path = self.prepare_dataset_for_verl()

        # Create VERL configuration
        config = {
            "trainer": {
                "type": "grpo",
                "total_epochs": 1,
                "save_interval": 100,
                "logging_interval": 1,
                "eval_interval": 50,
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp",
                    "model": {
                        "type": "causal_lm",
                        "model_name": self.model_name,
                        "use_lora": self.use_lora,
                        "lora_config": (
                            {
                                "r": 16,
                                "lora_alpha": 32,
                                "target_modules": [
                                    "q_proj",
                                    "v_proj",
                                    "k_proj",
                                    "o_proj",
                                ],
                                "lora_dropout": 0.01,
                                "bias": "none",
                            }
                            if self.use_lora
                            else None
                        ),
                    },
                    "optimizer": {
                        "lr": self.learning_rate,
                        "eps": 1e-5,
                        "weight_decay": 0.01,
                    },
                    "lr_scheduler": {
                        "type": "cosine",
                        "warmup_ratio": 0.1,
                    },
                },
                "rollout": {
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "max_new_tokens": 512,
                    "num_generations": 8,
                },
            },
            "critic": {
                "strategy": "fsdp",
                "model": {
                    "type": "causal_lm",
                    "model_name": self.model_name,
                    "use_lora": self.use_lora,
                    "lora_config": (
                        {
                            "r": 16,
                            "lora_alpha": 32,
                            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                            "lora_dropout": 0.01,
                            "bias": "none",
                        }
                        if self.use_lora
                        else None
                    ),
                },
                "optimizer": {
                    "lr": self.learning_rate,
                    "eps": 1e-5,
                    "weight_decay": 0.01,
                },
                "lr_scheduler": {
                    "type": "cosine",
                    "warmup_ratio": 0.1,
                },
            },
            "data": {
                "train_path": dataset_path,
                "val_path": dataset_path,  # Use same dataset for validation
                "max_prompt_length": 768,
                "max_response_length": 512,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
            },
            "grpo": {
                "cliprange": 0.2,
                "cliprange_value": 0.2,
                "gamma": 0.99,
                "lam": 0.95,
                "vf_coef": 0.1,
                "ent_coef": 0.01,
                "max_grad_norm": 1.0,
            },
            "output_dir": output_dir,
            "seed": 42,
            "fp16": True,
            "bf16": False,
        }

        # Add wandb configuration if enabled
        if self.wandb_enabled:
            config["wandb"] = {
                "project": "verl-reasoning-grpo",
                "name": f"{model_name_short}-{lora_suffix}-VERL-GRPO",
                "tags": ["reasoning", "grpo", "verl"],
            }

        return Config(config)

    def setup_workers_and_resources(self, config: Config):
        """Setup VERL workers and resource management."""
        # Import worker classes based on strategy
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup

            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(
                f"Strategy {config.actor_rollout_ref.actor.strategy} not supported"
            )

        # Define role-worker mapping
        role_worker_mapping = {
            Role.ActorRollout: ActorRolloutRefWorker,
            Role.Critic: CriticWorker,
            Role.RefPolicy: ActorRolloutRefWorker,
        }

        # Define resource pools
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        return role_worker_mapping, resource_pool_manager, ray_worker_group_cls

    def print_directory_info(self) -> None:
        """Print information about workspace directories."""
        print("📁 VERL Workspace Configuration:")
        print(f"   Workspace Directory: {self.workspace_dir}")
        print(f"   Models Directory: {self.models_dir}")
        print(f"   Data Directory: {self.data_dir}")
        print(f"   Cache Directory: {self.cache_dir}")
        print(f"   Model: {self.model_name}")
        print("-" * 50)

    def train(self) -> None:
        """Execute the VERL GRPO training process."""
        # Print directory information
        self.print_directory_info()

        # Create VERL configuration
        config = self.create_verl_config()

        # Setup workers and resources
        (role_worker_mapping, resource_pool_manager, ray_worker_group_cls) = (
            self.setup_workers_and_resources(config)
        )

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create reward manager
        reward_fn = ReasoningRewardManager(tokenizer=tokenizer, num_examine=1)
        val_reward_fn = ReasoningRewardManager(tokenizer=tokenizer, num_examine=0)

        # Initialize VERL GRPO trainer
        trainer = RayGRPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        # Initialize workers and start training
        print("🚀 Initializing VERL workers...")
        trainer.init_workers()

        print("🎯 Starting VERL GRPO training...")
        trainer.fit()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VERL GRPO Training Script for Reasoning Tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["0.5B", "1.5B", "3B", "4B"],
        default="4B",
        help="Model size to use for training",
    )

    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA for efficient fine-tuning",
    )

    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum training steps",
    )

    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for training",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Number of gradient accumulation steps",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for accessing gated repositories",
    )

    return parser.parse_args()


def main():
    """Main entry point for the VERL GRPO training script."""
    # Parse command-line arguments
    args = parse_arguments()

    print("🚀 Starting VERL GRPO training with:")
    print(f"   Model: {args.model_size}")
    lora_status = "Enabled" if args.use_lora else "Disabled"
    wandb_status = "Disabled" if args.disable_wandb else "Enabled"
    hf_token_status = "Provided" if args.hf_token else "Not provided"
    print(f"   LoRA: {lora_status}")
    print(f"   Wandb: {wandb_status}")
    print(f"   HF Token: {hf_token_status}")
    print(f"   Max Steps: {args.max_steps}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print("-" * 50)

    # Create and run trainer
    trainer = ReasoningGRPOVERLTrainer(
        model_size=args.model_size,
        use_lora=args.use_lora,
        wandb_enabled=not args.disable_wandb,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        hf_token=args.hf_token,
    )

    trainer.train()


if __name__ == "__main__":
    main()
