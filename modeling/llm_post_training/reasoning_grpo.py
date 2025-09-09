"""
GRPO Training Script for Reasoning Tasks

This script implements GRPO (Generative Reward-Powered Optimization) training
for improving reasoning capabilities on the mini-reasoning-dataset.

Usage:
    # Single GPU training
    python reasoning_grpo.py --model-size 3B --use-lora
    python reasoning_grpo.py --model-size 8B --gradient-accumulation-steps 16
    python reasoning_grpo.py --disable-wandb
    
    # Multi-GPU training with torchrun (recommended for multiple GPUs)
    torchrun --nproc_per_node=4 reasoning_grpo.py --model-size 8B
    torchrun --nproc_per_node=2 reasoning_grpo.py --use-lora --batch-size 8
    
    # Multi-GPU training with DataParallel (fallback)
    python reasoning_grpo.py --model-size 8B  # Will use device_map="auto" if multiple GPUs detected

Arguments:
    --model-size: Model size to use ("0.5B", "1.5B", "3B", "8B") [default: 8B]
    --use-lora: Enable LoRA for efficient fine-tuning [default: False]
    --disable-wandb: Disable wandb logging [default: False]
    --max-steps: Maximum training steps [default: 500]
    --batch-size: Training batch size [default: 4]
    --learning-rate: Learning rate [default: 1e-5]
    --hf-token: Hugging Face token for accessing gated repositories
    [default: None]

Note:
    Multi-GPU training is supported in two modes:
    1. Distributed Data Parallel (DDP): Use torchrun for best performance
    2. DataParallel: Automatic fallback when running with python directly

Examples:
    # Single GPU training
    python reasoning_grpo.py
    python reasoning_grpo.py --use-lora
    
    # Multi-GPU with DDP (recommended)
    torchrun --nproc_per_node=4 reasoning_grpo.py
    torchrun --nproc_per_node=2 reasoning_grpo.py --use-lora
    
    # Multi-GPU with DataParallel (fallback)
    python reasoning_grpo.py  # Will automatically use multiple GPUs if available
    
    # Custom configuration with multi-GPU
    torchrun --nproc_per_node=2 reasoning_grpo.py --model-size 1.5B --max-steps 1000 --batch-size 8
    
    # With Hugging Face token
    torchrun --nproc_per_node=4 reasoning_grpo.py --hf-token your_token_here
"""

import re
import random
import os
import argparse
from datetime import datetime
from typing import List, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig


class ReasoningGRPOTrainer:
    """Main trainer class for GRPO reasoning training."""

    def __init__(
        self,
        model_size: str = "8B",
        use_lora: bool = False,
        wandb_enabled: bool = True,
        max_steps: int = 500,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 16,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize the trainer with model configuration.

        Args:
            model_size: Size of the model ("0.5B", "1.5B", "3B", "8B")
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
        self.dataset = None
        self.index = {}
        self.step_counter = 0

        # Auto-detect and setup multi-GPU
        self.num_gpus = self._detect_gpus()
        self.use_multi_gpu = self.num_gpus > 1
        self.is_distributed = self._init_distributed()
        self._setup_multi_gpu()

        # Setup workspace directories
        self.workspace_dir = os.environ.get("WORKSPACE_DIR", "/workspace")
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

        # Setup logging
        self.log_dir = "debug_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"grpo_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        # Tag constants
        self.reasoning_start = "<think>"
        self.reasoning_end = "</think>"
        self.answer_start = "<answer>"
        self.answer_end = "</answer>"

    def _get_model_name(self) -> str:
        """Get the model name based on size."""
        model_mapping = {
            "8B": "meta-llama/Llama-3.1-8B-Instruct",
            "3B": "meta-llama/Llama-3.2-3B-Instruct",
            "1.5B": "Qwen/Qwen2-1.5B-Instruct",
            "0.5B": "Qwen/Qwen2-0.5B-Instruct",
        }

        if self.model_size not in model_mapping:
            raise ValueError(f"Invalid model size: {self.model_size}")

        return model_mapping[self.model_size]

    def _detect_gpus(self) -> int:
        """Detect the number of available GPUs."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()

    def _init_distributed(self) -> bool:
        """Initialize distributed training if multiple GPUs are available."""
        if not self.use_multi_gpu:
            return False
        
        # Check if we're already running with torchrun/distributed launcher
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Already launched with torchrun, just initialize
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            return True
        
        # Auto-launch with torchrun if multiple GPUs detected and not already distributed
        return False

    def _setup_multi_gpu(self) -> None:
        """Setup multi-GPU configuration automatically."""
        if self.is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            print(f"🚀 Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
        elif self.use_multi_gpu:
            print(f"🚀 Multi-GPU training available with {self.num_gpus} GPUs")
            print("   → Use 'torchrun --nproc_per_node={} python reasoning_grpo.py [args]' for distributed training".format(self.num_gpus))
        else:
            print(f"💻 Single GPU training with {self.num_gpus} GPU(s)")

    def load_and_prepare_dataset(self) -> None:
        """Load and prepare the mini-reasoning-dataset."""
        # Load dataset from HuggingFace
        # Dataset: https://huggingface.co/datasets/tech-tao/
        #          mini-reasoning-dataset
        self.dataset = load_dataset("tech-tao/mini-reasoning-dataset", split="train")

        # Transform dataset with reasoning prompt template
        self.dataset = self.dataset.map(
            lambda x: {
                "prompt": self._create_reasoning_prompt(x["prompt"]),
                "ground_truth": x["completion"],
            }
        )

        # Build index for easy lookup
        self._build_dataset_index()

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

    def _build_dataset_index(self) -> None:
        """Build an index keyed by ground truth for easy lookup."""
        for i, row in enumerate(self.dataset):
            self.index[row["ground_truth"]] = row["prompt"]

    def match_format_func(self, completions: List[str], **kwargs) -> List[float]:
        """
        Format penalty function: perfect format gets 0, violations get
        penalties.

        Args:
            completions: List of model completions to evaluate

        Returns:
            List of format scores (penalties)
        """
        scores = []
        # Regex for matching the format:
        # <think>content</think><answer>content</answer>
        match_format = re.compile(
            rf"^[\s]{{0,}}"
            rf"{self.reasoning_start}.*?{self.reasoning_end}"
            rf"{self.answer_start}.*?{self.answer_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )

        for completion in completions:
            penalty = 0

            # Format compliance checking
            if match_format.search(completion) is not None:
                # Format is perfect - no penalty
                scores.append(penalty)
                continue

            # Missing or incorrect tags
            penalty -= 1.0 if completion.count(self.reasoning_start) != 1 else 0
            penalty -= 1.0 if completion.count(self.reasoning_end) != 1 else 0
            penalty -= 1.0 if completion.count(self.answer_start) != 1 else 0
            penalty -= 1.0 if completion.count(self.answer_end) != 1 else 0

            # Content structure penalties
            penalty += self._check_content_structure(completion)

            scores.append(penalty)

        return scores

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

    def penalize_short_think_func(
        self, completions: List[str], **kwargs
    ) -> List[float]:
        """
        Penalize thinking sections that are too short.

        Args:
            completions: List of model completions to evaluate

        Returns:
            List of thinking quality scores
        """
        scores = []
        for completion in completions:
            score = 0

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
                score -= 10.0 * penalty_ratio

            scores.append(score)

        return scores

    def check_answer_func(
        self, completions: List[str], ground_truth: List[str], **kwargs
    ) -> List[float]:
        """
        Reward if the answer is correct with partial matching.

        Args:
            completions: List of model completions to evaluate
            ground_truth: List of ground truth answers

        Returns:
            List of answer correctness scores
        """
        self.step_counter += 1
        scores = []

        # Debug logging for first completion
        if completions:
            self._log_debug_info(completions[0], ground_truth[0])

        # Score each completion
        for completion, gt in zip(completions, ground_truth):
            score = 0

            # Extract answer from completion
            answer_match = re.search(
                rf"{self.answer_start}\s*(.+?)\s*{self.answer_end}",
                completion,
                flags=re.DOTALL,
            )

            if answer_match is None:
                # No answer tags found - treat as wrong answer
                score -= 1.0
            else:
                answer = answer_match.group(1).strip()

                # Exact match gets full score
                if answer.lower() == gt.lower():
                    score += 8.0
                # Partial match if answer contains ground truth
                elif gt.lower() in answer.lower():
                    score += 3.0
                else:
                    score -= 1.0  # Penalty for wrong answers

            scores.append(score)

        return scores

    def _log_debug_info(self, completion: str, ground_truth: str) -> None:
        """Log debug information for monitoring training progress."""
        # Always print when there's a full score, occasionally print other
        # cases
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
                    print_reason = "✅ PARTIAL SCORE (3.0) - " "Contains ground truth"
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
        format_reward = self.match_format_func([completion])[0]
        think_reward = self.penalize_short_think_func([completion])[0]

        # Calculate answer score manually
        answer_reward = self._calculate_answer_reward(completion, ground_truth)
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

    def _calculate_answer_reward(self, completion: str, ground_truth: str) -> float:
        """Calculate answer reward score."""
        answer_match = re.search(
            rf"{self.answer_start}\s*(.+?)\s*{self.answer_end}",
            completion,
            flags=re.DOTALL,
        )

        if answer_match is None:
            return -1.0

        answer = answer_match.group(1).strip()
        if answer.lower() == ground_truth.lower():
            return 8.0
        elif ground_truth.lower() in answer.lower():
            return 3.0
        else:
            return -1.0

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
            f"SPOT CHECK: PROMPT AND COMPLETIONS " f"(Step: {self.step_counter})"
        )
        debug_output.append("=" * 60)
        debug_output.append(f"==Prompt:==\n {self.index[ground_truth]}\n")
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

    def get_lora_config(self) -> Optional[LoraConfig]:
        """Get LoRA configuration if enabled."""
        if not self.use_lora:
            return None

        return LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha parameter
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
            ],  # Target attention modules
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def get_training_config(self) -> GRPOConfig:
        """Get GRPO training configuration."""
        # Create output directory in models folder
        model_name_short = self.model_name.split("/")[-1]
        lora_suffix = "LoRA" if self.use_lora else "Full"
        multi_gpu_suffix = f"-{self.num_gpus}GPU" if self.use_multi_gpu else ""
        model_output_name = (
            f"{model_name_short}-{lora_suffix}-GRPO" f"{multi_gpu_suffix}"
        )
        output_dir = os.path.join(self.models_dir, model_output_name)

        config_params = {
            "output_dir": output_dir,
            "learning_rate": self.learning_rate,
            "temperature": 1.0,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "logging_steps": 1,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_generations": 8,
            "max_prompt_length": 768,
            "max_steps": self.max_steps,
            "fp16": True,
            "fp16_full_eval": False,
            "fp16_opt_level": "O1",
        }

        # Multi-GPU configuration
        if self.is_distributed:
            # Distributed training configuration
            config_params.update(
                {
                    "ddp_find_unused_parameters": False,
                    "dataloader_pin_memory": True,
                    "dataloader_num_workers": 4,
                    "remove_unused_columns": False,
                }
            )
        elif self.use_multi_gpu:
            # DataParallel configuration for single-node multi-GPU
            config_params.update(
                {
                    "dataloader_pin_memory": True,
                    "dataloader_num_workers": min(4, self.num_gpus * 2),
                }
            )

        # Add wandb configuration if enabled
        if self.wandb_enabled:
            config_params.update(
                {
                    "report_to": "wandb",
                    "run_name": f"{self.model_name}-{lora_suffix}-GRPO",
                }
            )
        else:
            config_params["report_to"] = "none"

        return GRPOConfig(**config_params)

    def print_directory_info(self) -> None:
        """Print information about workspace directories."""
        # Only print from rank 0 in distributed training
        if self.is_distributed and dist.get_rank() != 0:
            return
            
        print("📁 Workspace Configuration:")
        print(f"   Workspace Directory: {self.workspace_dir}")
        print(f"   Models Directory: {self.models_dir}")
        print(f"   Data Directory: {self.data_dir}")
        print(f"   Cache Directory: {self.cache_dir}")
        print(f"   Model: {self.model_name}")
        
        if self.is_distributed:
            print(f"   Multi-GPU: Distributed ({dist.get_world_size()} processes)")
        elif self.use_multi_gpu:
            print(f"   Multi-GPU: Available ({self.num_gpus} GPUs)")
        else:
            print(f"   Multi-GPU: Disabled")
        print(f"   Available GPUs: {self.num_gpus}")
        print("-" * 50)

    def train(self) -> None:
        """Execute the training process."""
        # Print directory information
        self.print_directory_info()

        # Load and prepare dataset
        self.load_and_prepare_dataset()

        # Get configurations
        lora_config = self.get_lora_config()
        training_args = self.get_training_config()

        # Initialize trainer
        # Note: Token is handled via environment variable HUGGINGFACE_HUB_TOKEN
        # which is set in __init__ when hf_token is provided
        trainer_kwargs = {
            "model": self.model_name,
            "reward_funcs": [
                self.match_format_func,
                self.penalize_short_think_func,
                self.check_answer_func,
            ],
            "args": training_args,
            "train_dataset": self.dataset,
            "peft_config": lora_config,
        }
        
        # Add device_map for non-distributed multi-GPU setups
        if self.use_multi_gpu and not self.is_distributed:
            trainer_kwargs["model_kwargs"] = {"device_map": "auto"}
        
        trainer = GRPOTrainer(**trainer_kwargs)

        # Start training
        trainer.train()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO Training Script for Reasoning Tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["0.5B", "1.5B", "3B", "8B"],
        default="8B",
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
        "--max-steps", type=int, default=500, help="Maximum training steps"
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
    """Main entry point for the training script."""
    # Parse command-line arguments
    args = parse_arguments()

    # Only print from rank 0 in distributed training
    if not (os.environ.get("RANK") and int(os.environ.get("RANK", 0)) != 0):
        print("🚀 Starting GRPO training with:")
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
        print(f"   Gradient Accumulation Steps: " f"{args.gradient_accumulation_steps}")
        
        if "RANK" in os.environ:
            world_size = os.environ.get("WORLD_SIZE", "1")
            print(f"   Distributed: {world_size} processes")
        
        print("-" * 50)

    # Create and run trainer
    trainer = ReasoningGRPOTrainer(
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
