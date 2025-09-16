"""
VERL-based GRPO Training Script for Reasoning Tasks

This script implements GRPO (Generative Reward-Powered Optimization) training
using VERL library for improving reasoning capabilities on the mini-reasoning-dataset.

Usage:
    # Single GPU training with VERL
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
    # Basic VERL GRPO training
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

from datasets import load_dataset
from peft import LoraConfig

# VERL imports - try different import paths for different VERL versions
VERL_AVAILABLE = False
VERL_TRAINER_AVAILABLE = False

try:
    # Try the main VERL import first
    import verl

    VERL_AVAILABLE = True
    print(f"✅ VERL {verl.__version__} imported successfully")

    # Try to import trainer components
    try:
        from verl.trainer.config.hybrid_entrypoint import HybridEngineEntrypointConfig
        from verl.trainer.config.algorithm import AlgoConfig
        from verl.trainer.config.config import CriticConfig
        from omegaconf import DictConfig

        VERL_TRAINER_AVAILABLE = True
        print("✅ VERL trainer config components imported")
    except ImportError as e:
        print(f"⚠ VERL trainer config import failed: {e}")

    # Try to import trainer class
    try:
        from verl.trainer.trainer import Trainer

        print("✅ VERL Trainer class imported")
    except ImportError as e:
        print(f"⚠ VERL Trainer import failed: {e}")
        # Try alternative import paths
        try:
            from verl.trainer import Trainer

            print("✅ VERL Trainer imported from alternative path")
        except ImportError:
            print("⚠ VERL Trainer not available in any path")

except ImportError as e:
    print(f"❌ VERL import failed: {e}")
    print("Please ensure VERL is properly installed: pip install verl")


class ReasoningGRPOTrainerVERL:
    """VERL-based trainer class for GRPO reasoning training."""

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
        Initialize the VERL trainer with model configuration.

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
        if not VERL_AVAILABLE:
            raise ImportError(
                "VERL is not available. Please install it with: pip install verl"
            )

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

        # Setup workspace directories - prioritize remote VM paths
        # Check common remote VM workspace locations
        possible_workspace_dirs = [
            os.environ.get("WORKSPACE_DIR"),
            "/root/workspace",  # Common remote VM path
            "/workspace",  # Alternative remote VM path
            os.path.expanduser("~/verl_workspace"),  # Local fallback
            os.path.expanduser("~/workspace"),  # Local fallback
        ]

        self.workspace_dir = None
        for workspace_dir in possible_workspace_dirs:
            if workspace_dir and os.path.exists(workspace_dir):
                try:
                    # Test if we can write to this directory
                    test_file = os.path.join(workspace_dir, ".test_write")
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)
                    self.workspace_dir = workspace_dir
                    break
                except (OSError, PermissionError):
                    continue

        # If no writable directory found, use home directory
        if not self.workspace_dir:
            self.workspace_dir = os.path.expanduser("~/verl_workspace")
            print(f"⚠️  Using fallback workspace directory: {self.workspace_dir}")
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
            f"verl_grpo_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        # Tag constants
        self.reasoning_start = "<think>"
        self.reasoning_end = "</think>"
        self.answer_start = "<answer>"
        self.answer_end = "</answer>"

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

    def load_and_prepare_dataset(self) -> None:
        """Load and prepare the mini-reasoning-dataset."""
        # Load dataset from HuggingFace
        # Dataset: https://huggingface.co/datasets/tech-tao/mini-reasoning-dataset
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

    def reward_function(
        self, completions: List[str], ground_truths: List[str], **kwargs
    ) -> List[float]:
        """
        VERL reward function that combines format, thinking quality, and answer correctness.

        Args:
            completions: List of model completions to evaluate
            ground_truths: List of ground truth answers

        Returns:
            List of reward scores
        """
        self.step_counter += 1
        scores = []

        # Debug logging for first completion
        if completions:
            self._log_debug_info(completions[0], ground_truths[0])

        # Score each completion
        for completion, gt in zip(completions, ground_truths):
            # Format reward
            format_reward = self._calculate_format_reward(completion)

            # Thinking quality reward
            think_reward = self._calculate_thinking_reward(completion)

            # Answer correctness reward
            answer_reward = self._calculate_answer_reward(completion, gt)

            # Total reward
            total_reward = format_reward + think_reward + answer_reward
            scores.append(total_reward)

        return scores

    def _calculate_format_reward(self, completion: str) -> float:
        """Calculate format compliance reward."""
        # Regex for matching the format: <think>content</think><answer>content</answer>
        match_format = re.compile(
            rf"^[\s]{{0,}}"
            rf"{self.reasoning_start}.*?{self.reasoning_end}"
            rf"{self.answer_start}.*?{self.answer_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )

        if match_format.search(completion) is not None:
            return 0.0  # Perfect format - no penalty

        # Calculate penalties for format violations
        penalty = 0
        penalty -= 1.0 if completion.count(self.reasoning_start) != 1 else 0
        penalty -= 1.0 if completion.count(self.reasoning_end) != 1 else 0
        penalty -= 1.0 if completion.count(self.answer_start) != 1 else 0
        penalty -= 1.0 if completion.count(self.answer_end) != 1 else 0

        # Content structure penalties
        penalty += self._check_content_structure(completion)
        return penalty

    def _calculate_thinking_reward(self, completion: str) -> float:
        """Calculate thinking quality reward."""
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
            return -10.0 * penalty_ratio

        return 0.0

    def _calculate_answer_reward(self, completion: str, ground_truth: str) -> float:
        """Calculate answer correctness reward."""
        # Extract answer from completion
        answer_match = re.search(
            rf"{self.answer_start}\s*(.+?)\s*{self.answer_end}",
            completion,
            flags=re.DOTALL,
        )

        if answer_match is None:
            return -1.0  # No answer tags found

        answer = answer_match.group(1).strip()

        # Exact match gets full score
        if answer.lower() == ground_truth.lower():
            return 8.0
        # Partial match if answer contains ground truth
        elif ground_truth.lower() in answer.lower():
            return 3.0
        else:
            return -1.0  # Penalty for wrong answers

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
        format_reward = self._calculate_format_reward(completion)
        think_reward = self._calculate_thinking_reward(completion)
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
        debug_output.append("==VERL REWARD BREAKDOWN==")
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

    def get_verl_config(self):
        """Get VERL GRPO training configuration."""
        # Create output directory in models folder
        model_name_short = self.model_name.split("/")[-1]
        lora_suffix = "LoRA" if self.use_lora else "Full"
        model_output_name = f"{model_name_short}-{lora_suffix}-VERL-GRPO"
        output_dir = os.path.join(self.models_dir, model_output_name)

        # Algorithm configuration with GRPO
        algorithm_config = AlgoConfig(
            adv_estimator="grpo",
            norm_adv_by_std_in_grpo=True,
            gamma=1.0,
            lam=1.0,
            use_kl_in_reward=False,
            kl_penalty="kl",
        )

        # Critic configuration
        critic_config = CriticConfig(
            rollout_n=4,  # Number of samples per prompt
            strategy="fsdp",
            ppo_mini_batch_size=16,
            ppo_epochs=4,
            cliprange_value=0.2,
            loss_agg_mode="token-mean",
            optim={
                "lr": self.learning_rate,
                "weight_decay": 0.01,
            },
            model={
                "path": self.model_name,
                "tokenizer_path": self.model_name,
            },
        )

        # Actor rollout configuration
        actor_rollout_ref = DictConfig(
            {
                "ref": {
                    "rollout": {"n": 4},  # Number of samples per prompt
                    "actor": {
                        "ppo_mini_batch_size": 16,
                        "ppo_epochs": 4,
                        "clip_ratio": 0.2,
                        "use_kl_loss": True,
                        "kl_loss_coef": 0.001,
                        "kl_loss_type": "k1",
                        "loss_agg_mode": "token-mean",
                    },
                }
            }
        )

        # Data configuration
        data_config = DictConfig(
            {
                "train_batch_size": self.batch_size * self.gradient_accumulation_steps,
                "max_prompt_length": 768,
                "temperature": 1.0,
            }
        )

        # Trainer configuration
        trainer_config = DictConfig(
            {
                "output_dir": output_dir,
                "max_steps": self.max_steps,
                "per_device_train_batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "warmup_ratio": 0.1,
                "lr_scheduler_type": "cosine",
                "fp16": True,
                "logging_steps": 1,
                "report_to": "wandb" if self.wandb_enabled else "none",
                "run_name": (
                    f"{model_name_short}-{lora_suffix}-VERL-GRPO"
                    if self.wandb_enabled
                    else None
                ),
            }
        )

        # Main VERL configuration
        config = HybridEngineEntrypointConfig(
            data=data_config,
            trainer=trainer_config,
            algorithm=algorithm_config,
            critic=critic_config,
            actor_rollout_ref=actor_rollout_ref,
        )

        return config

    def print_directory_info(self) -> None:
        """Print information about workspace directories."""
        print("📁 VERL Workspace Configuration:")
        print(f"   Workspace Directory: {self.workspace_dir}")
        print(f"   Models Directory: {self.models_dir}")
        print(f"   Data Directory: {self.data_dir}")
        print(f"   Cache Directory: {self.cache_dir}")
        print(f"   Model: {self.model_name}")
        print(f"   VERL Version: {self._get_verl_version()}")
        print("-" * 50)

    def _get_verl_version(self) -> str:
        """Get VERL version."""
        try:
            import verl

            return verl.__version__
        except Exception:
            return "unknown"

    def train(self) -> None:
        """Execute the VERL training process."""
        # Print directory information
        self.print_directory_info()

        # Load and prepare dataset
        self.load_and_prepare_dataset()

        # Get configurations
        lora_config = self.get_lora_config()
        verl_config = self.get_verl_config()

        # Note: Dataset and LoRA config need to be handled separately
        # as OmegaConf cannot serialize complex objects
        print(f"📊 Dataset loaded: {len(self.dataset)} examples")
        if lora_config:
            print("🔧 LoRA configuration available")

        # Start training using VERL trainer
        print("🚀 Starting VERL GRPO training...")

        # Print configuration details
        print("📋 VERL Configuration created successfully!")
        print(f"   Algorithm: {verl_config.algorithm.adv_estimator}")
        print(f"   Model: {verl_config.critic.model.get('path', 'Not set')}")
        print(f"   Output dir: {verl_config.trainer.output_dir}")
        print(f"   Max steps: {verl_config.trainer.max_steps}")
        print(f"   Batch size: " f"{verl_config.trainer.per_device_train_batch_size}")
        print(
            f"   Gradient accumulation: "
            f"{verl_config.trainer.gradient_accumulation_steps}"
        )

        # Run the actual VERL training
        try:
            print("🔄 Starting actual VERL training...")

            if not VERL_TRAINER_AVAILABLE:
                print("❌ VERL trainer components not available")
                print("Falling back to configuration-only mode")
                print("📋 VERL Configuration created successfully!")
                print(f"   Algorithm: {verl_config.algorithm.adv_estimator}")
                print(f"   Model: {verl_config.critic.model.get('path', 'Not set')}")
                print(f"   Output dir: {verl_config.trainer.output_dir}")
                print("⚠️  Note: Actual training requires proper VERL trainer setup")
                return

            # Try to create VERL trainer with proper configuration
            try:
                trainer = Trainer(
                    config=verl_config,
                    reward_fn=self.reward_function,
                    train_dataset=self.dataset,
                )

                # Start training
                print("🚀 Starting VERL trainer...")
                trainer.train()
                print("✅ VERL GRPO training completed successfully!")

            except Exception as trainer_error:
                print(f"⚠️  VERL Trainer creation failed: {trainer_error}")
                print("This might be due to API changes in VERL version")
                print("Falling back to configuration validation mode")

                # Fallback: just validate the configuration
                print("📋 VERL Configuration validation:")
                print(f"   Algorithm: {verl_config.algorithm.adv_estimator}")
                print(f"   Model: {verl_config.critic.model.get('path', 'Not set')}")
                print(f"   Output dir: {verl_config.trainer.output_dir}")
                print(f"   Max steps: {verl_config.trainer.max_steps}")
                print(
                    f"   Batch size: {verl_config.trainer.per_device_train_batch_size}"
                )
                print(
                    "⚠️  Note: Training requires proper VERL trainer setup on remote VM"
                )

        except Exception as e:
            print(f"❌ VERL training setup failed: {e}")
            print(
                "This might be due to missing reward function integration "
                "or dataset configuration."
            )
            print(f"Error details: {str(e)}")
            print(
                "💡 Suggestion: Check VERL installation and API compatibility on remote VM"
            )
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VERL-based GRPO Training Script for Reasoning Tasks",
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
    """Main entry point for the VERL training script."""
    if not VERL_AVAILABLE:
        print("❌ VERL is not available. Please install it with: pip install verl")
        print("💡 Suggestion: Use the TRL-based implementation instead:")
        print("   python reasoning_grpo.py --model-size 1.5B --use-lora")
        return

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
    trainer = ReasoningGRPOTrainerVERL(
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
