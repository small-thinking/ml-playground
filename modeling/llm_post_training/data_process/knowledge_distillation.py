#!/usr/bin/env python3
"""
Knowledge distillation orchestrator for LLM-based data generation.

This script provides a generic interface for knowledge distillation using
different LLM APIs and configurable prompts for various data generation tasks.
"""

import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from dataclasses import dataclass, asdict
import time
from datetime import datetime

from llm_api_interface import (
    LLMRequest,
    LLMResponse,
    BatchLLMRequest,
    create_llm_provider,
    LLMAPIError,
)
from prompt_manager import PromptManager


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation process."""

    provider_type: str
    provider_config: Dict[str, Any]
    task_name: str
    prompt_name: str
    input_file: str
    output_file: str
    batch_size: int = 10
    max_concurrent: int = 5
    retry_attempts: int = 3
    delay_between_batches: float = 1.0
    save_intermediate: bool = True
    intermediate_dir: str = "intermediate_results"


@dataclass
class DistillationResult:
    """Result of knowledge distillation process."""

    total_processed: int
    successful: int
    failed: int
    total_cost: Optional[float] = None
    processing_time: Optional[float] = None
    output_file: Optional[str] = None
    errors: List[str] = None


class KnowledgeDistillationOrchestrator:
    """
    Orchestrates the knowledge distillation process using LLM APIs.

    Handles batch processing, error recovery, progress tracking, and
    result management for large-scale data generation tasks.
    """

    def __init__(self, config: DistillationConfig, prompt_manager: PromptManager):
        """
        Initialize the orchestrator.

        Args:
            config: Distillation configuration
            prompt_manager: Prompt manager instance
        """
        self.config = config
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize LLM provider
        self.provider = create_llm_provider(
            config.provider_type, config.provider_config
        )

        if not self.provider.validate_config():
            raise ValueError(
                f"Invalid configuration for {config.provider_type} provider"
            )

        # Create output directories
        self.output_path = Path(config.output_file)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.save_intermediate:
            self.intermediate_dir = Path(config.intermediate_dir)
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "errors": [],
            "start_time": None,
            "end_time": None,
        }

    async def process_dataset(self) -> DistillationResult:
        """
        Process the entire dataset through knowledge distillation.

        Returns:
            Distillation result with statistics
        """
        self.logger.info(
            f"Starting knowledge distillation for task: {self.config.task_name}"
        )
        self.stats["start_time"] = time.time()

        try:
            # Load input data
            input_data = self._load_input_data()
            self.logger.info(f"Loaded {len(input_data)} input records")

            # Process in batches
            results = []
            total_batches = (
                len(input_data) + self.config.batch_size - 1
            ) // self.config.batch_size

            for batch_idx in range(0, len(input_data), self.config.batch_size):
                batch_data = input_data[batch_idx : batch_idx + self.config.batch_size]
                batch_num = batch_idx // self.config.batch_size + 1

                self.logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch_data)} records)"
                )

                # Process batch
                batch_results = await self._process_batch(batch_data, batch_num)
                results.extend(batch_results)

                # Save intermediate results
                if self.config.save_intermediate:
                    self._save_intermediate_results(results, batch_num)

                # Delay between batches
                if batch_num < total_batches and self.config.delay_between_batches > 0:
                    await asyncio.sleep(self.config.delay_between_batches)

            # Save final results
            self._save_final_results(results)

            # Calculate final statistics
            self.stats["end_time"] = time.time()
            result = self._create_result()

            self.logger.info(f"Knowledge distillation completed: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Knowledge distillation failed: {e}")
            self.stats["errors"].append(str(e))
            raise

    def _load_input_data(self) -> List[Dict[str, Any]]:
        """
        Load input data from file.

        Returns:
            List of input records
        """
        input_path = Path(self.config.input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix.lower() == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif input_path.suffix.lower() in [".csv", ".tsv"]:
            df = pd.read_csv(input_path)
            return df.to_dict("records")
        elif input_path.suffix.lower() == ".jsonl":
            data = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

    async def _process_batch(
        self, batch_data: List[Dict[str, Any]], batch_num: int
    ) -> List[Dict[str, Any]]:
        """
        Process a single batch of data.

        Args:
            batch_data: List of input records
            batch_num: Batch number for logging

        Returns:
            List of processed results
        """
        # Prepare requests
        requests = []
        for i, record in enumerate(batch_data):
            try:
                # Render prompt with record data
                prompt = self.prompt_manager.render_prompt(
                    self.config.task_name, self.config.prompt_name, record
                )

                if not prompt:
                    self.logger.error(
                        f"Failed to render prompt for record {i} in batch {batch_num}"
                    )
                    self.stats["failed"] += 1
                    continue

                # Create LLM request
                request = LLMRequest(
                    prompt=prompt,
                    metadata={
                        "batch_num": batch_num,
                        "record_index": i,
                        "original_record": record,
                    },
                )
                requests.append(request)

            except Exception as e:
                self.logger.error(
                    f"Error preparing request for record {i} in batch {batch_num}: {e}"
                )
                self.stats["failed"] += 1
                continue

        if not requests:
            self.logger.warning(f"No valid requests in batch {batch_num}")
            return []

        # Process requests with retry logic
        responses = await self._process_requests_with_retry(requests, batch_num)

        # Process responses
        results = []
        for i, (request, response) in enumerate(zip(requests, responses)):
            try:
                result = self._process_response(request, response, batch_data[i])
                results.append(result)
                self.stats["successful"] += 1
            except Exception as e:
                self.logger.error(
                    f"Error processing response {i} in batch {batch_num}: {e}"
                )
                self.stats["failed"] += 1
                self.stats["errors"].append(str(e))

        self.stats["total_processed"] += len(batch_data)
        return results

    async def _process_requests_with_retry(
        self, requests: List[LLMRequest], batch_num: int
    ) -> List[LLMResponse]:
        """
        Process requests with retry logic.

        Args:
            requests: List of LLM requests
            batch_num: Batch number for logging

        Returns:
            List of LLM responses
        """
        for attempt in range(self.config.retry_attempts):
            try:
                if len(requests) == 1:
                    # Single request
                    response = await self.provider.generate_single(requests[0])
                    return [response]
                else:
                    # Batch request
                    batch_request = BatchLLMRequest(
                        requests=requests, batch_id=f"batch_{batch_num}_{attempt}"
                    )
                    batch_response = await self.provider.generate_batch(batch_request)
                    return batch_response.responses

            except LLMAPIError as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed for batch {batch_num}: {e}"
                )
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

        return []

    def _process_response(
        self,
        request: LLMRequest,
        response: LLMResponse,
        original_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a single response into the final result format.

        Args:
            request: Original LLM request
            response: LLM response
            original_record: Original input record

        Returns:
            Processed result record
        """
        result = {
            "original_input": original_record,
            "generated_output": response.content,
            "metadata": {
                "prompt_used": self.config.prompt_name,
                "task_name": self.config.task_name,
                "provider": self.provider.get_provider_name(),
                "request_metadata": request.metadata,
                "response_metadata": response.metadata,
                "usage": response.usage,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Add any errors
        if response.metadata and "error" in response.metadata:
            result["metadata"]["error"] = response.metadata["error"]

        return result

    def _save_intermediate_results(
        self, results: List[Dict[str, Any]], batch_num: int
    ) -> None:
        """Save intermediate results to file."""
        if not self.config.save_intermediate:
            return

        intermediate_file = self.intermediate_dir / f"batch_{batch_num:04d}.json"
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved intermediate results: {intermediate_file}")

    def _save_final_results(self, results: List[Dict[str, Any]]) -> None:
        """Save final results to output file."""
        output_path = Path(self.config.output_file)

        if output_path.suffix.lower() == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif output_path.suffix.lower() == ".jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            # Default to JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved final results: {output_path}")

    def _create_result(self) -> DistillationResult:
        """Create distillation result from statistics."""
        processing_time = None
        if self.stats["start_time"] and self.stats["end_time"]:
            processing_time = self.stats["end_time"] - self.stats["start_time"]

        return DistillationResult(
            total_processed=self.stats["total_processed"],
            successful=self.stats["successful"],
            failed=self.stats["failed"],
            processing_time=processing_time,
            output_file=self.config.output_file,
            errors=self.stats["errors"],
        )


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("knowledge_distillation.log"),
        ],
    )


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation with LLM APIs")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Load configuration
    with open(args.config, "r") as f:
        config_data = json.load(f)

    # Create configuration objects
    distillation_config = DistillationConfig(**config_data["distillation"])
    prompt_manager = PromptManager(config_data["prompt_config"]["config_dir"])

    # Create orchestrator and run
    orchestrator = KnowledgeDistillationOrchestrator(
        distillation_config, prompt_manager
    )
    result = await orchestrator.process_dataset()

    print(f"\n✅ Knowledge distillation completed!")
    print(f"📊 Total processed: {result.total_processed}")
    print(f"✅ Successful: {result.successful}")
    print(f"❌ Failed: {result.failed}")
    if result.processing_time:
        print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
    print(f"📁 Output file: {result.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
