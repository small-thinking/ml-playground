#!/usr/bin/env python3
"""
OpenAI API provider implementation for knowledge distillation.

This module implements the OpenAI provider with support for both
single requests and batch inference API.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import openai
from openai import AsyncOpenAI

from llm_api_interface import (
    LLMAPIProvider,
    LLMRequest,
    LLMResponse,
    BatchLLMRequest,
    BatchLLMResponse,
    LLMAPIProviderError,
)


class OpenAIProvider(LLMAPIProvider):
    """
    OpenAI API provider implementation.

    Supports both single requests and batch inference API for efficient
    processing of large datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: OpenAI API key
                - model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
                - base_url: Optional custom base URL
                - use_batch_api: Whether to use batch inference API
                - batch_timeout: Timeout for batch operations (seconds)
        """
        super().__init__(config)

        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.base_url = config.get("base_url")
        self.use_batch_api = config.get("use_batch_api", True)
        self.batch_timeout = config.get("batch_timeout", 3600)  # 1 hour

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = AsyncOpenAI(**client_kwargs)

        # Batch API specific
        self.batch_client = (
            openai.OpenAI(**client_kwargs) if self.use_batch_api else None
        )

    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        if not self.api_key:
            self.logger.error("OpenAI API key is required")
            return False

        if not self.model:
            self.logger.error("OpenAI model is required")
            return False

        return True

    async def generate_single(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a single response using OpenAI API.

        Args:
            request: Single LLM request

        Returns:
            Single LLM response

        Raises:
            LLMAPIProviderError: If API call fails
        """
        try:
            # Prepare request parameters
            request_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            # Remove None values
            request_params = {k: v for k, v in request_params.items() if v is not None}

            # Make API call
            response = await self.client.chat.completions.create(**request_params)

            # Extract response content
            content = response.choices[0].message.content or ""

            # Extract usage information
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            )

            return LLMResponse(
                content=content,
                usage=usage,
                metadata={"model": self.model},
                request_metadata=request.metadata,
            )

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise LLMAPIProviderError(f"OpenAI API call failed: {e}")

    async def generate_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """
        Generate responses for a batch of requests.

        Uses OpenAI batch inference API for efficiency when available,
        otherwise falls back to concurrent single requests.

        Args:
            batch_request: Batch of LLM requests

        Returns:
            Batch of LLM responses
        """
        if self.use_batch_api and self.batch_client:
            return await self._generate_batch_api(batch_request)
        else:
            return await self._generate_batch_concurrent(batch_request)

    async def _generate_batch_api(
        self, batch_request: BatchLLMRequest
    ) -> BatchLLMResponse:
        """
        Generate batch using OpenAI batch inference API.

        Args:
            batch_request: Batch of LLM requests

        Returns:
            Batch of LLM responses
        """
        try:
            # Prepare batch file
            batch_data = []
            for i, req in enumerate(batch_request.requests):
                batch_data.append(
                    {
                        "custom_id": f"request_{i}_{batch_request.batch_id or ''}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [{"role": "user", "content": req.prompt}],
                            "max_tokens": req.max_tokens,
                            "temperature": req.temperature,
                            "top_p": req.top_p,
                        },
                    }
                )

            # Create batch file
            batch_file_path = f"/tmp/openai_batch_{int(time.time())}.jsonl"
            with open(batch_file_path, "w") as f:
                for item in batch_data:
                    f.write(json.dumps(item) + "\n")

            # Upload batch file
            with open(batch_file_path, "rb") as f:
                batch_file = self.batch_client.files.create(file=f, purpose="batch")

            # Create batch
            batch = self.batch_client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            self.logger.info(f"Created batch {batch.id}, waiting for completion...")

            # Wait for batch completion
            batch = self._wait_for_batch_completion(batch.id)

            # Retrieve results
            batch_output = self.batch_client.files.content(batch.output_file_id)
            results = []
            for line in batch_output.text.split("\n"):
                if line.strip():
                    results.append(json.loads(line))

            # Process results
            responses = []
            for result in results:
                if result.get("response", {}).get("status_code") == 200:
                    response_data = result["response"]["body"]
                    content = response_data["choices"][0]["message"]["content"]
                    usage = response_data.get("usage")

                    responses.append(
                        LLMResponse(
                            content=content,
                            usage=usage,
                            metadata={"batch_id": batch.id},
                            request_metadata=result.get("custom_id"),
                        )
                    )
                else:
                    # Handle error
                    error_msg = (
                        result.get("response", {})
                        .get("body", {})
                        .get("error", "Unknown error")
                    )
                    responses.append(
                        LLMResponse(
                            content="",
                            metadata={"error": error_msg, "batch_id": batch.id},
                            request_metadata=result.get("custom_id"),
                        )
                    )

            # Cleanup
            Path(batch_file_path).unlink(missing_ok=True)

            return BatchLLMResponse(
                responses=responses,
                batch_id=batch.id,
                metadata={"provider": "openai_batch_api"},
            )

        except Exception as e:
            self.logger.error(f"OpenAI batch API error: {e}")
            # Fallback to concurrent requests
            self.logger.info("Falling back to concurrent single requests...")
            return await self._generate_batch_concurrent(batch_request)

    def _wait_for_batch_completion(
        self, batch_id: str, check_interval: int = 30
    ) -> Any:
        """
        Wait for batch completion with polling.

        Args:
            batch_id: OpenAI batch ID
            check_interval: Seconds between status checks

        Returns:
            Completed batch object
        """
        while True:
            batch = self.batch_client.batches.retrieve(batch_id)

            if batch.status == "completed":
                return batch
            elif batch.status == "failed":
                raise LLMAPIProviderError(f"Batch {batch_id} failed: {batch.errors}")
            elif batch.status in ["cancelled", "cancelling"]:
                raise LLMAPIProviderError(f"Batch {batch_id} was cancelled")

            self.logger.info(f"Batch {batch_id} status: {batch.status}, waiting...")
            time.sleep(check_interval)

    async def _generate_batch_concurrent(
        self, batch_request: BatchLLMRequest
    ) -> BatchLLMResponse:
        """
        Generate batch using concurrent single requests.

        Args:
            batch_request: Batch of LLM requests

        Returns:
            Batch of LLM responses
        """
        # Use the utility function from the interface
        from llm_api_interface import process_requests_concurrently

        responses = await process_requests_concurrently(
            self, batch_request.requests, max_concurrent=10
        )

        return BatchLLMResponse(
            responses=responses,
            batch_id=batch_request.batch_id,
            metadata={"provider": "openai_concurrent"},
        )

    def get_cost_estimate(self, requests: List[LLMRequest]) -> Dict[str, float]:
        """
        Estimate cost for a list of requests.

        Args:
            requests: List of requests to estimate

        Returns:
            Dictionary with cost estimates
        """
        # Rough token estimation (4 chars per token)
        total_input_tokens = sum(len(req.prompt) // 4 for req in requests)
        estimated_output_tokens = sum((req.max_tokens or 100) for req in requests)

        # GPT-3.5-turbo pricing (as of 2024)
        input_cost_per_1k = 0.0015
        output_cost_per_1k = 0.002

        input_cost = (total_input_tokens / 1000) * input_cost_per_1k
        output_cost = (estimated_output_tokens / 1000) * output_cost_per_1k

        return {
            "input_tokens": total_input_tokens,
            "output_tokens": estimated_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "currency": "USD",
        }
