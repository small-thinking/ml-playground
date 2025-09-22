#!/usr/bin/env python3
"""
Generic LLM API interface for knowledge distillation.

This module provides an abstract base class for different LLM providers,
enabling easy switching between OpenAI, Anthropic, and other APIs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
import logging


@dataclass
class LLMRequest:
    """Represents a single LLM request."""

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Represents a single LLM response."""

    content: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    request_metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchLLMRequest:
    """Represents a batch of LLM requests."""

    requests: List[LLMRequest]
    batch_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchLLMResponse:
    """Represents a batch of LLM responses."""

    responses: List[LLMResponse]
    batch_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMAPIProvider(ABC):
    """
    Abstract base class for LLM API providers.

    This interface allows for easy switching between different LLM providers
    while maintaining consistent batch processing capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def generate_single(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a single response from the LLM.

        Args:
            request: Single LLM request

        Returns:
            Single LLM response
        """
        pass

    @abstractmethod
    async def generate_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """
        Generate responses for a batch of requests.

        Args:
            batch_request: Batch of LLM requests

        Returns:
            Batch of LLM responses
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the provider configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return self.__class__.__name__

    def log_usage(self, response: Union[LLMResponse, BatchLLMResponse]) -> None:
        """
        Log usage statistics for monitoring and cost tracking.

        Args:
            response: Single or batch response to log
        """
        if isinstance(response, LLMResponse):
            if response.usage:
                self.logger.info(f"Usage: {response.usage}")
        elif isinstance(response, BatchLLMResponse):
            total_usage = {}
            for resp in response.responses:
                if resp.usage:
                    for key, value in resp.usage.items():
                        total_usage[key] = total_usage.get(key, 0) + value
            if total_usage:
                self.logger.info(f"Batch usage: {total_usage}")


class LLMAPIError(Exception):
    """Base exception for LLM API errors."""

    pass


class LLMAPIProviderError(LLMAPIError):
    """Exception raised when provider-specific errors occur."""

    pass


class LLMAPIValidationError(LLMAPIError):
    """Exception raised when request validation fails."""

    pass


def create_llm_provider(provider_type: str, config: Dict[str, Any]) -> LLMAPIProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Type of provider ('openai', 'anthropic', etc.)
        config: Provider-specific configuration

    Returns:
        Initialized LLM provider instance

    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        from providers.openai_provider import OpenAIProvider

        return OpenAIProvider(config)
    elif provider_type == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


async def process_requests_concurrently(
    provider: LLMAPIProvider, requests: List[LLMRequest], max_concurrent: int = 10
) -> List[LLMResponse]:
    """
    Process multiple requests concurrently with rate limiting.

    Args:
        provider: LLM provider instance
        requests: List of requests to process
        max_concurrent: Maximum number of concurrent requests

    Returns:
        List of responses in the same order as requests
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single(request: LLMRequest) -> LLMResponse:
        async with semaphore:
            return await provider.generate_single(request)

    tasks = [process_single(req) for req in requests]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    processed_responses = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            provider.logger.error(f"Request {i} failed: {response}")
            # Create error response
            error_response = LLMResponse(
                content="",
                metadata={"error": str(response)},
                request_metadata=requests[i].metadata,
            )
            processed_responses.append(error_response)
        else:
            processed_responses.append(response)

    return processed_responses
