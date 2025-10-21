#!/usr/bin/env python3
"""
Prompt management system for knowledge distillation.

This module handles loading and managing prompts from YAML configuration files,
enabling easy customization of different data generation tasks.
"""

import yaml
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import logging
from jinja2 import Template, Environment, FileSystemLoader


@dataclass
class PromptConfig:
    """Configuration for a specific prompt template."""

    name: str
    template: str
    variables: List[str]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskConfig:
    """Configuration for a knowledge distillation task."""

    task_name: str
    description: str
    prompts: Dict[str, PromptConfig]
    input_format: Dict[str, Any]
    output_format: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class PromptManager:
    """
    Manages prompt templates and configurations for knowledge distillation.

    Loads prompts from YAML files and provides template rendering capabilities
    using Jinja2 templating engine.
    """

    def __init__(self, config_dir: Union[str, Path]):
        """
        Initialize prompt manager.

        Args:
            config_dir: Directory containing YAML configuration files
        """
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.config_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache for loaded configurations
        self._config_cache: Dict[str, TaskConfig] = {}

        # Load all available configurations
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        """Load all YAML configuration files from the config directory."""
        if not self.config_dir.exists():
            self.logger.warning(f"Config directory {self.config_dir} does not exist")
            return

        yaml_files = list(self.config_dir.glob("*.yaml")) + list(
            self.config_dir.glob("*.yml")
        )

        for yaml_file in yaml_files:
            try:
                self._load_config_file(yaml_file)
            except Exception as e:
                self.logger.error(f"Failed to load config {yaml_file}: {e}")

    def _load_config_file(self, config_file: Path) -> None:
        """
        Load a single YAML configuration file.

        Args:
            config_file: Path to YAML configuration file
        """
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        task_name = config_data.get("task_name", config_file.stem)

        # Parse prompts
        prompts = {}
        for prompt_name, prompt_data in config_data.get("prompts", {}).items():
            prompts[prompt_name] = PromptConfig(
                name=prompt_name,
                template=prompt_data["template"],
                variables=prompt_data.get("variables", []),
                description=prompt_data.get("description"),
                metadata=prompt_data.get("metadata", {}),
            )

        # Create task configuration
        task_config = TaskConfig(
            task_name=task_name,
            description=config_data.get("description", ""),
            prompts=prompts,
            input_format=config_data.get("input_format", {}),
            output_format=config_data.get("output_format", {}),
            metadata=config_data.get("metadata", {}),
        )

        self._config_cache[task_name] = task_config
        self.logger.info(f"Loaded task configuration: {task_name}")

    def get_available_tasks(self) -> List[str]:
        """
        Get list of available task names.

        Returns:
            List of task names
        """
        return list(self._config_cache.keys())

    def get_task_config(self, task_name: str) -> Optional[TaskConfig]:
        """
        Get configuration for a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Task configuration or None if not found
        """
        return self._config_cache.get(task_name)

    def get_prompt_config(
        self, task_name: str, prompt_name: str
    ) -> Optional[PromptConfig]:
        """
        Get prompt configuration for a specific task and prompt.

        Args:
            task_name: Name of the task
            prompt_name: Name of the prompt

        Returns:
            Prompt configuration or None if not found
        """
        task_config = self.get_task_config(task_name)
        if task_config:
            return task_config.prompts.get(prompt_name)
        return None

    def render_prompt(
        self, task_name: str, prompt_name: str, variables: Dict[str, Any]
    ) -> Optional[str]:
        """
        Render a prompt template with given variables.

        Args:
            task_name: Name of the task
            prompt_name: Name of the prompt
            variables: Variables to substitute in the template

        Returns:
            Rendered prompt string or None if not found
        """
        prompt_config = self.get_prompt_config(task_name, prompt_name)
        if not prompt_config:
            self.logger.error(f"Prompt not found: {task_name}.{prompt_name}")
            return None

        try:
            template = Template(prompt_config.template)
            return template.render(**variables)
        except Exception as e:
            self.logger.error(f"Failed to render prompt {task_name}.{prompt_name}: {e}")
            return None

    def render_prompt_from_config(
        self, prompt_config: PromptConfig, variables: Dict[str, Any]
    ) -> Optional[str]:
        """
        Render a prompt template from a PromptConfig object.

        Args:
            prompt_config: Prompt configuration object
            variables: Variables to substitute in the template

        Returns:
            Rendered prompt string or None if rendering fails
        """
        try:
            template = Template(prompt_config.template)
            return template.render(**variables)
        except Exception as e:
            self.logger.error(f"Failed to render prompt {prompt_config.name}: {e}")
            return None

    def validate_variables(
        self, task_name: str, prompt_name: str, variables: Dict[str, Any]
    ) -> bool:
        """
        Validate that all required variables are provided.

        Args:
            task_name: Name of the task
            prompt_name: Name of the prompt
            variables: Variables to validate

        Returns:
            True if all required variables are provided
        """
        prompt_config = self.get_prompt_config(task_name, prompt_name)
        if not prompt_config:
            return False

        required_vars = set(prompt_config.variables)
        provided_vars = set(variables.keys())

        missing_vars = required_vars - provided_vars
        if missing_vars:
            self.logger.error(
                f"Missing required variables for {task_name}.{prompt_name}: {missing_vars}"
            )
            return False

        return True

    def get_task_info(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a task.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with task information
        """
        task_config = self.get_task_config(task_name)
        if not task_config:
            return None

        return {
            "task_name": task_config.task_name,
            "description": task_config.description,
            "available_prompts": list(task_config.prompts.keys()),
            "input_format": task_config.input_format,
            "output_format": task_config.output_format,
            "metadata": task_config.metadata,
        }

    def reload_configs(self) -> None:
        """Reload all configuration files."""
        self._config_cache.clear()
        self._load_all_configs()
        self.logger.info("Reloaded all prompt configurations")

    def add_custom_prompt(
        self,
        task_name: str,
        prompt_name: str,
        template: str,
        variables: List[str],
        description: Optional[str] = None,
    ) -> None:
        """
        Add a custom prompt configuration at runtime.

        Args:
            task_name: Name of the task
            prompt_name: Name of the prompt
            template: Prompt template string
            variables: List of required variables
            description: Optional description
        """
        prompt_config = PromptConfig(
            name=prompt_name,
            template=template,
            variables=variables,
            description=description,
        )

        if task_name not in self._config_cache:
            # Create new task configuration
            task_config = TaskConfig(
                task_name=task_name,
                description="Custom task",
                prompts={},
                input_format={},
                output_format={},
            )
            self._config_cache[task_name] = task_config

        self._config_cache[task_name].prompts[prompt_name] = prompt_config
        self.logger.info(f"Added custom prompt: {task_name}.{prompt_name}")
