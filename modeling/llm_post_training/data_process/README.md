# Knowledge Distillation System

A generic, pluggable system for knowledge distillation using different LLM APIs and configurable prompts for various data generation tasks.

## Features

- **Generic LLM API Interface**: Easy switching between different LLM providers (OpenAI, Anthropic, etc.)
- **Batch Processing**: Efficient processing of large datasets with OpenAI batch inference API
- **Configurable Prompts**: YAML-based prompt templates for different tasks
- **Error Handling**: Robust error handling with retry logic and progress tracking
- **Flexible Output**: Support for JSON, JSONL, and CSV output formats
- **Intermediate Results**: Save progress during long-running tasks

## Architecture

```
data_process/
├── llm_api_interface.py          # Abstract base class for LLM providers
├── prompt_manager.py             # YAML-based prompt management
├── knowledge_distillation.py     # Main orchestrator
├── providers/                    # LLM provider implementations
│   ├── __init__.py
│   └── openai_provider.py        # OpenAI API implementation
├── prompt_configs/               # YAML prompt configurations
│   ├── instruction_generation.yaml
│   ├── qa_generation.yaml
│   └── summarization.yaml
├── example_config.json           # Sample configuration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install openai pyyaml jinja2 pandas
```

### 2. Set Up API Keys

Create a `.env` file in the parent directory:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Create Input Data

Create a JSON file with your input data:

```json
[
  {
    "text": "Your source text here",
    "domain": "optional domain context",
    "difficulty": "optional difficulty level"
  }
]
```

### 4. Configure the System

Create a configuration file (e.g., `my_config.json`):

```json
{
  "distillation": {
    "provider_type": "openai",
    "provider_config": {
      "api_key": "your-openai-api-key-here",
      "model": "gpt-3.5-turbo",
      "use_batch_api": true,
      "batch_timeout": 3600
    },
    "task_name": "instruction_generation",
    "prompt_name": "basic_instruction",
    "input_file": "input_data.json",
    "output_file": "output_data.json",
    "batch_size": 10,
    "max_concurrent": 5,
    "retry_attempts": 3,
    "delay_between_batches": 1.0,
    "save_intermediate": true,
    "intermediate_dir": "intermediate_results"
  },
  "prompt_config": {
    "config_dir": "prompt_configs"
  }
}
```

### 5. Run Knowledge Distillation

```bash
python knowledge_distillation.py --config my_config.json
```

## Available Tasks

### Instruction Generation

Generate instruction-following training data from raw text.

**Available Prompts:**

- `basic_instruction`: Generate basic instruction-following examples
- `creative_task`: Generate creative tasks from text content
- `analysis_prompt`: Generate analytical tasks from text

**Input Format:**

```json
{
  "text": "Source text to generate instructions from",
  "domain": "Optional domain context",
  "difficulty": "Optional difficulty level"
}
```

### Q&A Generation

Generate question-answer pairs from text content.

**Available Prompts:**

- `factual_qa`: Generate factual question-answer pairs
- `analytical_qa`: Generate analytical question-answer pairs
- `application_qa`: Generate application-based question-answer pairs

**Input Format:**

```json
{
  "text": "Source text to generate Q&A from",
  "context": "Optional additional context",
  "question_type": "Type of questions to generate"
}
```

### Summarization

Generate summaries and abstractive content from source text.

**Available Prompts:**

- `extractive_summary`: Generate extractive summary by selecting key sentences
- `abstractive_summary`: Generate abstractive summary by paraphrasing
- `bullet_point_summary`: Generate structured bullet-point summary

**Input Format:**

```json
{
  "text": "Source text to summarize",
  "length": "Desired summary length (short/medium/long)",
  "focus": "Optional focus area for summary"
}
```

## Creating Custom Prompts

### 1. Create YAML Configuration

Create a new YAML file in `prompt_configs/`:

```yaml
task_name: "my_custom_task"
description: "Description of your custom task"

input_format:
  field1: "string"
  field2: "string"

output_format:
  output_field: "string"

prompts:
  my_prompt:
    description: "Description of your prompt"
    variables: ["field1", "field2"]
    template: |
      Your prompt template here using Jinja2 syntax.

      Field 1: {{ field1 }}
      Field 2: {{ field2 }}

      Please provide your response in the following format:
      Output: [your response here]

metadata:
  version: "1.0"
  author: "Your Name"
  created: "2024-01-01"
```

### 2. Use Custom Prompt

Update your configuration to use the new task and prompt:

```json
{
  "distillation": {
    "task_name": "my_custom_task",
    "prompt_name": "my_prompt"
    // ... other configuration
  }
}
```

## Adding New LLM Providers

### 1. Create Provider Class

Create a new file in `providers/` (e.g., `anthropic_provider.py`):

```python
from ..llm_api_interface import LLMAPIProvider, LLMRequest, LLMResponse, BatchLLMRequest, BatchLLMResponse

class AnthropicProvider(LLMAPIProvider):
    def __init__(self, config):
        super().__init__(config)
        # Initialize Anthropic client

    async def generate_single(self, request: LLMRequest) -> LLMResponse:
        # Implement single request generation

    async def generate_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        # Implement batch request generation

    def validate_config(self) -> bool:
        # Validate configuration
        return True
```

### 2. Register Provider

Update `llm_api_interface.py` to include your new provider:

```python
def create_llm_provider(provider_type: str, config: Dict[str, Any]) -> LLMAPIProvider:
    # ... existing code ...
    elif provider_type == 'anthropic':
        from .providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(config)
```

## Configuration Options

### Distillation Configuration

| Option                  | Type   | Default                | Description                                  |
| ----------------------- | ------ | ---------------------- | -------------------------------------------- |
| `provider_type`         | string | -                      | LLM provider type (openai, anthropic, etc.)  |
| `provider_config`       | dict   | -                      | Provider-specific configuration              |
| `task_name`             | string | -                      | Name of the task to use                      |
| `prompt_name`           | string | -                      | Name of the prompt to use                    |
| `input_file`            | string | -                      | Path to input data file                      |
| `output_file`           | string | -                      | Path to output data file                     |
| `batch_size`            | int    | 10                     | Number of requests per batch                 |
| `max_concurrent`        | int    | 5                      | Maximum concurrent requests                  |
| `retry_attempts`        | int    | 3                      | Number of retry attempts for failed requests |
| `delay_between_batches` | float  | 1.0                    | Delay between batches (seconds)              |
| `save_intermediate`     | bool   | true                   | Save intermediate results                    |
| `intermediate_dir`      | string | "intermediate_results" | Directory for intermediate results           |

### OpenAI Provider Configuration

| Option          | Type   | Default         | Description                            |
| --------------- | ------ | --------------- | -------------------------------------- |
| `api_key`       | string | -               | OpenAI API key                         |
| `model`         | string | "gpt-3.5-turbo" | Model to use                           |
| `base_url`      | string | None            | Custom base URL                        |
| `use_batch_api` | bool   | true            | Use batch inference API                |
| `batch_timeout` | int    | 3600            | Timeout for batch operations (seconds) |

## Error Handling

The system includes comprehensive error handling:

- **Retry Logic**: Automatic retry for failed requests with exponential backoff
- **Progress Tracking**: Save intermediate results to resume from failures
- **Error Logging**: Detailed logging of errors and warnings
- **Graceful Degradation**: Fallback to single requests if batch API fails

## Performance Optimization

- **Batch Processing**: Use OpenAI batch inference API for cost efficiency
- **Concurrent Requests**: Configurable concurrency limits
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Intermediate Results**: Save progress to avoid reprocessing

## Cost Estimation

The OpenAI provider includes cost estimation:

```python
# Get cost estimate for a list of requests
cost_estimate = provider.get_cost_estimate(requests)
print(f"Estimated cost: ${cost_estimate['total_cost']:.4f}")
```

## Examples

See `example_usage.py` for comprehensive examples of:

- Instruction generation
- Q&A generation
- Custom prompt creation
- Error handling
- Progress tracking

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure your API key is set in the configuration
2. **Invalid Prompt Variables**: Check that all required variables are provided in input data
3. **Batch API Timeout**: Increase `batch_timeout` or use single requests
4. **Rate Limiting**: Reduce `max_concurrent` or increase `delay_between_batches`

### Debug Mode

Enable debug logging:

```bash
python knowledge_distillation.py --config my_config.json --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your provider or prompt configuration
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License.
