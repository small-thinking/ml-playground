# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY README.md ./

# Create virtual environment and install dependencies
RUN uv venv && \
    uv pip install -e .

# Copy source code
COPY . .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs

# Set default command
CMD ["python", "-c", "import torch; print(f'PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}')"]
