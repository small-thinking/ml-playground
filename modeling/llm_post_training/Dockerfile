# Use CUDA-enabled PyTorch base image for ML workloads
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/cache
ENV WORKSPACE_DIR=/workspace

# Set working directory
WORKDIR ${WORKSPACE_DIR}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    vim \
    git \
    curl \
    wget \
    tmux \
    unzip \
    openssh-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install virtualenv ipython

# Install uv for dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY README.md ./

# Create virtual environment and install dependencies
RUN uv venv && \
    uv pip install -e .

# Install oh-my-bash
RUN git clone https://github.com/ohmybash/oh-my-bash.git ~/.oh-my-bash && \
    cp ~/.oh-my-bash/templates/bashrc.osh-template ~/.bashrc && \
    sed -i 's/^OSH_THEME=.*/OSH_THEME="font"/' ~/.bashrc

# Configure Git (must be passed in as build args)
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
ENV GIT_USER_NAME=${GIT_USER_NAME}
ENV GIT_USER_EMAIL=${GIT_USER_EMAIL}

RUN if [ -n "$GIT_USER_NAME" ] && [ -n "$GIT_USER_EMAIL" ]; then \
        git config --global user.name "${GIT_USER_NAME}" && \
        git config --global user.email "${GIT_USER_EMAIL}" && \
        git config --global init.defaultBranch main && \
        git config --global core.editor vim && \
        git config --global color.ui auto; \
    fi

# Create SSH directory and set permissions
RUN mkdir -p ~/.ssh && \
    chmod 700 ~/.ssh

# Install Ollama (commented out by default, can be enabled via build arg)
ARG INSTALL_OLLAMA=false
RUN if [ "$INSTALL_OLLAMA" = "true" ]; then \
        curl -fsSL https://ollama.com/install.sh | sh; \
    fi

# Create necessary directories
RUN mkdir -p /workspace/cache /workspace/data /workspace/models /workspace/logs

# Copy source code
COPY . .

# Set default command
CMD ["/bin/bash"]
