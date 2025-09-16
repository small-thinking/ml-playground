# Docker Setup for VERL

This guide explains how to set up and use VERL with Docker for your ML playground project.

## Quick Start

### Option 0: Auto-Detection Setup (Recommended for Remote VMs)

The script now automatically detects your environment and adapts accordingly:

```bash
cd modeling
bash setup_env.sh --email your-email@example.com
```

**What it detects:**
- ✓ Docker availability (switches to local mode if Docker not found)
- ✓ Remote vs local environment (SSH session detection)
- ✓ Existing SSH keys (preserves them, won't overwrite)
- ✓ Provides environment-specific guidance

### Option 1: Automated Setup (Recommended)

Use the provided `docker_setup.sh` script for automated setup:

```bash
cd modeling
./docker_setup.sh --email your-email@example.com
```

**Skip VERL installation** (useful for basic environment setup first):

```bash
./docker_setup.sh --email your-email@example.com --skip-verl
```

### Option 2: Manual Setup

1. **Pull the VERL Docker image:**

   ```bash
   docker pull verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
   ```

2. **Run the container with proper mounts:**

   ```bash
   docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
     -v $(pwd)/workspace:/workspace \
     -v $(pwd)/modeling:/app/modeling \
     -p 8888:8888 -p 6006:6006 \
     verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3
   ```

3. **Setup environment inside container:**

   ```bash
   cd /app/modeling
   bash setup_env.sh --email your-email@example.com --docker
   ```

   **Skip VERL installation:**

   ```bash
   bash setup_env.sh --email your-email@example.com --docker --skip-verl
   ```

## Updated setup_env.sh Features

The `setup_env.sh` script now supports Docker mode with the `--docker` flag:

### Key Changes:

- **Auto-Detection**: Automatically detects Docker availability and environment type
- **Docker Mode**: Skip system package installation (handled by Docker image)
- **VERL Installation**: Automatically install VERL when in Docker mode (moved to Step 5)
- **Skip VERL Option**: Option to skip VERL installation for basic environment setup
- **SSH Key Preservation**: Won't overwrite existing SSH keys
- **Remote VM Support**: Detects remote environments and provides appropriate guidance
- **Workspace Paths**: Use `/workspace` for Docker, `~/workspace` for local
- **Environment Variables**: Add VERL-specific CUDA and NCCL settings
- **Megatron Setup**: Provide instructions for optional Megatron installation

### Usage:

```bash
# Local setup (original behavior)
bash setup_env.sh --email your-email@example.com

# Docker setup with VERL
bash setup_env.sh --email your-email@example.com --docker

# Docker setup without VERL (basic environment only)
bash setup_env.sh --email your-email@example.com --docker --skip-verl
```

## Docker Setup Script Options

The `docker_setup.sh` script provides several options:

```bash
./docker_setup.sh [OPTIONS]

Options:
  -e, --email EMAIL    Email address for SSH key generation (required)
  -i, --image IMAGE    VERL Docker image (default: verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3)
  -w, --workspace DIR  Local workspace directory (default: ./workspace)
  -n, --name NAME      Container name (default: verl-container)
  -s, --skip-verl      Skip VERL installation during setup
  -h, --help          Show this help message
```

## Workspace Structure

When using Docker, the workspace is structured as follows:

```
/workspace/          # Inside container (mounted from host)
├── models/          # Model storage
├── data/            # Dataset storage
├── cache/           # HuggingFace cache
└── logs/            # Training logs

/app/modeling/       # Project files (mounted from host)
├── setup_env.sh     # Updated setup script
├── docker_setup.sh  # Docker automation script
└── ...              # Other project files
```

## Environment Variables

The Docker setup automatically configures these environment variables:

```bash
# HuggingFace
export HF_HOME=/workspace/cache
export TRANSFORMERS_CACHE=/workspace/models
export HF_DATASETS_CACHE=/workspace/data
export WORKSPACE_DIR=/workspace

# VERL-specific (Docker mode only)
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

## Prerequisites

1. **Docker**: Install Docker with NVIDIA support
2. **nvidia-docker2**: For GPU acceleration
3. **NVIDIA Drivers**: Compatible with your GPU

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker runtime
docker info | grep nvidia

# Install nvidia-docker2 if missing
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Permission Issues

```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Container Won't Start

```bash
# Check container logs
docker logs verl-container

# Remove and recreate
docker rm verl-container
./docker_setup.sh --email your-email@example.com
```

## Next Steps

After setup:

1. **Access container:**

   ```bash
   docker exec -it verl-container bash
   ```

2. **Validate VERL installation:**

   ```bash
   # Quick validation (built into setup)
   python3 -c "import verl; print(f'VERL version: {verl.__version__}')"

   # Comprehensive validation
   python3 /app/modeling/validate_verl.py
   ```

3. **Login to Hugging Face:**

   ```bash
   huggingface-cli login
   ```

4. **Start training:**
   ```bash
   # Your VERL training commands here
   ```

## VERL Installation Validation

After installing VERL, you can validate the installation using the provided validation script:

### Quick Validation

```bash
python3 -c "import verl; print(f'VERL version: {verl.__version__}')"
```

### Comprehensive Validation

```bash
python3 /app/modeling/validate_verl.py
```

The validation script checks:

- ✓ VERL core module import
- ✓ VERL version information
- ✓ Key VERL modules (trainer, models, data, utils)
- ✓ Required dependencies (torch, transformers, datasets, etc.)
- ✓ GPU availability and CUDA support
- ✓ Basic VERL functionality

### Expected Output

```
=== VERL Installation Validation ===

✓ VERL core imported successfully (version: 0.2.0)
✓ VERL version: 0.2.0

=== Checking VERL Modules ===
✓ VERLTrainer imported successfully
✓ VERLModel imported successfully
✓ VERL Data utilities imported successfully
✓ VERL utilities imported successfully

=== Checking Dependencies ===
✓ PyTorch imported successfully (version: 2.1.0)
✓ Hugging Face Transformers imported successfully (version: 4.36.0)
...

=== Validation Summary ===
✓ VERL installation validation PASSED!
✓ All core modules are available
✓ Ready for VERL training!
```

## Benefits of Docker Setup

- **Consistent Environment**: Same setup across different machines
- **GPU Support**: Built-in NVIDIA CUDA support
- **Isolation**: No conflicts with host system packages
- **Reproducibility**: Easy to share exact environment
- **Easy Cleanup**: Remove container to clean up completely
- **Built-in Validation**: Automatic installation verification
