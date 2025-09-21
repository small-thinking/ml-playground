#!/bin/bash
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --email EMAIL    Email address for SSH key generation (required)"
    echo "  -d, --docker         Setup for Docker environment (auto-installs Docker if not available)"
    echo "  -s, --skip-verl      Skip VERL installation (Docker mode only)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --email user@example.com          # Auto-detect environment and setup"
    echo "  $0 -e user@example.com -d            # Force Docker setup (auto-installs if needed)"
    echo "  $0 -e user@example.com -d -s         # Docker setup without VERL"
    echo "  $0 -e user@example.com               # Short form"
    echo ""
    echo "Features:"
    echo "  • Auto-detects Docker availability and remote/local environment"
    echo "  • Auto-installs Docker with GPU support when --docker flag is used"
    echo "  • Preserves existing SSH keys (won't overwrite)"
    echo "  • Provides environment-specific guidance and next steps"
    exit 1
}

# Parse command line arguments
EMAIL=""
DOCKER_MODE=false
SKIP_VERL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--email)
            EMAIL="$2"
            shift 2
            ;;
        -d|--docker)
            DOCKER_MODE=true
            shift
            ;;
        -s|--skip-verl)
            SKIP_VERL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate that email is provided
if [ -z "$EMAIL" ]; then
    echo "Error: Email address is required for SSH key generation"
    echo ""
    usage
fi

# Validate email format (basic validation)
if [[ ! "$EMAIL" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "Error: Invalid email format: $EMAIL"
    echo "Please provide a valid email address"
    exit 1
fi

# Check if Docker is available
DOCKER_AVAILABLE=false
if command -v docker >/dev/null 2>&1; then
    DOCKER_AVAILABLE=true
    echo "✓ Docker detected and available"
else
    echo "⚠ Docker not found"
    if [ "$DOCKER_MODE" = true ]; then
        echo "=== Installing Docker (--docker flag specified) ==="
        echo "Installing Docker on this system..."
        
        # Install Docker using the official script
        curl -fsSL https://get.docker.com | sh
        
        # Add current user to docker group (skip if running as root)
        if [ "$(id -u)" -eq 0 ]; then
            echo "Running as root - skipping user group configuration"
        else
            echo "Adding user to docker group..."
            if command -v sudo >/dev/null 2>&1; then
                sudo usermod -aG docker $USER
            else
                echo "⚠ sudo not available - user group configuration skipped"
                echo "You may need to manually add user to docker group after setup"
            fi
        fi
        
        # Start and enable Docker service (skip if running in container)
        if [ "$(id -u)" -eq 0 ] && [ -f /.dockerenv ]; then
            echo "Running in Docker container - skipping systemctl commands"
            echo "Docker daemon should already be running in container"
        else
            echo "Starting Docker service..."
            if command -v sudo >/dev/null 2>&1 && command -v systemctl >/dev/null 2>&1; then
                sudo systemctl start docker
                sudo systemctl enable docker
            else
                echo "⚠ systemctl not available - Docker service management skipped"
                echo "Docker may need to be started manually"
            fi
        fi
        
        # Install nvidia-docker2 for GPU support
        echo "Installing nvidia-docker2 for GPU support..."
        if [ "$(id -u)" -eq 0 ] && [ -f /.dockerenv ]; then
            echo "Running in Docker container - skipping nvidia-docker2 installation"
            echo "GPU support should be handled by the host Docker daemon"
        else
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            if command -v sudo >/dev/null 2>&1; then
                curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
                curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
                sudo apt-get update
                sudo apt-get install -y nvidia-docker2
                if command -v systemctl >/dev/null 2>&1; then
                    sudo systemctl restart docker
                fi
            else
                echo "⚠ sudo not available - nvidia-docker2 installation skipped"
                echo "GPU support may not be available"
            fi
        fi
        
        # Verify Docker installation
        echo "Verifying Docker installation..."
        if command -v docker >/dev/null 2>&1; then
            DOCKER_AVAILABLE=true
            echo "✓ Docker installed successfully"
            echo "✓ Docker version: $(docker --version)"
            
            # Test Docker with hello-world
            echo "Testing Docker with hello-world..."
            if [ "$(id -u)" -eq 0 ]; then
                # Running as root, no need for sudo
                if docker run hello-world >/dev/null 2>&1; then
                    echo "✓ Docker test successful"
                else
                    echo "⚠ Docker test failed, but installation appears complete"
                fi
            else
                # Not root, try with sudo if available
                if command -v sudo >/dev/null 2>&1; then
                    if sudo docker run hello-world >/dev/null 2>&1; then
                        echo "✓ Docker test successful"
                    else
                        echo "⚠ Docker test failed, but installation appears complete"
                    fi
                else
                    echo "⚠ Cannot test Docker without sudo, but installation appears complete"
                fi
            fi
            
            echo ""
            if [ "$(id -u)" -eq 0 ] && [ -f /.dockerenv ]; then
                echo "✓ Running as root in Docker container - no group membership needed"
                echo "Docker is ready to use immediately"
            else
                echo "⚠ IMPORTANT: Docker group membership requires re-login"
                echo "You may need to:"
                echo "1. Log out and back in, OR"
                echo "2. Run: newgrp docker, OR" 
                echo "3. Restart your SSH session"
                echo ""
                echo "After re-login, verify with: docker --version"
                echo "Then re-run this script with the same --docker flag"
            fi
            echo ""
            
            # Set DOCKER_MODE to true since we just installed it
            DOCKER_MODE=true
        else
            echo "✗ Docker installation failed"
            echo "Falling back to local installation mode"
            DOCKER_MODE=false
        fi
    else
        echo "Will use local installation mode"
    fi
fi

# Detect environment type
ENVIRONMENT_TYPE="local"
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ]; then
    ENVIRONMENT_TYPE="remote"
    echo "✓ Remote environment detected (SSH session)"
elif [ -n "$TMUX" ] || [ -n "$TERM_PROGRAM" ]; then
    echo "✓ Local environment detected"
fi

# Provide environment-specific guidance
if [ "$ENVIRONMENT_TYPE" = "remote" ] && [ "$DOCKER_AVAILABLE" = false ] && [ "$DOCKER_MODE" = false ]; then
    echo ""
    echo "=== Remote VM Setup Detected ==="
    echo "You're running on a remote VM without Docker. This setup will:"
    echo "• Install packages locally on the VM"
    echo "• Set up Python environment directly"
    echo "• Configure SSH keys and Git"
    echo ""
    echo "For Docker-based setup, you can:"
    echo "1. Re-run with --docker flag: $0 --email $EMAIL --docker"
    echo "2. Run Docker commands from your local machine"
    echo "3. Continue with this local installation (recommended for VMs)"
    echo ""
fi

if [ "$DOCKER_MODE" = true ]; then
    echo "=== [Docker Mode] Skipping system package installation ==="
    echo "Assuming Docker image already has required system packages"
else
    echo "=== [Step 1] Updating package lists ==="
    apt update

    echo "=== [Step 2] Installing basic tools ==="
    apt install -y \
      vim \
      git \
      curl \
      wget \
      tmux \
      unzip \
      openssh-client \
      build-essential
fi

echo "=== [Step 3] Setting up Python environment ==="
pip3 install --upgrade pip
pip3 install virtualenv ipython

# Install project requirements
pip install -r requirements.txt

echo "=== [Step 4] Installing Hugging Face CLI ==="
pip install huggingface_hub[cli]

# Install VERL if in Docker mode (moved to later step)
if [ "$DOCKER_MODE" = true ] && [ "$SKIP_VERL" = false ]; then
    echo "=== [Step 5] Installing VERL ==="
    pip3 install verl
    
    echo "=== [Step 5.1] Validating VERL Installation ==="
    python3 -c "
import sys
try:
    import verl
    print('✓ VERL imported successfully')
    print(f'✓ VERL version: {verl.__version__}')
    
    # Test basic functionality
    from verl import __version__
    print('✓ VERL version check passed')
    
    # Check if key modules are available
    try:
        from verl.trainer import VERLTrainer
        print('✓ VERLTrainer module available')
    except ImportError as e:
        print(f'⚠ VERLTrainer import warning: {e}')
    
    try:
        from verl.models import VERLModel
        print('✓ VERLModel module available')
    except ImportError as e:
        print(f'⚠ VERLModel import warning: {e}')
    
    print('✓ VERL installation validation completed successfully!')
    
except ImportError as e:
    print(f'✗ VERL import failed: {e}')
    print('✗ VERL installation validation failed!')
    sys.exit(1)
except Exception as e:
    print(f'✗ VERL validation error: {e}')
    print('✗ VERL installation validation failed!')
    sys.exit(1)
"
    
    echo "=== [Step 5.2] Setting up Megatron (optional) ==="
    echo "To set up Megatron for training, run:"
    echo "  cd .."
    echo "  git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git"
    echo "  cp verl/patches/megatron_v4.patch Megatron-LM/"
    echo "  cd Megatron-LM && git apply megatron_v4.patch"
    echo "  pip3 install -e ."
    echo "  export PYTHONPATH=\$PYTHONPATH:\$(pwd)"
elif [ "$DOCKER_MODE" = true ] && [ "$SKIP_VERL" = true ]; then
    echo "=== [Step 5] Skipping VERL Installation ==="
    echo "VERL installation skipped. To install VERL later, run:"
    echo "  pip3 install verl"
    echo "  python3 /app/modeling/validate_verl.py"
fi

echo "=== [Step 6] Installing oh-my-bash ==="
if [ ! -d "$HOME/.oh-my-bash" ]; then
  git clone https://github.com/ohmybash/oh-my-bash.git ~/.oh-my-bash
  cp ~/.oh-my-bash/templates/bashrc.osh-template ~/.bashrc
  sed -i 's/^OSH_THEME=.*/OSH_THEME="font"/' ~/.bashrc
fi

echo "=== [Step 7] Configuring Git ==="
# Replace the values below with your identity if needed
git config --global user.name "Yexi Jiang"
git config --global user.email "2237303+yxjiang@users.noreply.github.com"
git config --global init.defaultBranch main
git config --global core.editor vim
git config --global color.ui auto
git config --global credential.helper store

echo "=== [Step 8] Setting up SSH ==="
# Create SSH directory and set permissions
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Check if SSH key already exists
if [ -f ~/.ssh/id_ed25519 ] && [ -f ~/.ssh/id_ed25519.pub ]; then
    echo "✓ SSH key already exists, skipping generation"
    echo "Existing public key:"
    echo "----------------------------------------"
    cat ~/.ssh/id_ed25519.pub
    echo "----------------------------------------"
    echo ""
    echo "If you need to add this key to GitHub/GitLab:"
    echo "1. Go to GitHub Settings > SSH and GPG keys"
    echo "2. Click 'New SSH key'"
    echo "3. Copy the public key above and paste it"
    echo ""
else
    echo "Generating SSH key with email: $EMAIL"
    # Generate SSH key non-interactively
    ssh-keygen -t ed25519 -C "$EMAIL" -f ~/.ssh/id_ed25519 -N ""
    chmod 600 ~/.ssh/id_ed25519
    chmod 644 ~/.ssh/id_ed25519.pub

    echo "SSH key generated successfully!"
    echo "Public key (add this to GitHub/GitLab):"
    echo "----------------------------------------"
    cat ~/.ssh/id_ed25519.pub
    echo "----------------------------------------"
    echo ""
    echo "To add this key to GitHub:"
    echo "1. Go to GitHub Settings > SSH and GPG keys"
    echo "2. Click 'New SSH key'"
    echo "3. Copy the public key above and paste it"
    echo ""
fi

if [ "$DOCKER_MODE" = false ]; then
    echo "=== [Step 9] Install Ollama ==="
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &
else
    echo "=== [Step 9] Skipping Ollama installation (Docker mode) ==="
    echo "Ollama can be installed separately if needed"
fi

echo "=== [Step 10] Setting up workspace directories ==="
# Create workspace directories (use /workspace for Docker, ~/workspace for local)
if [ "$DOCKER_MODE" = true ]; then
    WORKSPACE_BASE="/workspace"
    echo "Docker mode: Using /workspace as base directory"
else
    WORKSPACE_BASE="$HOME/workspace"
    echo "Local mode: Using ~/workspace as base directory"
fi

mkdir -p "$WORKSPACE_BASE/models"
mkdir -p "$WORKSPACE_BASE/data"
mkdir -p "$WORKSPACE_BASE/cache"
mkdir -p "$WORKSPACE_BASE/logs"

# Set environment variables for HuggingFace
echo "export HF_HOME=$WORKSPACE_BASE/cache" >> ~/.bashrc
echo "export TRANSFORMERS_CACHE=$WORKSPACE_BASE/models" >> ~/.bashrc
echo "export HF_DATASETS_CACHE=$WORKSPACE_BASE/data" >> ~/.bashrc
echo "export WORKSPACE_DIR=$WORKSPACE_BASE" >> ~/.bashrc

# Add VERL-specific environment variables if in Docker mode
if [ "$DOCKER_MODE" = true ]; then
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    echo "export NCCL_P2P_DISABLE=1" >> ~/.bashrc
    echo "export NCCL_IB_DISABLE=1" >> ~/.bashrc
fi

echo "=== Done ==="
echo "Workspace setup complete!"
echo "Directories created:"
echo "  - $WORKSPACE_BASE/models (for model storage)"
echo "  - $WORKSPACE_BASE/data (for dataset storage)"
echo "  - $WORKSPACE_BASE/cache (for HuggingFace cache)"
echo "  - $WORKSPACE_BASE/logs (for log files)"
echo ""
echo "To activate the environment, run: source ~/.bashrc"
echo "SSH key generated and ready to use!"
echo ""
if [ "$DOCKER_MODE" = true ]; then
    echo "Docker setup complete! Next steps:"
    echo "1. Login to Hugging Face: huggingface-cli login"
    echo "2. Add your SSH key to GitHub/GitLab (see instructions above)"
    if [ "$SKIP_VERL" = true ]; then
        echo "3. Install VERL when ready: pip3 install verl"
        echo "4. Validate VERL: python3 /app/modeling/validate_verl.py"
    else
        echo "3. VERL is already installed and validated!"
    fi
    echo "5. For VERL training, use the provided Docker commands:"
    echo "   - Example: docker run --runtime=nvidia -it --rm --shm-size=\"10g\" --cap-add=SYS_ADMIN verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3"
    echo "6. Mount your workspace: -v $WORKSPACE_BASE:/workspace"
else
    echo "Setup complete! Next steps:"
    echo "1. Login to Hugging Face: huggingface-cli login"
    echo "2. Add your SSH key to GitHub/GitLab (see instructions above)"
    
    if [ "$ENVIRONMENT_TYPE" = "remote" ]; then
        echo "3. For remote VM training:"
        echo "   - GRPO: python reasoning_grpo.py --model-size 3B --use-lora"
        echo "   - SFT:  python instruction_sft.py --model-size 3B --use-lora"
        echo ""
        echo "4. To use Docker on this VM later:"
        echo "   - Run: $0 --email $EMAIL --docker (auto-installs Docker)"
        echo "   - Or manually: curl -fsSL https://get.docker.com | sh"
        echo "   - Then: ./docker_setup.sh --email $EMAIL"
    else
        echo "3. Start training:"
        echo "   - GRPO: python reasoning_grpo.py --model-size 3B --use-lora"
        echo "   - SFT:  python instruction_sft.py --model-size 3B --use-lora"
    fi
fi
