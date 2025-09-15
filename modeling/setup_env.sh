#!/bin/bash
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --email EMAIL    Email address for SSH key generation (required)"
    echo "  -d, --docker         Setup for Docker environment (skip system packages)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --email user@example.com          # Setup with SSH key generation"
    echo "  $0 -e user@example.com -d            # Docker setup"
    echo "  $0 -e user@example.com               # Short form"
    exit 1
}

# Parse command line arguments
EMAIL=""
DOCKER_MODE=false
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

# Install VERL if in Docker mode
if [ "$DOCKER_MODE" = true ]; then
    echo "=== [Step 3.1] Installing VERL ==="
    pip3 install verl
    
    echo "=== [Step 3.2] Setting up Megatron (optional) ==="
    echo "To set up Megatron for training, run:"
    echo "  cd .."
    echo "  git clone -b core_v0.4.0 https://github.com/NVIDIA/Megatron-LM.git"
    echo "  cp verl/patches/megatron_v4.patch Megatron-LM/"
    echo "  cd Megatron-LM && git apply megatron_v4.patch"
    echo "  pip3 install -e ."
    echo "  export PYTHONPATH=\$PYTHONPATH:\$(pwd)"
fi

# Install project requirements
pip install -r requirements.txt

echo "=== [Step 3.5] Installing Hugging Face CLI ==="
pip install huggingface_hub[cli]

echo "=== [Step 4] Installing oh-my-bash ==="
if [ ! -d "$HOME/.oh-my-bash" ]; then
  git clone https://github.com/ohmybash/oh-my-bash.git ~/.oh-my-bash
  cp ~/.oh-my-bash/templates/bashrc.osh-template ~/.bashrc
  sed -i 's/^OSH_THEME=.*/OSH_THEME="font"/' ~/.bashrc
fi

echo "=== [Step 5] Configuring Git ==="
# Replace the values below with your identity if needed
git config --global user.name "Yexi Jiang"
git config --global user.email "2237303+yxjiang@users.noreply.github.com"
git config --global init.defaultBranch main
git config --global core.editor vim
git config --global color.ui auto

echo "=== [Step 6] Setting up SSH ==="
# Create SSH directory and set permissions
mkdir -p ~/.ssh
chmod 700 ~/.ssh

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

if [ "$DOCKER_MODE" = false ]; then
    echo "=== [Step 7] Install Ollama ==="
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &
else
    echo "=== [Step 7] Skipping Ollama installation (Docker mode) ==="
    echo "Ollama can be installed separately if needed"
fi

echo "=== [Step 8] Setting up workspace directories ==="
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
    echo "3. For VERL training, use the provided Docker commands:"
    echo "   - Example: docker run --runtime=nvidia -it --rm --shm-size=\"10g\" --cap-add=SYS_ADMIN verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3"
    echo "4. Mount your workspace: -v $WORKSPACE_BASE:/workspace"
else
    echo "Next steps:"
    echo "1. Login to Hugging Face: huggingface-cli login"
    echo "2. Add your SSH key to GitHub/GitLab (see instructions above)"
    echo "3. Start training:"
    echo "   - GRPO: python reasoning_grpo.py --model-size 3B --use-lora"
    echo "   - SFT:  python instruction_sft.py --model-size 3B --use-lora"
fi
