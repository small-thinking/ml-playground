#!/bin/bash
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --email EMAIL    Email address for SSH key generation (required)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --email user@example.com          # Setup with SSH key generation"
    echo "  $0 -e user@example.com               # Short form"
    exit 1
}

# Parse command line arguments
EMAIL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--email)
            EMAIL="$2"
            shift 2
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

echo "=== [Step 3] Setting up Python environment ==="
pip3 install --upgrade pip
pip3 install virtualenv ipython
pip install -r requirements.txt

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

echo "=== [Step 7] Install Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

echo "=== [Step 8] Setting up workspace directories ==="
# Create workspace directories
mkdir -p /workspace/models
mkdir -p /workspace/data
mkdir -p /workspace/cache
mkdir -p /workspace/logs

# Set environment variables for HuggingFace
echo 'export HF_HOME=/workspace/cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/workspace/models' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/data' >> ~/.bashrc
echo 'export WORKSPACE_DIR=/workspace' >> ~/.bashrc

echo "=== Done ==="
echo "Workspace setup complete!"
echo "Directories created:"
echo "  - /workspace/models (for model storage)"
echo "  - /workspace/data (for dataset storage)"
echo "  - /workspace/cache (for HuggingFace cache)"
echo "  - /workspace/logs (for log files)"
echo ""
echo "To activate the environment, run: source ~/.bashrc"
echo "SSH key generated and ready to use!"
