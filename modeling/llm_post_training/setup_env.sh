#!/bin/bash
set -e

echo "=== [Step 1] Updating package lists ==="
apt update

echo "=== [Step 2] Installing basic tools ==="
apt install -y \
  vim \
  git \
  curl \
  wget \
  tmux \
  unzip

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

# Soft link ssh and Git setting
mkdir -p ~/.ssh
cp /workspace/bootstrap/config/.ssh/id_ed25519 ~/.ssh/id_ed25519
cp /workspace/bootstrap/config/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub
cp /workspace/bootstrap/config/.ssh/config ~/.ssh/config
cp /workspace/bootstrap/config/.gitconfig ~/.gitconfig

chmod 600 ~/.ssh/id_ed25519

# Start ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

echo "=== [Step 6] Install Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

echo "=== [Step 7] Ramp up workspace directory ==="
source .bashrc
cat /workspace/bootstrap/.bashrc >> ~/.bashrc
echo 'export HF_HOME=/workspace/cache' >> ~/.bashrc

echo "=== Done ==="
echo "Run 'source ~/.bashrc' to activate oh-my-bash"
source ~/.bashrc
