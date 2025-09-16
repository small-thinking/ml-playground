#!/bin/bash
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --email EMAIL    Email address for SSH key generation (required)"
    echo "  -i, --image IMAGE    VERL Docker image (default: verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3)"
    echo "  -w, --workspace DIR  Local workspace directory (default: ./workspace)"
    echo "  -n, --name NAME      Container name (default: verl-container)"
    echo "  -s, --skip-verl      Skip VERL installation during setup"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --email user@example.com"
    echo "  $0 -e user@example.com -w /path/to/workspace"
    echo "  $0 -e user@example.com -i verlai/verl:latest"
    echo "  $0 -e user@example.com -s  # Skip VERL installation"
    exit 1
}

# Default values
EMAIL=""
VERL_IMAGE="verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3"
WORKSPACE_DIR="./workspace"
CONTAINER_NAME="verl-container"
SKIP_VERL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--email)
            EMAIL="$2"
            shift 2
            ;;
        -i|--image)
            VERL_IMAGE="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
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

# Convert to absolute path
WORKSPACE_DIR=$(realpath "$WORKSPACE_DIR")

echo "=== VERL Docker Setup ==="
echo "Email: $EMAIL"
echo "Docker Image: $VERL_IMAGE"
echo "Workspace: $WORKSPACE_DIR"
echo "Container Name: $CONTAINER_NAME"
echo ""

# Create workspace directory if it doesn't exist
mkdir -p "$WORKSPACE_DIR"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU acceleration may not work."
    echo "Make sure you have nvidia-docker2 installed and configured."
fi

echo "=== Pulling VERL Docker image ==="
docker pull "$VERL_IMAGE"

echo "=== Starting VERL container ==="
docker run -d \
    --name "$CONTAINER_NAME" \
    --runtime=nvidia \
    --shm-size="10g" \
    --cap-add=SYS_ADMIN \
    -v "$WORKSPACE_DIR:/workspace" \
    -v "$(pwd)/modeling:/app/modeling" \
    -p 8888:8888 \
    -p 6006:6006 \
    "$VERL_IMAGE" \
    tail -f /dev/null

echo "=== Container started successfully! ==="
echo "Container ID: $(docker ps -q -f name=$CONTAINER_NAME)"
echo ""

echo "=== Setting up environment inside container ==="
if [ "$SKIP_VERL" = true ]; then
    docker exec "$CONTAINER_NAME" bash -c "
        cd /app/modeling && 
        bash setup_env.sh --email '$EMAIL' --docker --skip-verl
    "
else
    docker exec "$CONTAINER_NAME" bash -c "
        cd /app/modeling && 
        bash setup_env.sh --email '$EMAIL' --docker
    "
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To access the container:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "To remove the container:"
echo "  docker rm $CONTAINER_NAME"
echo ""
echo "Your workspace is mounted at: $WORKSPACE_DIR"
echo "Project files are mounted at: /app/modeling"
echo ""
echo "Next steps:"
echo "1. Access the container: docker exec -it $CONTAINER_NAME bash"
echo "2. Login to Hugging Face: huggingface-cli login"
echo "3. Start your VERL training!"
