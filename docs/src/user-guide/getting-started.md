# Getting Started

This guide will help you set up and run the Neko Agent system for AI-powered GUI automation.

## Prerequisites

- **Nix with flakes enabled** - for development environment
- **Neko server** - target environment for automation
- **NVIDIA GPU** (optional) - for optimal AI model performance

## Development Environment Setup

The project uses Nix flakes for reproducible development environments. Choose the appropriate shell based on your needs:

### Available Development Shells

```bash
# Basic CPU environment
nix develop

# GPU-accelerated environment (recommended)
nix develop .#gpu

# Shell with additional AI CLI tools
nix develop .#ai

# Docker-compose helper environment for the bundled Neko stack
nix develop .#neko

# Documentation tooling
nix develop .#docs

# Optimised variants
nix develop .#cpu-opt
nix develop .#gpu-opt

# Trusted Execution Environment tooling
nix develop .#tee
```

### GPU Environment (Recommended)

For the best performance with AI models:

```bash
nix develop .#gpu
```

This provides:
- CUDA 12.8 toolkit and libraries
- PyTorch with CUDA support
- All required Python dependencies
- Pre-configured environment variables

## Neko Server Setup

The agent requires a running Neko server to connect to. You can either:

### Option 1: Use Docker Compose (Included)

```bash
# Enter the Neko environment
nix develop .#neko

# Start Neko services
neko-services up

# View logs
neko-services logs

# Stop services
neko-services down
```

This starts a Neko server at `http://localhost:8080` with Chrome ready for automation.

### Option 2: External Neko Server

Configure connection to an existing Neko server by setting environment variables:

```bash
export NEKO_URL="https://your-neko-server.com"
export NEKO_USER="your-username"
export NEKO_PASS="your-password"
```

## Running the Agent

### Basic Usage

```bash
# Run with a simple task
uv run src/agent.py --task "Navigate to google.com and search for 'AI automation'"

# Run with REST authentication (the agent performs the login handshake)
uv run src/agent.py \
  --task "Your automation task" \
  --neko-url "http://localhost:8080" \
  --username "user" \
  --password "password"

# Keep the agent online and accept new tasks from chat
uv run src/agent.py --online --neko-url "http://localhost:8080" --username user --password password
```

### With Training Data Capture

To enable training data collection during automation:

```bash
# Terminal 1: Start capture service
uv run src/capture.py

# Terminal 2: Run agent (capture watches chat messages for /start and /stop)
uv run src/agent.py --task "Your task description"
```

See [Training Data Capture](./capture.md) for detailed capture configuration.

### With Voice Output

For voice feedback during automation:

```bash
# Terminal 1: Start TTS service (F5-TTS + WebRTC audio)
uv run src/yap.py

# Terminal 2: Run the agent – leave audio enabled (default) so the browser hears YAP
uv run src/agent.py --task "Describe what you see"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (copy `.env.example` first):

```bash
# Neko server connection
NEKO_URL=http://localhost:8080
NEKO_USER=user
NEKO_PASS=password

# Agent behaviour
NEKO_TASK="Default task description"
NEKO_MODE=web
NEKO_MAX_STEPS=8
NEKO_AUDIO=1
REFINEMENT_STEPS=5
NEKO_ICE_POLICY=strict
NEKO_RTCP_KEEPALIVE=0
NEKO_SKIP_INITIAL_FRAMES=5
NEKO_FORCE_EXIT_GUARD_MS=0
NEKO_LOGLEVEL=INFO
NEKO_LOG_FORMAT=text
NEKO_METRICS_PORT=9000
# FRAME_SAVE_PATH="./tmp/frame.png"
# CLICK_SAVE_PATH="./tmp/actions"

# Capture settings (optional)
CAPTURE_OUT=./data/mds
CAPTURE_FPS=2.0
CAPTURE_REMOTE=s3://your-bucket/training-data

# TTS settings (optional)
YAP_VOICES_DIR=./voices
YAP_SR=48000
YAP_PARALLEL=2
YAP_MAX_CHARS=350
```

### GPU Configuration

For optimal GPU usage:

```bash
# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Configure memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set target architecture
export TORCH_CUDA_ARCH_LIST=8.6
```

## Verification

Test your setup:

```bash
# Check dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test agent connection
uv run src/agent.py --healthcheck

# Explore capture options (the script exits when interrupted)
uv run src/capture.py --help

# Test TTS (if enabled)
uv run src/yap.py --healthcheck
```

## Next Steps

- **Read the [Architecture Overview](../developer-guide/architecture.md)** to understand the system design
- **Explore [Training Data Capture](./capture.md)** to collect data for model improvement
- **Check the [Developer Guide](../developer-guide/components.md)** for technical details
- **Review [Neko Integration](../developer-guide/neko.md)** for advanced server configuration

## Troubleshooting

### Common Issues

**CUDA not detected:**
```bash
# Verify NVIDIA drivers
nvidia-smi

# Check CUDA installation in shell
echo $CUDA_HOME
```

**Neko connection failed:**
```bash
# Test Neko server accessibility
curl http://localhost:8080/health

# Check WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  http://localhost:8080/api/ws
```

**Python import errors:**
```bash
# Regenerate environment
nix develop .#gpu --command python -c "import streaming; import torch; print('OK')"
```

For more help, see the [Developer Guide](../developer-guide/development.md) or check the project issues.