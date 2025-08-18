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

# Documentation environment
nix develop .#docs

# Neko Docker environment
nix develop .#neko
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
python src/agent.py --task "Navigate to google.com and search for 'AI automation'"

# Run with custom configuration
python src/agent.py \
  --task "Your automation task" \
  --max-steps 20 \
  --neko-url "http://localhost:8080" \
  --neko-user "user" \
  --neko-pass "password"
```

### With Training Data Capture

To enable training data collection during automation:

```bash
# Terminal 1: Start capture service
python src/capture.py

# Terminal 2: Run agent (capture will automatically record)
python src/agent.py --task "Your task description"
```

See [Training Data Capture](./capture.md) for detailed capture configuration.

### With Voice Output

For voice feedback during automation:

```bash
# Terminal 1: Start TTS service
python src/yap.py

# Terminal 2: Run agent (with voice announcements)
python src/agent.py --task "Your task" --enable-voice
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Neko server connection
NEKO_URL=http://localhost:8080
NEKO_USER=user
NEKO_PASS=password

# Agent settings
AGENT_LOGLEVEL=INFO
AGENT_MAX_STEPS=15
AGENT_TIMEOUT=300

# Capture settings (optional)
CAPTURE_OUT=./data/mds
CAPTURE_FPS=2.0
CAPTURE_REMOTE=s3://your-bucket/training-data

# TTS settings (optional)
YAP_VOICES_DIR=./voices
YAP_SR=48000
YAP_PARALLEL=2
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
python src/agent.py --help

# Test capture (if enabled)
python src/capture.py --healthcheck

# Test TTS (if enabled)  
python src/yap.py --healthcheck
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