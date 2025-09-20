# Neko Agent

An AI-powered automation agent that interfaces with Neko servers to perform GUI automation tasks through WebRTC connections.

## What is Neko Agent?

Neko Agent is a Python-based automation system that:

- **Connects to Neko servers** via WebRTC for real-time GUI interaction
- **Uses AI vision models** (ShowUI-2B/Qwen2VL) for visual reasoning and action planning
- **Captures training data** automatically during automation sessions
- **Provides voice synthesis** capabilities via F5-TTS and WebRTC audio
- **Supports various actions** like clicking, typing, scrolling, and navigation

## Quick Start

### Prerequisites

- **Nix with flakes enabled** - for development environment
- **NVIDIA GPU** (optional) - for optimal AI model performance

### Setup Development Environment

```bash
# GPU-accelerated environment (recommended)
nix develop .#gpu

# Basic CPU environment
nix develop

# Documentation environment
nix develop .#docs
```

### Start Neko Server

```bash
# Using included Docker Compose
nix develop .#neko
neko-services up

# View at http://localhost:8080
```

### Run the Agent

```bash
# Basic usage
python src/agent.py --task "Navigate to google.com and search for 'AI automation'"

# With custom configuration
python src/agent.py \
  --task "Your automation task" \
  --max-steps 20 \
  --neko-url "http://localhost:8080" \
  --neko-user "user" \
  --neko-pass "password"
```

### Using Just Commands

```bash
# Start agent with default task
just agent

# Start agent with custom task
just agent-task "search for pics of cats"

# Manual control mode
just manual

# View logs
just log

# Clean up
just clean
```

## Core Components

- **`src/agent.py`** - Core automation agent with WebRTC integration
- **`src/capture.py`** - Training data capture service using MosaicML Streaming
- **`src/yap.py`** - Text-to-speech service with F5-TTS and voice management
- **`src/train.py`** - Model training on captured data
- **`src/manual.py`** - Manual control interface

## Development Environments

| Shell | Purpose | Features |
|-------|---------|----------|
| `default` | Basic development | PyTorch CPU, Python deps |
| `gpu` | GPU development | CUDA 12.8, GPU-accelerated PyTorch |
| `cpu-opt` | CPU optimized | Znver2 compiler flags |
| `gpu-opt` | GPU optimized | Znver2 + CUDA sm_86 |
| `docs` | Documentation | mdbook, linkcheck, mermaid |
| `neko` | Docker services | Neko server, Docker Compose |
| `tee` | TEE deployment | Phala Cloud CLI, VMM tools |

## Container Images

Build reproducible Docker images with attestation metadata:

```bash
# Build all images
just build-images

# Build specific variants
nix build .#neko-agent-docker-generic    # Portable CUDA
nix build .#neko-agent-docker-opt        # Optimized (znver2 + sm_86)

# Run with Docker
just docker-run-generic
just docker-run-optimized
```

## TEE Deployment

Deploy to Trusted Execution Environments with attestation:

```bash
# Deploy to TEE with reproducible images
nix run .#deploy-to-tee

# Deploy to ttl.sh ephemeral registry
nix run .#deploy-to-ttl 1h

# Build and push to ttl.sh
nix run .#push-to-ttl 24h
```

### Registry Options

```bash
# GitHub Container Registry
NEKO_REGISTRY=ghcr.io/your-org nix run .#deploy-to-tee

# Docker Hub
NEKO_REGISTRY=docker.io/your-org nix run .#deploy-to-tee

# Local registry
NEKO_REGISTRY=localhost:5000 nix run .#deploy-to-tee

# ttl.sh (anonymous, ephemeral)
NEKO_REGISTRY=ttl.sh NEKO_TTL=1h nix run .#deploy-to-tee
```

## Local Registry

Start a local OCI registry for development:

```bash
# HTTP registry
nix run .#start-registry

# HTTPS registry with Tailscale certificates
nix run .#start-registry-https

# Expose via Tailscale Funnel
nix run .#start-tailscale-funnel

# Expose via Cloudflare Tunnel
nix run .#start-cloudflare-tunnel
```

## Training Data Collection

Enable automatic training data capture:

```bash
# Terminal 1: Start capture service
python src/capture.py

# Terminal 2: Run agent (capture will automatically record)
python src/agent.py --task "Your task description"

# Train on collected data
just train
```

## Voice Output

Enable voice feedback during automation:

```bash
# Terminal 1: Start TTS service
python src/yap.py

# Terminal 2: Run agent with voice announcements
python src/agent.py --task "Your task" --enable-voice
```

## Configuration

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

# GPU configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_CUDA_ARCH_LIST=8.6

# Capture settings (optional)
CAPTURE_OUT=./data/mds
CAPTURE_FPS=2.0
CAPTURE_REMOTE=s3://your-bucket/training-data

# TTS settings (optional)
YAP_VOICES_DIR=./voices
YAP_SR=48000
YAP_PARALLEL=2
```

## Documentation

View the full documentation:

```bash
# Start documentation server
nix develop .#docs
cd docs && mdbook serve --open

# Or use the app
nix run .#docs-serve
```

## GPU Management

List and manage GPUs for TEE deployment:

```bash
# List available GPUs
just gpus

# Deploy with specific GPUs
just deploy-gpu docker-compose.yml "0a:00.0 1a:00.0"

# Deploy without GPU
just deploy docker-compose.yml
```

## Verification

Test your setup:

```bash
# Check dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Test agent connection
python src/agent.py --help

# Show running processes
just ps

# View system status
just status
```

## Architecture

The system is designed around WebRTC-based communication with Neko containers, allowing for:

- Real-time video frame analysis and action execution
- Automated training data collection in MosaicML format
- Voice feedback through WebRTC audio streams
- Scalable deployment patterns for production use

## Security & Attestation

All container images are built with reproducible timestamps and include attestation metadata for TEE verification. The deployment system supports:

- Digest pinning for supply chain security
- Reproducible builds with Nix
- TEE attestation via TDX quotes
- Multi-registry support for availability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Use the appropriate development shell
4. Run tests and ensure builds pass
5. Submit a pull request

## License

See LICENSE file for details.

## Support

- Documentation: Run `nix run .#docs-serve`
- Issues: Open an issue on the repository
- Development: Use `nix develop .#gpu` for the full environment