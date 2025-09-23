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

- **Nix with flakes enabled** for the reproducible development shells
- **NVIDIA GPU** (optional) for fast inference – the agent will fall back to CPU/MPS automatically

### Prepare your environment

```bash
# Copy the sample environment and fill in your credentials
cp .env.example .env
$EDITOR .env

# Enter a development shell (choose the one that fits your hardware)
nix develop            # Default CPU shell
# nix develop .#gpu    # CUDA-enabled shell
```

The flake also exposes additional shells when needed:

| Shell | Purpose |
|-------|---------|
| `default` | Standard CPU-oriented development |
| `gpu` | CUDA 12.8 stack with binary PyTorch |
| `ai` | Adds common CLI tools (codex/claude/gemini) on top of the default shell |
| `neko` | Bundles Docker Compose helpers (`neko-services`) for the reference server |
| `cpu-opt` | Builds Python packages with `znver2` optimisations |
| `gpu-opt` | CUDA + `znver2` tuned builds |
| `docs` | mdBook + linkcheck stack for documentation work |
| `tee` | Tooling for the trusted execution environment pipeline |

### Start a Neko server (optional)

```bash
nix develop .#neko
neko-services up
# The reference stack listens on http://localhost:8080
```

### Run the agent

```bash
# Basic usage
uv run src/agent.py --task "Navigate to google.com and search for 'AI automation'"

# REST login flow (the agent will mint a WebSocket token)
uv run src/agent.py \
  --task "Your automation task" \
  --neko-url "http://localhost:8080" \
  --username "user" \
  --password "password"

# Stay online and receive new tasks from chat
uv run src/agent.py --online --neko-url "http://localhost:8080" --username user --password password

# Health-check the configuration without starting the session
uv run src/agent.py --healthcheck
```

If you prefer `python` over `uv`, the commands work the same – the scripts have no side effects at import time.

### Helpful `just` targets

The repository includes a `justfile` with shortcuts. The commands assume you have customised paths (for example `NEKO_LOGFILE`) to match your machine.

```bash
just uv-agent               # Load .env and run the agent via uv
just agent-task "look up cats"  # Fire-and-forget example task (edit paths first!)
just manual                 # Launch the manual REPL controller
just kill-all               # Stop leftover automation processes
just docker-build-generic   # Build the portable Docker image via Nix
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
| `ai` | Extended tooling | Adds Codex/Claude/Gemini CLI utilities |
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
uv run src/capture.py

# Terminal 2: Run the agent (capture will automatically observe chat + frames)
uv run src/agent.py --task "Your task description"

# Train on collected data
uv run src/train.py
```

## Voice Output

Enable voice feedback during automation:

```bash
# Terminal 1: Start TTS service (uses F5-TTS)
uv run src/yap.py

# Terminal 2: Run the agent – audio is enabled by default (set NEKO_AUDIO=0 to disable)
uv run src/agent.py --task "Narrate what you see"
```

The agent streams audio from the Neko session; YAP injects synthesised speech over the same WebRTC connection when it sees `/yap` chat commands.

## Configuration

Create a `.env` file in the project root (see `.env.example` for the exhaustive list):

```bash
# Neko server connection
NEKO_URL=http://localhost:8080
NEKO_USER=user
NEKO_PASS=password

# Agent behaviour
NEKO_TASK="Search the weather"
NEKO_MODE=web               # web | phone
NEKO_MAX_STEPS=8
NEKO_AUDIO=1                # 0 disables audio negotiation
REFINEMENT_STEPS=5
NEKO_RTCP_KEEPALIVE=0
NEKO_FORCE_EXIT_GUARD_MS=0
NEKO_SKIP_INITIAL_FRAMES=5
NEKO_LOGLEVEL=INFO
NEKO_LOG_FORMAT=text        # text | json
NEKO_METRICS_PORT=9000
# FRAME_SAVE_PATH="./tmp/frame.png"   # optional last-frame snapshot (relative paths land in /tmp/neko-agent)
# CLICK_SAVE_PATH="./tmp/actions"     # optional directory for annotated frames

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
YAP_MAX_CHARS=350
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
uv run src/agent.py --healthcheck

# Show running processes
just ps

# Inspect the bundled Neko stack (from the `neko` shell)
neko-services status
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

## Metrics

Prometheus metrics are exposed on `NEKO_METRICS_PORT` (or `$PORT` when running on hosting platforms):

- `neko_frames_received_total`
- `neko_actions_executed_total{action_type="..."}`
- `neko_parse_errors_total`
- `neko_navigation_steps_total`
- `neko_inference_latency_seconds`
- `neko_resize_duration_seconds`
- `neko_reconnects_total`

Scrape them with:

```bash
curl http://localhost:${NEKO_METRICS_PORT:-9000}/metrics
```

## Support

- Documentation: `nix run .#docs-serve`
- Issues: Open an issue on the repository
- Development: `nix develop .#gpu` for the full CUDA-enabled environment