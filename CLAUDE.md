# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neko Agent is an AI-powered automation system that connects to Neko servers via WebRTC for real-time GUI automation. It uses AI vision models (ShowUI-2B/Qwen2VL) for visual reasoning and action planning, captures training data automatically, and provides voice synthesis capabilities.

## Development Environment Setup

This project uses Nix flakes for reproducible development environments:

```bash
# GPU-accelerated environment (recommended)
nix develop .#gpu

# Basic CPU environment
nix develop

# Documentation environment
nix develop .#docs

# TEE deployment environment
nix develop .#tee

# Neko server environment
nix develop .#neko
```

## Common Development Commands

### Agent Operations
```bash
# Start agent with default task
just agent

# Start agent with custom task
just agent-task "your automation task here"

# Manual control mode
just manual

# Kill running processes
just kill-agent
just kill-manual
just kill-all
```

### Development Utilities
```bash
# View logs
just log

# Show running processes
just ps

# Clean temporary files
just clean

# Setup directories
just setup
```

### Neko Server Management
```bash
# Start Neko server via Docker Compose
nix develop .#neko
neko-services up

# View server at http://localhost:8080
just browser
```

### Docker Image Building
```bash
# Build all Docker images
just build-images

# Build specific variants
nix build .#neko-agent-docker-generic    # Portable CUDA
nix build .#neko-agent-docker-opt        # Optimized (znver2 + sm_86)

# Run with Docker
just docker-run-generic
just docker-run-optimized
```

### Training and Data Capture
```bash
# Train on captured data
just train

# Start training data capture service
uv run src/capture.py

# Start TTS service
uv run src/yap.py
```

### UV Package Management (REQUIRED)
```bash
# Initialize new Python project
uv init

# Add dependencies
uv add package-name

# Remove dependencies
uv remove package-name

# Install all dependencies
uv sync

# Run Python scripts
uv run src/script.py

# Run agent with UV
just uv-agent

# Train with UV
just uv-train
```

## Core Architecture

### Main Components
- **`src/agent.py`** - Core automation agent with WebRTC integration and AI vision models
- **`src/capture.py`** - Training data capture service using MosaicML Streaming (MDS) format
- **`src/yap.py`** - Text-to-speech service with F5-TTS and WebRTC audio broadcasting
- **`src/train.py`** - Model fine-tuning on captured automation data
- **`src/manual.py`** - Manual control interface for testing

### Key Patterns
- **WebRTC Communication**: Real-time video frame analysis and action execution via aiortc
- **AI Vision Pipeline**: ShowUI-2B/Qwen2VL models for visual reasoning and GUI automation
- **Training Data Flow**: MosaicML Streaming format for scalable training data collection
- **12-Factor App Design**: Environment-based configuration, structured logging, stateless design

### Configuration
All configuration is environment-based following 12-factor principles. Copy `.env.example` to `.env` and configure:

- **NEKO_URL/NEKO_USER/NEKO_PASS** - Neko server connection
- **MODEL_KEY/REPO_ID** - AI model configuration
- **CAPTURE_OUT/CAPTURE_REMOTE** - Training data paths
- **YAP_VOICES_DIR** - Voice synthesis configuration

### Development Environments
- `default` - Basic Python development with PyTorch CPU
- `gpu` - CUDA 12.8 with GPU-accelerated PyTorch for model inference
- `cpu-opt/gpu-opt` - Compiler-optimized variants (znver2/sm_86 targets)
- `docs` - Documentation tools (mdbook, linkcheck)
- `neko` - Docker services for Neko server
- `tee` - TEE deployment tools (Phala Cloud CLI, VMM)

### Testing
No formal test framework is configured. Testing is done via:
- Manual validation using `just manual`
- Integration testing with live Neko servers
- Model performance validation via captured training episodes

### Deployment
Supports multiple deployment targets:
- Local development via Nix shells
- Docker containers with CUDA support
- TEE (Trusted Execution Environment) deployment via Phala Cloud
- OCI registry management for container distribution

## Python Development Standards (MANDATORY)

### Package Management
**ALL DEVELOPMENT MUST USE UV TOOL** - This is non-negotiable. UV replaces pip, pip-tools, pipx, poetry, virtualenv, and twine with a unified interface.

#### Essential UV Commands
```bash
# Project setup
uv init                    # Create new project with pyproject.toml
uv venv                    # Create virtual environment
uv sync                    # Install dependencies from lockfile

# Dependency management
uv add package-name        # Add to pyproject.toml + update lock
uv remove package-name     # Remove from pyproject.toml + update lock
uv lock                    # Regenerate uv.lock with exact versions

# Running code
uv run script.py           # Run script in project environment
uv run --script name       # Run script defined in pyproject.toml

# Tool management
uv tool run tool-name      # Run tool in ephemeral environment
uvx tool-name              # Alias for uv tool run
uv tool install tool-name  # Install tool globally

# Publishing
uv build                   # Build source and wheel distributions
uv publish                 # Publish to PyPI
```

### Documentation Standards
Follow PEP 257 for docstring conventions and PEP 287 for reStructuredText formatting:

```python
def example_function(param1: str, param2: int) -> bool:
    """One-line summary that fits on one line.

    More detailed explanation of the function, its behavior,
    and any important implementation details. Use reStructuredText
    for rich formatting when needed.

    :param param1: Description of first parameter
    :param param2: Description of second parameter
    :return: Description of return value
    :raises ValueError: When param2 is negative
    """
```

### pyproject.toml Structure
All projects must use pyproject.toml following PEP 518/517/621:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "package-name"
version = "0.1.0"
description = "Brief description"
authors = [{name = "Author Name", email = "author@example.com"}]
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
]
requires-python = ">=3.11"

[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]

[project.scripts]
command-name = "module:function"
```

### Code Style Requirements
- Follow PEP 8 for style guidelines
- Use type hints consistently (PEP 484/526)
- Wrap docstrings at 72 characters
- Use reStructuredText markup for complex documentation
- Prefer dependency injection and environment-based configuration
- All imports should be sorted and organized

### Development Workflow
1. **Always use `uv run`** instead of direct Python execution
2. **Maintain `uv.lock`** for reproducible builds - commit this file
3. **Use `uv sync`** before running tests or building
4. **Leverage `uv tool run`** for development tools to avoid dependency conflicts
5. **Document all functions and classes** following PEP 257/287
6. **Validate pyproject.toml** follows PEP 621 metadata standards