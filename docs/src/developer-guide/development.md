# Development Setup

This guide covers the development environment setup, coding standards, and contribution workflow for the Neko Agent project.

## Development Environment

### Prerequisites

- **Nix with flakes** - Package management and development shells
- **Git** - Version control
- **NVIDIA GPU** (optional but recommended) - For AI model acceleration

### Setting Up

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd neko-agent
   ```

2. **Enter development environment:**
   ```bash
   # For GPU development (recommended)
   nix develop .#gpu
   
   # For CPU-only development
   nix develop
   ```

3. **Verify setup:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

### Available Development Shells

| Shell | Purpose | Key Packages |
|-------|---------|--------------|
| `default` | Basic Python development | PyTorch CPU, transformers, websockets |
| `gpu` | GPU-accelerated development | CUDA 12.8, PyTorch GPU, all dependencies |
| `docs` | Documentation development | mdBook, Sphinx, preprocessing tools |
| `neko` | Neko server management | Docker, Colima, compose tools |
| `ai` | AI tool integration | Claude Code CLI, OpenAI tools |

## Project Structure

```text
├── src/                    # Source code
│   ├── agent.py           # Core automation agent
│   ├── capture.py         # Training data capture
│   └── yap.py             # TTS service
├── docs/                  # Documentation
│   ├── src/               # mdBook source files
│   └── book.toml          # Documentation configuration
├── voices/                # Voice models and assets
├── data/                  # Training data output
├── overlays/              # Nix package overlays
├── nix/                   # Nix configuration
└── flake.nix              # Development environment
```

## Coding Standards

### Python Style

- **Follow PEP 8** for code style
- **Use type hints** for all function signatures
- **Write docstrings** for public functions and classes
- **Use async/await** for I/O operations

Example:
```python
async def process_frame(frame: np.ndarray, task: str) -> Optional[Action]:
    """
    Process a video frame and determine the next action.
    
    :param frame: RGB frame data as numpy array
    :param task: Natural language task description
    :return: Action to execute, or None if task complete
    """
    # Implementation here
    pass
```

### Error Handling

- **Use specific exceptions** rather than generic Exception
- **Log errors with context** for debugging
- **Implement graceful degradation** where possible

```python
try:
    result = await risky_operation()
except SpecificError as e:
    logger.warning(f"Operation failed: {e}, falling back to default")
    result = default_fallback()
```

### Configuration Management

- **Use environment variables** for configuration
- **Provide sensible defaults** in code
- **Document all configuration options**

```python
NEKO_URL = os.environ.get("NEKO_URL", "http://localhost:8080")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_agent.py
```

### Test Structure

- **Unit tests** for individual functions
- **Integration tests** for component interaction
- **End-to-end tests** for full workflows

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_action_execution():
    """Test that agent executes actions correctly."""
    agent = NekoAgent()
    
    with patch.object(agent, 'webrtc_client') as mock_client:
        mock_client.execute_action = AsyncMock()
        
        action = ClickAction(x=100, y=200)
        await agent.execute_action(action)
        
        mock_client.execute_action.assert_called_once_with(action)
```

## Documentation

### Building Documentation

```bash
# Enter docs environment
nix develop .#docs

# Serve documentation locally
cd docs && mdbook serve --open

# Build static documentation
cd docs && mdbook build
```

### Documentation Standards

- **Write in Markdown** using mdBook extensions
- **Include code examples** for all APIs
- **Use mermaid diagrams** for architecture
- **Keep docs up-to-date** with code changes

## Git Workflow

### Branch Strategy

- **`main`** - Stable release branch
- **`dev`** - Development integration branch  
- **`feature/*`** - Feature development branches
- **`fix/*`** - Bug fix branches

### Commit Guidelines

- **Use conventional commits** format
- **Write descriptive messages** explaining the why
- **Keep commits atomic** - one logical change per commit

Examples:
```text
feat(agent): add support for swipe actions
fix(capture): handle missing frames gracefully  
docs(api): update WebRTC connection examples
refactor(tts): improve voice loading performance
```

### Pull Request Process

1. **Create feature branch** from `dev`
2. **Implement changes** following coding standards
3. **Add/update tests** for new functionality
4. **Update documentation** if needed
5. **Submit PR** with clear description
6. **Address review feedback**
7. **Merge after approval**

## Debugging

### Logging Configuration

```bash
# Set log levels
export AGENT_LOGLEVEL=DEBUG
export CAPTURE_LOGLEVEL=INFO
export YAP_LOGLEVEL=WARNING
```

### Common Debug Tasks

**WebRTC connection issues:**
```bash
# Check Neko server status
curl http://localhost:8080/health

# Test WebSocket endpoint
wscat -c ws://localhost:8080/api/ws?token=<token>
```

**AI model problems:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test model loading
python -c "from transformers import Qwen2VLForConditionalGeneration; print('OK')"
```

**Training data capture:**
```bash
# Verify MDS output
python -c "from streaming import StreamingDataset; print('MDS OK')"

# Check S3 connectivity
aws s3 ls s3://your-bucket/
```

## Performance Optimization

### Profiling

```bash
# Profile with cProfile
python -m cProfile -o profile.stats src/agent.py

# Analyze with snakeviz
snakeviz profile.stats
```

### GPU Memory Management

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Set memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### WebRTC Optimization

- **Reduce frame rate** for lower bandwidth
- **Adjust video quality** based on task complexity
- **Use hardware encoding** when available

## Contributing

### Getting Started

1. **Read the documentation** to understand the system
2. **Set up development environment** 
3. **Pick an issue** from the project board
4. **Ask questions** if anything is unclear

### Areas for Contribution

- **New action types** (drag, hover, etc.)
- **Additional AI models** integration
- **Performance improvements**
- **Documentation enhancements**
- **Test coverage** expansion

### Code Review Guidelines

- **Be constructive** in feedback
- **Explain the reasoning** behind suggestions
- **Test the changes** locally when possible
- **Approve promptly** for good contributions

## Release Process

### Version Management

- **Semantic versioning** (MAJOR.MINOR.PATCH)
- **Tag releases** in Git
- **Update changelog** for each release

### Deployment

- **Build Docker images** for production
- **Test in staging** environment
- **Deploy with rolling updates**
- **Monitor metrics** post-deployment

## Support

### Getting Help

- **Check documentation** first
- **Search existing issues** on GitHub
- **Ask in discussions** for general questions
- **Open issues** for bugs or feature requests

### Community Guidelines

- **Be respectful** and inclusive
- **Help others** when you can
- **Share knowledge** and experiences
- **Follow the code of conduct**