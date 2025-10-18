# Vision Agents Architecture

This directory contains pluggable vision model implementations for the Neko automation agent.

## Overview

The agent supports multiple vision models through a clean interface abstraction:

```
src/agents/
├── __init__.py          # Factory function for creating agents
├── base.py              # Abstract VisionAgent interface
├── showui_agent.py      # ShowUI-2B/Qwen2VL local model implementation
└── remote_agent.py      # Remote API implementations (Claude, Qwen3VL)
```

## Usage

### Configuration

Set the agent type via environment variable:

```bash
# Use local ShowUI model (default)
export NEKO_AGENT_TYPE=showui

# Use Claude Computer Use API (not yet implemented)
export NEKO_AGENT_TYPE=claude

# Use Qwen3VL API (not yet implemented)
export NEKO_AGENT_TYPE=qwen3vl
```

### Running the Agent

```bash
# With ShowUI (default)
just agent

# Or explicitly set agent type
NEKO_AGENT_TYPE=showui just agent
```

## Architecture

### VisionAgent Interface

All vision agents implement the `VisionAgent` abstract base class:

```python
class VisionAgent(ABC):
    @abstractmethod
    async def generate_action(
        self,
        image: Image.Image,
        task: str,
        system_prompt: str,
        action_history: List[Dict[str, Any]],
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
    ) -> Optional[str]:
        """Generate action string from screen image and task context."""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return device and model information for logging."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up model resources."""
        pass
```

### Factory Pattern

Agents are created using the factory function:

```python
from agents import create_vision_agent

vision_agent = create_vision_agent(
    agent_type="showui",  # or "claude", "qwen3vl"
    settings=settings,
    logger=logger
)
```

## Implemented Agents

### ShowUIAgent (Local)

**Status:** ✅ Fully implemented

Uses ShowUI-2B/Qwen2VL models running locally with GPU acceleration (CUDA/MPS) or CPU fallback.

**Configuration:**
- `NEKO_AGENT_TYPE=showui`
- `REPO_ID` - Model repository (default: "showlab/ShowUI-2B")
- `SIZE_SHORTEST_EDGE` - Image preprocessing (default: 224)
- `SIZE_LONGEST_EDGE` - Image preprocessing (default: 1344)
- `OFFLOAD_FOLDER` - MPS offload directory (default: "./offload")

**Features:**
- Local inference with torch/transformers
- GPU acceleration support (CUDA/MPS)
- Automatic device detection
- Chat template formatting
- Iterative refinement support

### ClaudeComputerUseAgent (Remote)

**Status:** 🚧 Placeholder (not implemented)

Will use Anthropic's Claude Computer Use API for remote inference.

**Planned Configuration:**
- `NEKO_AGENT_TYPE=claude`
- `CLAUDE_API_KEY` - Anthropic API key
- `CLAUDE_MODEL` - Model identifier (e.g., "claude-3-5-sonnet-20241022")

### Qwen3VLAgent (Remote)

**Status:** 🚧 Placeholder (not implemented)

Will use Alibaba's Qwen3VL API for remote inference.

**Planned Configuration:**
- `NEKO_AGENT_TYPE=qwen3vl`
- `QWEN_API_KEY` - Qwen API key
- `QWEN_API_ENDPOINT` - API endpoint URL
- `QWEN_MODEL` - Model identifier

## Adding New Agents

To add a new vision model:

1. Create a new file in `src/agents/` (e.g., `new_agent.py`)
2. Implement the `VisionAgent` interface
3. Register in the factory function in `__init__.py`
4. Add environment variable configuration
5. Update validation in `agent_refactored.py` Settings

Example:

```python
# src/agents/new_agent.py
from .base import VisionAgent

class NewVisionAgent(VisionAgent):
    def __init__(self, settings, logger):
        self.settings = settings
        self.logger = logger
        # Initialize your model here

    async def generate_action(self, image, task, system_prompt,
                            action_history, crop_box, iteration, full_size):
        # Implement inference logic
        pass

    def get_device_info(self):
        return {"device": "custom", "model": "my-model"}

    async def cleanup(self):
        # Clean up resources
        pass
```

```python
# src/agents/__init__.py
def create_vision_agent(agent_type, settings, logger):
    if agent_type == "new":
        from .new_agent import NewVisionAgent
        return NewVisionAgent(settings, logger)
    # ... existing cases
```

## Benefits

- **Clean separation:** Model-specific code isolated in dedicated modules
- **Easy swapping:** Change models via environment variable
- **Testability:** Mock the VisionAgent interface for testing
- **Extensibility:** Add new models without modifying core agent logic
- **Type safety:** Abstract interface ensures consistent API

## Migration from Legacy

The refactoring maintains 100% compatibility with existing functionality:

- All ShowUI/Qwen2VL logic preserved in `ShowUIAgent`
- Same environment variables and configuration
- Same inference behavior and refinement logic
- Same action generation format

The only change is the indirection through the `VisionAgent` interface, which makes the code more modular and maintainable.
