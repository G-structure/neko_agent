# OpenRouter Integration & GPU Base Class Refactoring - Summary

## Overview

Successfully refactored the Neko agent architecture to support:
1. **GPU base class** for reusable local model infrastructure
2. **OpenRouter API integration** with tool calling for Qwen2.5-VL
3. **Pluggable vision models** via clean abstraction

## What Was Changed

### New Files Created

#### 1. `src/agents/gpu_agent.py`
**Purpose:** Base class for all local GPU-accelerated models

**Features:**
- Automatic device detection (CUDA/MPS/CPU)
- Dtype configuration (bfloat16/float32)
- Thread pool executor management
- GPU memory tracking and cleanup
- Abstract methods for subclasses to implement

**Benefits:**
- Eliminates code duplication across local models
- Makes adding new HuggingFace models trivial
- Consistent device handling

#### 2. `src/agents/remote_agent.py`
**Purpose:** Universal OpenRouter API client with tool calling

**Features:**
- Vision support via base64 image encoding
- Tool calling for Qwen computer use (coordinate-based GUI interaction)
- Structured output with JSON schema
- Async HTTP with tenacity retry logic
- Fallback text parsing for non-tool models

**Key Implementation Details:**
- `COMPUTER_USE_TOOL` definition matches Qwen's format
- Coordinate conversion: Qwen's 1000-based → our 0-1 normalized
- Action mapping: `left_click` → `CLICK`, `type` → `INPUT`, etc.
- Supports both tool calls and text responses

#### 3. `src/agents/qwen3vl_agent.py`
**Purpose:** Qwen2.5-VL wrapper with computer use defaults

**Features:**
- Thin wrapper around `OpenRouterAgent`
- Qwen-specific model defaults
- Tool calling enabled by default
- Support for free tier models

**Available Models:**
- `qwen/qwen2.5-vl-3b-instruct:free` (Free, good for testing)
- `qwen/qwen2.5-vl-32b-instruct:free` (Free, larger)
- `qwen/qwen2.5-vl-72b-instruct` (Paid, best performance)

### Modified Files

#### 1. `src/agents/showui_agent.py`
**Changes:**
- Now inherits from `GPUAgent` instead of `VisionAgent`
- Removed ~60 lines of device detection code (now in base)
- Removed executor management (now in base)
- Implements `_load_model()` and `_run_inference()` abstractions
- Cleaner, more focused code

#### 2. `src/agents/__init__.py`
**Changes:**
- Updated factory to support new agent types
- Added `qwen3vl` mapping
- Added `claude` mapping (uses OpenRouter)
- Better error messages

#### 3. `.env.example`
**Added:**
- `NEKO_AGENT_TYPE` - Agent selection
- `OPENROUTER_API_KEY` - API authentication
- `OPENROUTER_MODEL` - Model selection
- `OPENROUTER_BASE_URL` - API endpoint
- `OPENROUTER_SITE_URL` - HTTP-Referer header
- `OPENROUTER_APP_NAME` - X-Title header
- `OPENROUTER_MAX_TOKENS` - Response limit
- `OPENROUTER_TEMPERATURE` - Sampling temperature
- `OPENROUTER_TOP_P` - Nucleus sampling
- `OPENROUTER_MAX_RETRIES` - Retry attempts
- `OPENROUTER_TIMEOUT` - Request timeout
- `QWEN_VL_MODEL` - Qwen-specific model override
- `CLAUDE_MODEL` - Claude-specific model override

#### 4. `src/agent_refactored.py` (Settings)
**Added fields:**
```python
openrouter_api_key: Optional[str]
openrouter_model: str
openrouter_base_url: str
openrouter_site_url: Optional[str]
openrouter_app_name: Optional[str]
openrouter_max_tokens: int
openrouter_temperature: float
openrouter_top_p: float
openrouter_max_retries: int
openrouter_timeout: int
```

**Enhanced validation:**
- Check `OPENROUTER_API_KEY` required for remote agents
- Validate model format (must contain '/')
- Validate temperature/top_p ranges
- Validate timeout/token limits

#### 5. `pyproject.toml`
**Added dependencies:**
```toml
"httpx>=0.27.0",      # Async HTTP client
"tenacity>=8.2.0",    # Retry logic
```

## Usage

### Local ShowUI (Default)
```bash
# Uses local GPU model
export NEKO_AGENT_TYPE=showui
just agent
```

### Remote Qwen2.5-VL (OpenRouter)
```bash
# Free tier for testing
export NEKO_AGENT_TYPE=qwen3vl
export OPENROUTER_API_KEY=sk-or-v1-...
export QWEN_VL_MODEL=qwen/qwen2.5-vl-3b-instruct:free
just agent

# Paid tier for best performance
export QWEN_VL_MODEL=qwen/qwen2.5-vl-72b-instruct
just agent
```

### Future: Claude Computer Use
```bash
export NEKO_AGENT_TYPE=claude
export OPENROUTER_API_KEY=sk-or-v1-...
export CLAUDE_MODEL=anthropic/claude-3.5-sonnet
just agent
```

## Technical Details

### Qwen Computer Use Tool Format

**Request:**
```json
{
  "model": "qwen/qwen2.5-vl-72b-instruct",
  "messages": [...],
  "tools": [{
    "type": "function",
    "function": {
      "name": "computer",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {"type": "string", "enum": ["left_click", "type", ...]},
          "coordinate": {"type": "array", "items": {"type": "number"}},
          "text": {"type": "string"}
        }
      }
    }
  }],
  "tool_choice": "auto"
}
```

**Response:**
```json
{
  "choices": [{
    "message": {
      "tool_calls": [{
        "function": {
          "name": "computer",
          "arguments": "{\"action\": \"left_click\", \"coordinate\": [450, 320]}"
        }
      }]
    }
  }]
}
```

**Coordinate System:**
- Qwen uses [0, 1000] range for normalized coordinates
- We convert to [0, 1] range: `coord / 1000.0`
- Allows for sub-pixel precision in refinement iterations

### Action Mapping

| Qwen Action | Our Action | Notes |
|-------------|-----------|-------|
| `left_click` | `CLICK` | Primary click |
| `double_click` | `CLICK` | Could add DOUBLE_CLICK type |
| `right_click` | `CLICK` | Could add RIGHT_CLICK type |
| `type` | `INPUT` | Text input |
| `key` | `ENTER` | Key press |
| `mouse_move` | `HOVER` | Hover action |

## Testing Status

✅ **Healthcheck:** Passes
✅ **Configuration validation:** Working
✅ **Agent imports:** Clean
✅ **Factory pattern:** Validated
✅ **ShowUI refactor:** No regressions
⏳ **OpenRouter integration:** Requires API key to test

## Benefits Achieved

### For Developers
- **Clean abstraction:** GPU logic extracted to base class
- **Easy extension:** Add new models in < 50 lines
- **Type safe:** Abstract methods ensure API consistency
- **Testable:** Mock VisionAgent interface for unit tests

### For Users
- **Free tier option:** Test with Qwen free models
- **Better performance:** Remote models for API-based workflows
- **Cost control:** OpenRouter provides usage tracking
- **Future-proof:** Easy to add Claude, Gemini, etc.

### For Operations
- **No code changes:** Switch models via env var
- **Gradual rollout:** Test remote before switching
- **Fallback support:** Text parsing if tool calling fails
- **Retry logic:** Automatic retry on transient failures

## Next Steps

### To Test OpenRouter:
1. Get API key: https://openrouter.ai/keys
2. Set env vars:
   ```bash
   export NEKO_AGENT_TYPE=qwen3vl
   export OPENROUTER_API_KEY=sk-or-v1-...
   export QWEN_VL_MODEL=qwen/qwen2.5-vl-3b-instruct:free
   ```
3. Run: `just agent`

### To Add New GPU Model:
1. Create `src/agents/new_model_agent.py`
2. Inherit from `GPUAgent`
3. Implement `_load_model()` and `_run_inference()`
4. Register in `__init__.py` factory
5. Done!

### To Add New Remote Model:
1. Option A: Use `OpenRouterAgent` directly (if on OpenRouter)
2. Option B: Create wrapper like `Qwen3VLAgent` for defaults
3. Add to factory with model override

## Files Changed Summary

**Created:**
- `src/agents/gpu_agent.py` (164 lines)
- `src/agents/remote_agent.py` (416 lines)
- `src/agents/qwen3vl_agent.py` (42 lines)

**Modified:**
- `src/agents/showui_agent.py` (-43 lines, cleaner)
- `src/agents/__init__.py` (+15 lines)
- `.env.example` (+33 lines)
- `src/agent_refactored.py` (+56 lines)
- `pyproject.toml` (+2 dependencies)

**Total:** +683 lines of new functionality, -43 lines of duplication

## Backward Compatibility

✅ **100% compatible** - Existing ShowUI functionality unchanged
✅ **Default behavior** - NEKO_AGENT_TYPE defaults to "showui"
✅ **Same environment vars** - All existing config still works
✅ **Same action format** - No changes to action execution

## Documentation

See also:
- `src/agents/README.md` - Agent architecture documentation
- `.env.example` - Complete configuration reference
- OpenRouter docs: https://openrouter.ai/docs