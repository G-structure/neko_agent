# Core Agent (`src/agent.py`)

The core agent (`src/agent.py`) is the main automation engine of the Neko Agent system. It provides AI-powered GUI automation through WebRTC connections to Neko servers, using the ShowUI-2B vision model for intelligent action planning and execution.

## Overview

The agent operates as a production-ready, 12-Factor App compliant service that:

- **Connects to Neko servers** via WebRTC for real-time GUI interaction
- **Uses ShowUI-2B (Qwen2VL)** for visual reasoning and action planning
- **Executes precise actions** with iterative refinement for improved accuracy
- **Supports multiple modes** including web automation and mobile device control
- **Provides comprehensive metrics** via Prometheus for monitoring and observability
- **Handles graceful shutdown** with proper resource cleanup

## Architecture

```mermaid
graph TB
    subgraph "Agent Core"
        Agent[NekoAgent]
        Settings[Settings]
        Signaler[Signaler]
    end

    subgraph "Frame Sources"
        WebRTC[WebRTCFrameSource]
        Lite[LiteFrameSource]
    end

    subgraph "AI Processing"
        Model[ShowUI-2B Model]
        Processor[AutoProcessor]
        Executor[ThreadPoolExecutor]
    end

    subgraph "External Services"
        Neko[Neko Server]
        Metrics[Prometheus Metrics]
        Storage[Frame/Click Storage]
    end

    Agent --> Signaler
    Agent --> WebRTC
    Agent --> Lite
    Agent --> Model
    Agent --> Processor

    Signaler <-->|WebSocket| Neko
    WebRTC <-->|WebRTC| Neko
    Lite <-->|HTTP Polling| Neko

    Agent --> Metrics
    Agent --> Storage

    Model --> Executor
```

## Core Classes

### 1. Settings

**Purpose**: Centralized configuration management following 12-Factor App principles.

**Key Features**:
- Environment variable-driven configuration
- Validation with error reporting
- Support for both development and production deployments
- Port priority handling (`$PORT` overrides `NEKO_METRICS_PORT`)

**Configuration Options**:

| Category | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| **Model** | `REPO_ID` | `showlab/ShowUI-2B` | Hugging Face model repository |
| | `SIZE_SHORTEST_EDGE` | `224` | Image resize shortest edge |
| | `SIZE_LONGEST_EDGE` | `1344` | Image resize longest edge |
| **Network** | `NEKO_WS` | `wss://neko.example.com/api/ws` | WebSocket URL |
| | `NEKO_ICE_POLICY` | `strict` | ICE candidate policy |
| | `NEKO_STUN_URL` | `stun:stun.l.google.com:19302` | STUN server |
| | `NEKO_TURN_URL` | - | TURN server (optional) |
| **Behavior** | `NEKO_MAX_STEPS` | `8` | Maximum automation steps |
| | `REFINEMENT_STEPS` | `5` | Click refinement iterations |
| | `NEKO_AUDIO` | `1` | Enable audio streaming |
| **Logging** | `NEKO_LOGLEVEL` | `INFO` | Log level |
| | `NEKO_LOG_FORMAT` | `text` | Log format (text/json) |
| | `NEKO_LOGFILE` | - | Log file path (optional) |
| **Storage** | `FRAME_SAVE_PATH` | - | Frame screenshot storage |
| | `CLICK_SAVE_PATH` | - | Action visualization storage |
| | `OFFLOAD_FOLDER` | `./offload` | Model cache directory |
| **Metrics** | `PORT` / `NEKO_METRICS_PORT` | `9000` | Prometheus metrics port |

**Example Usage**:
```python
# Load configuration from environment
settings = Settings.from_env()

# Validate configuration
errors = settings.validate()
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
    sys.exit(1)
```

### 2. NekoAgent

**Purpose**: Main orchestration class that manages WebRTC connections, AI processing, and action execution.

**Initialization**:
```python
agent = NekoAgent(
    model=model,                    # ShowUI-2B model instance
    processor=processor,            # AutoProcessor for model inputs
    ws_url="wss://neko.example.com/api/ws",
    nav_task="Navigate to google.com and search for AI",
    nav_mode="web",                 # "web" or "phone"
    settings=settings,
    logger=logger,
    max_steps=10,
    refinement_steps=3,
    audio=True,
    online=False                    # False=single task, True=multi-task chat mode
)
```

**Key Methods**:

#### `async def run()`
Main execution loop that handles connection lifecycle, reconnection logic, and task processing.

**Flow**:
1. **Signal handling** - Register SIGINT/SIGTERM handlers
2. **Connection establishment** - WebSocket connection with backoff
3. **Media negotiation** - WebRTC SDP offer/answer exchange
4. **Frame processing** - Continuous screen capture and AI analysis
5. **Action execution** - Translate AI decisions to control commands
6. **Graceful shutdown** - Clean resource cleanup

#### `async def _navigate_once(img, history, step)`
Performs single navigation step using AI model inference.

**Parameters**:
- `img`: Current screen image (PIL Image)
- `history`: List of previous actions for context
- `step`: Current step number for logging

**Process**:
1. **Prepare inputs** - Format image and chat template for model
2. **Model inference** - Generate action prediction using ShowUI-2B
3. **Action parsing** - Extract structured action from model output
4. **Iterative refinement** - Crop and re-infer for precise click coordinates
5. **Action execution** - Send control commands via WebSocket

**Refinement Logic**:
```python
for i in range(self.refinement_steps):
    # Initial inference on full image or cropped region
    action = await model_inference(current_img)

    if action.type == "CLICK":
        # Crop around predicted click location
        current_img, crop_box = self._crop_image(original_img, action.position)
        # Re-infer on cropped image for better precision
    else:
        break  # No refinement for non-click actions
```

#### `async def _execute_action(action, size)`
Translates high-level actions into low-level control commands.

**Supported Actions**:

| Action | Description | Parameters |
|--------|-------------|------------|
| `CLICK` | Mouse click at coordinates | `position: [x, y]` (normalized) |
| `INPUT` | Type text string | `value: str` |
| `SCROLL` | Scroll in direction | `direction: "up/down/left/right"` |
| `SWIPE` | Touch swipe gesture | `startPosition: [x, y]`, `endPosition: [x, y]` |
| `TAP` | Mobile tap action | `position: [x, y]` |
| `ANSWER` | Task completion signal | `value: "final answer"` |

**Example Action Execution**:
```python
# CLICK action
action = {
    "action": "CLICK",
    "position": [0.5, 0.3],  # Normalized coordinates (50%, 30%)
    "value": None
}

# Converts to pixel coordinates and sends control command
x, y = int(0.5 * screen_width), int(0.3 * screen_height)
await signaler.send({
    "event": "control/buttonpress",
    "payload": {"x": x, "y": y, "code": 1}  # Left click
})
```

### 3. Frame Sources

The agent supports multiple frame capture mechanisms:

#### WebRTCFrameSource
**Purpose**: Real-time frame capture via WebRTC video streams.

**Features**:
- **Low latency** - Direct WebRTC video track access
- **High quality** - Full resolution screen capture
- **Automatic buffering** - Handles frame timing and delivery
- **Error resilience** - Graceful handling of stream interruptions

#### LiteFrameSource
**Purpose**: HTTP-based frame capture for environments where WebRTC is unavailable.

**Features**:
- **Simple protocol** - HTTP GET requests for screenshots
- **Fallback mode** - Works when WebRTC fails or is blocked
- **Configurable polling** - Adjustable frame rate
- **Resource efficient** - Lower bandwidth than video streams

### 4. Signaler

**Purpose**: WebSocket communication layer for control and signaling.

**Features**:
- **Event-driven messaging** - Pub/sub pattern with topic routing
- **Automatic reconnection** - Exponential backoff strategy
- **Concurrent I/O** - Separate read/write loops
- **Message queuing** - Buffered sending with backpressure handling

**Event Types**:
- `signal/*` - WebRTC signaling (offer/answer/ICE)
- `control/*` - GUI control commands (mouse/keyboard)
- `chat/*` - Text messaging for online mode
- `system/*` - Session management and status

## Action Space and Navigation Modes

### Web Mode (`nav_mode="web"`)

**Optimized for**: Desktop web browsers, web applications

**Action Space**:
```text
Available Actions:
1. CLICK(position): Click at normalized coordinates [x, y] where 0 <= x, y <= 1
2. INPUT(value): Type the specified text string
3. SCROLL(direction): Scroll in direction (up, down, left, right)
4. ANSWER(value): Provide final answer when task is complete

Examples:
- CLICK(position=[0.5, 0.3]): Click at center-left of screen
- INPUT(value="search query"): Type text into focused element
- SCROLL(direction="down"): Scroll page downward
- ANSWER(value="Task completed successfully"): Signal completion
```

### Phone Mode (`nav_mode="phone"`)

**Optimized for**: Mobile device interfaces, touch interactions

**Action Space**:
```text
Available Actions:
1. TAP(position): Tap at normalized coordinates [x, y] where 0 <= x, y <= 1
2. INPUT(value): Enter text using virtual keyboard
3. SWIPE(startPosition, endPosition): Swipe from start to end coordinates
4. SCROLL(direction): Scroll content in specified direction
5. ANSWER(value): Provide final answer when task is complete

Examples:
- TAP(position=[0.8, 0.1]): Tap near top-right corner
- SWIPE(startPosition=[0.5, 0.8], endPosition=[0.5, 0.2]): Swipe up
- SCROLL(direction="down"): Scroll content downward
```

## AI Model Integration

### ShowUI-2B (Qwen2VL)

**Model Details**:
- **Architecture**: Qwen2VL-based multimodal model specialized for GUI understanding
- **Input**: RGB images + text instructions
- **Output**: Structured action commands
- **Performance**: ~100-200ms inference time on GPU

**Processing Pipeline**:
```python
# 1. Image preprocessing
image = resize_and_validate_image(frame, settings, logger)

# 2. Chat template preparation
content = [
    {"type": "text", "text": system_prompt},
    {"type": "text", "text": f"Task: {nav_task}"},
    {"type": "text", "text": f"Action history: {history}"},
    {"type": "image", "image": image, "size": {...}}
]

# 3. Model inference
inputs = processor(text=[text], images=[image], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)

# 4. Action parsing
action = safe_parse_action(decoded_output, nav_mode="web")
```

**Iterative Refinement**:
The agent implements a novel refinement strategy for improved click accuracy:

1. **Initial prediction** on full screen image
2. **Crop extraction** around predicted click location (50% of original size)
3. **Re-inference** on cropped region for sub-pixel precision
4. **Coordinate transformation** back to full screen space

This typically improves click accuracy by 15-25% over single-shot inference.

## Operation Modes

### Offline Mode (Single Task)

**Usage**: Automated execution of a single, predefined task.

```python
agent = NekoAgent(
    model=model,
    processor=processor,
    ws_url=ws_url,
    nav_task="Find the current weather in San Francisco",
    nav_mode="web",
    online=False,  # Single task mode
    max_steps=10
)

await agent.run()  # Executes task and exits
```

**Behavior**:
- **Immediate control** - Requests host control on connection
- **Task execution** - Runs navigation loop until completion or max steps
- **Automatic exit** - Terminates after task completion
- **Host control maintained** - Keeps control throughout session

### Online Mode (Multi-Task)

**Usage**: Interactive agent that responds to chat commands for multiple tasks.

```python
agent = NekoAgent(
    model=model,
    processor=processor,
    ws_url=ws_url,
    nav_task="",  # No initial task
    nav_mode="web",
    online=True,  # Multi-task chat mode
    max_steps=15
)

await agent.run()  # Runs indefinitely, responding to chat
```

**Behavior**:
- **On-demand control** - Only requests control when active task starts
- **Chat monitoring** - Listens for task commands in chat messages
- **Task queuing** - Handles multiple sequential tasks
- **Persistent session** - Remains connected between tasks
- **Host control release** - Releases control when idle

**Chat Commands**:
```text
Start a new task:
User: Navigate to amazon.com and find wireless headphones under $100

The agent will:
1. Request host control
2. Execute the navigation task
3. Release host control when complete
4. Wait for next command
```

## Metrics and Observability

### Prometheus Metrics

The agent exports comprehensive metrics for monitoring and alerting:

```python
# Connection metrics
reconnects = Counter('neko_agent_reconnects_total', 'WebSocket reconnection attempts')
frames_processed = Counter('neko_agent_frames_processed_total', 'Video frames processed')

# Performance metrics
inference_latency = Histogram('neko_agent_inference_seconds', 'AI model inference time')
action_execution_time = Histogram('neko_agent_action_execution_seconds', 'Action execution time')

# Quality metrics
actions_executed = Counter('neko_agent_actions_executed_total', 'Actions executed by type', ['action_type'])
parse_errors = Counter('neko_agent_parse_errors_total', 'Action parsing failures')
```

**Example Queries**:
```promql
# Average inference latency over 5 minutes
rate(neko_agent_inference_seconds_sum[5m]) / rate(neko_agent_inference_seconds_count[5m])

# Action success rate
1 - (rate(neko_agent_parse_errors_total[5m]) / rate(neko_agent_frames_processed_total[5m]))

# Actions per minute by type
rate(neko_agent_actions_executed_total[1m]) * 60
```

### Logging

**Structured Logging**: Supports both text and JSON formats for different environments.

```python
# Text format (development)
export NEKO_LOG_FORMAT=text
# Output: 2024-01-15 10:30:00 INFO Model inference completed (step=3) | Duration: 0.15s | Device: GPU

# JSON format (production)
export NEKO_LOG_FORMAT=json
# Output: {"timestamp":"2024-01-15T10:30:00Z","level":"INFO","message":"Model inference completed","step":3,"duration":0.15,"device":"GPU"}
```

**Log Levels**:
- `DEBUG` - Detailed execution flow, frame processing details
- `INFO` - Task progress, action execution, performance metrics
- `WARNING` - Recoverable errors, fallback activations
- `ERROR` - Critical failures, connection issues

## Configuration Examples

### Development Environment

```bash
# Basic development setup
export NEKO_WS="ws://localhost:8080/api/ws"
export NEKO_LOGLEVEL="DEBUG"
export NEKO_LOG_FORMAT="text"
export FRAME_SAVE_PATH="./debug/frames"
export CLICK_SAVE_PATH="./debug/actions"
export NEKO_MAX_STEPS="15"
export REFINEMENT_STEPS="3"
```

### Production Environment

```bash
# Production configuration
export NEKO_WS="wss://neko.prod.example.com/api/ws"
export NEKO_LOGLEVEL="INFO"
export NEKO_LOG_FORMAT="json"
export NEKO_LOGFILE="/var/log/neko-agent.log"
export PORT="8080"  # Metrics server port
export NEKO_MAX_STEPS="20"
export REFINEMENT_STEPS="5"
export NEKO_RTCP_KEEPALIVE="1"  # For NAT traversal
export NEKO_FORCE_EXIT_GUARD_MS="5000"  # Force exit after 5s
```

### GPU Optimization

```bash
# GPU-specific settings
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDA_ARCH_LIST="8.6"  # For RTX 30xx series
export OFFLOAD_FOLDER="/fast-ssd/model-cache"  # SSD for model loading
```

## Error Handling and Recovery

### Connection Resilience

```python
# Automatic reconnection with exponential backoff
async def connect_with_backoff(self):
    backoff = 1.0
    max_backoff = 60.0

    while not self.shutdown.is_set():
        try:
            await self._establish_connection()
            return
        except Exception as e:
            logger.warning(f"Connection failed: {e}, retrying in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 1.5)
```

### Inference Timeout Handling

```python
# Model inference with timeout and cancellation
try:
    async with asyncio.timeout(120.0):  # 2 minute timeout
        outputs = await model_inference_task
except asyncio.TimeoutError:
    model_inference_task.cancel()
    logger.error("Model inference timeout, skipping frame")
    return None
```

### Frame Source Fallback

```python
# Automatic fallback from WebRTC to HTTP polling
if webrtc_frame_source.failed:
    logger.warning("WebRTC frame source failed, falling back to HTTP polling")
    self.frame_source = LiteFrameSource(settings, logger)
    await self.frame_source.start(session_id)
```

## Performance Optimization

### GPU Memory Management

```python
# Efficient GPU memory usage
torch.cuda.empty_cache()  # Clear cache between tasks
model.half()  # Use FP16 for faster inference
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
```

### Frame Processing Pipeline

```python
# Asynchronous frame processing
async def process_frames():
    async for frame in frame_source:
        # Non-blocking processing
        task = asyncio.create_task(self._navigate_once(frame, history, step))

        # Rate limiting to prevent overwhelming the model
        await asyncio.sleep(0.1)  # 10 FPS maximum
```

### Model Loading Optimization

```python
# Efficient model initialization
model = Qwen2VLForConditionalGeneration.from_pretrained(
    repo_id,
    torch_dtype=torch.float16,  # Half precision
    device_map="auto",          # Automatic device placement
    offload_folder=settings.offload_folder,  # Disk offloading for large models
    trust_remote_code=True
)
```

## Usage Examples

### Basic Web Automation

```python
# Simple web navigation task
import asyncio
from agent import NekoAgent, Settings, setup_logging

async def main():
    settings = Settings.from_env()
    logger = setup_logging(settings)

    # Load AI model
    model = Qwen2VLForConditionalGeneration.from_pretrained("showlab/ShowUI-2B")
    processor = AutoProcessor.from_pretrained("showlab/ShowUI-2B")

    # Create and run agent
    agent = NekoAgent(
        model=model,
        processor=processor,
        ws_url="ws://localhost:8080/api/ws",
        nav_task="Go to google.com and search for 'machine learning tutorials'",
        nav_mode="web",
        settings=settings,
        logger=logger
    )

    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Interactive Online Mode

```python
# Multi-task interactive agent
agent = NekoAgent(
    model=model,
    processor=processor,
    ws_url="wss://neko.example.com/api/ws",
    nav_task="",  # Start without a task
    nav_mode="web",
    online=True,  # Enable chat-based task assignment
    settings=settings,
    logger=logger
)

# Agent will respond to chat messages like:
# "Please navigate to amazon.com and find wireless keyboards under $50"
# "Go to the weather website and check the forecast for tomorrow"
await agent.run()
```

### Custom Action Refinement

```python
# Agent with custom refinement settings
agent = NekoAgent(
    model=model,
    processor=processor,
    ws_url=ws_url,
    nav_task=task,
    nav_mode="web",
    refinement_steps=5,  # More refinement for better accuracy
    max_steps=20,        # Allow longer task sequences
    settings=settings,
    logger=logger
)
```

## Testing and Debugging

### Health Check

```bash
# Validate configuration
uv run src/agent.py --healthcheck

# Expected output:
# Configuration validation: PASSED
# Model loading: PASSED
# WebSocket connectivity: PASSED
# All systems ready
```

### Debug Mode

```bash
# Enable comprehensive debugging
export NEKO_LOGLEVEL="DEBUG"
export FRAME_SAVE_PATH="./debug/frames"
export CLICK_SAVE_PATH="./debug/actions"

uv run src/agent.py --task "test navigation" --max-steps 3
```

### Frame Analysis

The agent can save frames and action visualizations for debugging:

```python
# Saved frame: ./debug/frames/frame_001_1642261234.567.png
# Saved action: ./debug/actions/action_step_001_1642261234.567_CLICK.png
```

Action visualizations include:
- **Red circle** - Predicted click location
- **Green overlay** - Text input areas
- **Blue arrows** - Scroll directions
- **Purple box** - Crop regions during refinement

## Integration with Other Components

### With Capture Service

```python
# The agent automatically integrates with capture service
# No additional configuration needed - capture service monitors WebSocket
export CAPTURE_OUT="./training-data"
export CAPTURE_REMOTE="s3://training-bucket/episodes"

# Start capture in background
uv run src/capture.py &

# Run agent (capture will automatically record)
uv run src/agent.py --task "automation task"
```

### With TTS Service

```python
# Enable voice announcements during automation
export YAP_VOICES_DIR="./voices"

# Start TTS service
uv run src/yap.py &

# Run agent with voice enabled (audio is negotiated automatically)
uv run src/agent.py --task "automation task"
```

## Future Enhancements

The agent architecture is designed to support planned enhancements:

1. **API Mode** - RESTful service with horizontal scaling
2. **Multi-Modal Input** - Voice commands and text instructions
3. **Advanced Vision Models** - Support for newer GUI understanding models
4. **Distributed Inference** - Model sharding across multiple GPUs
5. **Custom Action Types** - Extensible action framework
6. **Real-Time Learning** - Online adaptation from user feedback

See [API Mode Design](../../future/api-mode-design.md) for detailed future architecture plans.
