# Training Data Capture User Guide

The Neko capture tool records user interface interactions as training data for machine learning models. It captures screenshots and action sequences from Neko browser sessions, packaging them into datasets ready for training computer vision and UI automation models.

## Quick Start

### Prerequisites

- Running Neko server with WebSocket API enabled
- Python environment with `mosaicml-streaming` package installed
- (Optional) S3-compatible storage credentials for remote upload

### Basic Setup

1. **Install dependencies**:
   ```bash
   pip install streaming requests websockets
   ```

2. **Set your Neko connection**:
   ```bash
   export NEKO_URL="https://your-neko-server.com"
   export NEKO_USER="your-username"
   export NEKO_PASS="your-password"
   ```

3. **Start capturing**:
   ```bash
   python src/capture.py
   ```

4. **Record an episode** in Neko chat:
   ```text
   /start Navigate to login page and enter credentials
   Action: {"type": "click", "coordinate": [150, 200]}
   Action: {"type": "type", "text": "username"}
   /stop
   ```

That's it! Your episode is now saved as training data in `./data/mds/`.

## What Gets Captured

The capture tool records complete "episodes" of UI interaction:

- **Screenshots**: JPEG images captured at regular intervals (default: 2 FPS)
- **Actions**: Structured annotations of user interactions (clicks, typing, etc.)
- **Metadata**: Task descriptions, timestamps, screen dimensions
- **Context**: Complete session information for reproducible training

Each episode becomes a single training sample packaged as:

```text
episode.zip
├── meta.json           # Episode metadata and schema
├── frames/
│   ├── 000000.jpg     # Sequential screenshots
│   ├── 000001.jpg
│   └── ...
├── frames.ndjson      # Frame timing information
└── actions.ndjson     # Action annotations
```

## Architecture

### Data Flow

1. **Connect** to Neko WebSocket API (`/api/ws`) for chat monitoring
2. **Listen** for task boundaries (`/start` and `/stop` commands) and action annotations
3. **Capture** screenshots at configurable FPS via HTTP endpoint (`/api/room/screen/shot.jpg`)
4. **Package** episodes as ZIP archives containing metadata, frames, and action sequences
5. **Write** to MDS shards with automatic S3 upload and shard rotation

### Episode Structure

Each episode is packaged as a ZIP archive containing:

```text
episode.zip
├── meta.json           # Episode metadata and schema
├── frames/
│   ├── 000000.jpg     # Sequential JPEG screenshots
│   ├── 000001.jpg
│   └── ...
├── frames.ndjson      # Frame index with timestamps
└── actions.ndjson     # Action annotations with timestamps
```

## Configuration

All configuration is handled via environment variables for 12-factor compliance:

### Connection Settings

```bash
# REST login (preferred method)
export NEKO_URL="https://neko.example.com"
export NEKO_USER="username"
export NEKO_PASS="password"

# OR direct WebSocket URL (bypasses REST)
export NEKO_WS="wss://neko.example.com/api/ws?token=..."
```

### Output Configuration

```bash
# Local MDS directory
export CAPTURE_OUT="./data/mds"

# Remote storage (optional)
export CAPTURE_REMOTE="s3://bucket/prefix"
export CAPTURE_KEEP_LOCAL=0          # 0=delete local after upload, 1=keep

# MDS shard settings
export CAPTURE_COMPRESSION="zstd:6"   # Compression algorithm
export CAPTURE_SHARD_SIZE="512mb"     # Size before shard rotation
export CAPTURE_HASHES="sha1"          # Integrity checking
```

### Capture Parameters

```bash
export CAPTURE_FPS=2                  # Screenshots per second
export CAPTURE_JPEG_QUALITY=85        # JPEG quality (0-100)
export CAPTURE_MIN_FRAMES=4           # Minimum frames to save episode
export CAPTURE_EPISODE_TIMEOUT=900    # Auto-end after N seconds
```

### S3 Authentication

For S3 or S3-compatible storage:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
export AWS_SESSION_TOKEN="..."        # Optional for temporary credentials

# For S3-compatible endpoints (MinIO, R2, etc.)
export S3_ENDPOINT_URL="https://s3.example.com"
```

## Common Use Cases

### Training UI Automation Models

Capture complete workflows for training models to automate repetitive tasks:

```bash
# Set up for high-quality capture
export CAPTURE_FPS=3
export CAPTURE_JPEG_QUALITY=90
```

Then in Neko chat:
```text
/start Fill out customer registration form
Action: {"type": "click", "coordinate": [245, 156], "element": "first_name_field"}
Action: {"type": "type", "text": "John"}
Action: {"type": "click", "coordinate": [245, 186], "element": "last_name_field"}
Action: {"type": "type", "text": "Doe"}
Action: {"type": "click", "coordinate": [245, 216], "element": "email_field"}
Action: {"type": "type", "text": "john.doe@example.com"}
Action: {"type": "click", "coordinate": [300, 350], "element": "submit_button"}
/stop
```

### Collecting Web Navigation Data

Record browsing sessions for training navigation models:

```bash
# Lower FPS for longer sessions
export CAPTURE_FPS=1
export CAPTURE_EPISODE_TIMEOUT=1800  # 30 minutes
```

Then in Neko chat:
```text
/start Research product reviews for laptop purchase
Action: {"type": "navigate", "url": "https://amazon.com"}
Action: {"type": "type", "text": "gaming laptop", "element": "search_box"}
Action: {"type": "click", "coordinate": [400, 45], "element": "search_button"}
Action: {"type": "click", "coordinate": [200, 180], "element": "product_link"}
Action: {"type": "scroll", "direction": "down", "amount": 500}
/stop
```

### Capturing Error Recovery Workflows

Document how to handle and recover from errors:

In Neko chat:
```text
/start Handle login failure and password reset
Action: {"type": "type", "text": "wrong_password", "element": "password_field"}
Action: {"type": "click", "coordinate": [300, 200], "element": "login_button"}
Action: {"type": "click", "coordinate": [250, 150], "element": "forgot_password_link"}
Action: {"type": "type", "text": "user@example.com", "element": "email_field"}
Action: {"type": "click", "coordinate": [200, 180], "element": "send_reset_button"}
/stop
```

## Configuration Guide

### Basic Configuration

For most users, these environment variables are sufficient:

```bash
# Required: Neko connection
export NEKO_URL="https://your-neko-server.com"
export NEKO_USER="your-username"  
export NEKO_PASS="your-password"

# Optional: Local storage (defaults to ./data/mds)
export CAPTURE_OUT="/path/to/training/data"

# Optional: Capture quality
export CAPTURE_FPS=2              # Screenshots per second
export CAPTURE_JPEG_QUALITY=85    # Image quality (0-100)
```

### Cloud Storage Setup

#### AWS S3

```bash
export CAPTURE_REMOTE="s3://my-training-bucket/ui-episodes"
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

#### MinIO/Self-hosted S3

```bash
export CAPTURE_REMOTE="s3://training-data/episodes"
export S3_ENDPOINT_URL="https://minio.mycompany.com"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
```

#### Cloudflare R2

```bash
export CAPTURE_REMOTE="s3://my-r2-bucket/training"
export S3_ENDPOINT_URL="https://your-account.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="your-r2-token"
export AWS_SECRET_ACCESS_KEY="your-r2-secret"
```

### Advanced Configuration

Fine-tune capture behavior for specific use cases:

```bash
# Episode management
export CAPTURE_MIN_FRAMES=10       # Minimum frames to save episode
export CAPTURE_EPISODE_TIMEOUT=600 # Auto-stop after 10 minutes
export CAPTURE_KEEP_LOCAL=1        # Keep local copies when uploading

# Storage optimization
export CAPTURE_COMPRESSION="zstd:9"  # Maximum compression
export CAPTURE_SHARD_SIZE="1gb"      # Larger shards for fewer files
export CAPTURE_HASHES="sha1,md5"     # Multiple integrity checks

# Debugging
export CAPTURE_LOGLEVEL="DEBUG"
```

## Running the Capture Tool

### Command Line Usage

Basic usage with environment variables:
```bash
python src/capture.py
```

Override settings with command line flags:
```bash
python src/capture.py \
  --neko-url https://neko.example.com \
  --username myuser \
  --password mypass \
  --out ./custom/data/path \
  --remote s3://mybucket/training-data \
  --fps 3.0 \
  --jpeg-quality 95 \
  --episode-timeout 1200
```

### Direct WebSocket Connection

Skip REST authentication with a direct WebSocket URL:
```bash
export NEKO_WS="wss://neko.example.com/api/ws?token=your-jwt-token"
python src/capture.py
```

## Implementation Details

### Core Classes

#### `EpisodeBuffer` (`src/capture.py:161`)
- Manages temporary storage for a single episode
- Handles frame and action storage during capture
- Finalizes episode data into ZIP archive format

#### `NekoSession` (`src/capture.py:342`)
- WebSocket client for Neko server communication
- Handles authentication via REST API
- Manages screenshot polling and message queuing

#### `Capture` (`src/capture.py:542`)
- Main orchestrator coordinating capture workflow
- Processes chat events for episode boundaries
- Manages episode lifecycle and MDS writing

### Episode Lifecycle

1. **Start Detection**: Chat message matching `/start <task>` pattern
2. **Frame Capture**: Screenshot polling thread captures frames at specified FPS
3. **Action Parsing**: Chat messages matching `Action: {...}` are parsed and stored
4. **End Conditions**: 
   - Manual `/stop` command
   - Episode timeout (default 900 seconds)
   - Application shutdown
5. **Finalization**: Episode packaged and written to MDS if minimum frame count met

### MDS Schema

Each MDS record contains:

| Column | Type | Description |
|--------|------|-------------|
| `episode_id` | str | Unique episode identifier |
| `task` | str | Task description from `/start` command |
| `payload` | bytes | ZIP archive containing episode data |
| `num_frames` | int | Number of screenshot frames captured |
| `num_actions` | int | Number of action annotations |
| `started_at` | str | Episode start timestamp (ISO 8601) |
| `ended_at` | str | Episode end timestamp (ISO 8601) |
| `screen_w` | int | Screen width in pixels |
| `screen_h` | int | Screen height in pixels |
| `agent` | json | Agent metadata and configuration |

## Error Handling and Resilience

### Network Resilience
- Automatic WebSocket reconnection with exponential backoff
- Screenshot polling continues through temporary network issues
- MDS writer handles partial uploads and resumes interrupted transfers

### Data Integrity
- SHA1 hashing for shard integrity verification
- Episode validation before MDS writing
- Graceful handling of malformed action annotations

### Resource Management
- Temporary episode directories cleaned up after packaging
- Queue overflow protection prevents memory exhaustion
- Configurable episode timeouts prevent runaway captures

## Integration with Training Pipelines

### Loading Data

```python
from streaming import StreamingDataset
import zipfile
import json

# Create streaming dataset
dataset = StreamingDataset(
    remote="s3://bucket/training-data",
    local="./cache",
    shuffle=True
)

# Process episodes
for sample in dataset:
    episode_id = sample['episode_id']
    task = sample['task']
    payload = sample['payload']
    
    # Extract episode contents
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        meta = json.loads(zf.read('meta.json'))
        frames_index = [json.loads(line) for line in zf.read('frames.ndjson').decode().splitlines()]
        actions = [json.loads(line) for line in zf.read('actions.ndjson').decode().splitlines()]
        
        # Load frame images
        for frame_info in frames_index:
            frame_data = zf.read(frame_info['file'])
            # Process frame...
```

### Random Window Sampling

The episode-as-record format enables efficient random window sampling for sequence models:

```python
def sample_windows(episode_payload, window_size=32):
    with zipfile.ZipFile(io.BytesIO(episode_payload)) as zf:
        frames_index = [json.loads(line) for line in zf.read('frames.ndjson').decode().splitlines()]
        
        if len(frames_index) < window_size:
            return None
            
        start_idx = random.randint(0, len(frames_index) - window_size)
        window = frames_index[start_idx:start_idx + window_size]
        
        # Load frames for this window
        frames = []
        for frame_info in window:
            frame_data = zf.read(frame_info['file'])
            frames.append(frame_data)
            
        return frames
```

## Troubleshooting

### Quick Diagnostics

First, check if the capture tool can connect to your Neko server:

```bash
# Test basic connectivity
curl -I https://your-neko-server.com/api/health

# Test WebSocket endpoint (if available)
python -c "
import websockets
import asyncio
asyncio.run(websockets.connect('wss://your-neko-server.com/api/ws'))
print('WebSocket connection successful')
"
```

### Common Issues & Solutions

#### ❌ Connection Issues

**Problem**: `Connection refused` or `Unable to connect to Neko server`

**Solutions**:
1. Verify Neko server is running: `curl https://your-neko-server.com`
2. Check firewall settings - ensure ports 80/443 and WebSocket ports are open
3. Verify URL format: `https://` (not `http://`) for secure connections
4. Test from the same network as your Neko server first

**Problem**: `SSL certificate verification failed`

**Solutions**:
1. For self-signed certificates, set: `export PYTHONHTTPSVERIFY=0` (development only)
2. Or add certificate to system trust store
3. Use IP address instead of hostname if DNS issues

#### ❌ Authentication Issues

**Problem**: `Authentication failed` or `401 Unauthorized`

**Solutions**:
1. Double-check username/password: `echo $NEKO_USER $NEKO_PASS`
2. Verify user has WebSocket API access in Neko admin panel
3. Try direct WebSocket token if available:
   ```bash
   # Get token via REST API first
   curl -X POST https://your-neko-server.com/api/login \
     -H "Content-Type: application/json" \
     -d '{"username":"user","password":"pass"}'
   
   # Use token directly
   export NEKO_WS="wss://your-neko-server.com/api/ws?token=YOUR_TOKEN"
   ```

#### ❌ Episode Recording Issues

**Problem**: Episodes not being saved/captured

**Solutions**:
1. **Check command format**: Commands must be exact:
   ```text
   /start task description here    ✅
   /Start task description         ❌ (wrong case)
   / start task description        ❌ (extra space)
   ```

2. **Verify action format**: Actions must be valid JSON:
   ```text
   Action: {"type": "click", "coordinate": [100, 200]}    ✅
   Action: {type: "click", coordinate: [100, 200]}        ❌ (missing quotes)
   Action {"type": "click"}                               ❌ (missing colon)
   ```

3. **Check minimum frames**: Episodes with fewer than `CAPTURE_MIN_FRAMES` are discarded:
   ```bash
   export CAPTURE_MIN_FRAMES=1  # Save all episodes
   ```

4. **Enable debug logging**:
   ```bash
   export CAPTURE_LOGLEVEL=DEBUG
   python src/capture.py
   ```

#### ❌ Storage Issues

**Problem**: `Permission denied` writing to local directory

**Solutions**:
1. Check directory permissions: `ls -la ./data/`
2. Create directory manually: `mkdir -p ./data/mds`
3. Use a different path: `export CAPTURE_OUT=/tmp/capture-data`

**Problem**: S3 upload failures

**Solutions**:
1. **Verify credentials**:
   ```bash
   aws sts get-caller-identity  # Test AWS credentials
   ```

2. **Check bucket permissions**: Ensure your credentials can write to the bucket

3. **Test endpoint connectivity**:
   ```bash
   # For MinIO/R2
   curl -I $S3_ENDPOINT_URL
   ```

4. **Debug with minimal config**:
   ```bash
   unset CAPTURE_REMOTE  # Disable S3, use local only
   python src/capture.py
   ```

#### ❌ Performance Issues

**Problem**: High memory usage or slow performance

**Solutions**:
1. **Reduce capture rate**:
   ```bash
   export CAPTURE_FPS=1              # Lower frame rate
   export CAPTURE_JPEG_QUALITY=70    # Lower image quality
   ```

2. **Shorter episodes**:
   ```bash
   export CAPTURE_EPISODE_TIMEOUT=300  # 5 minutes max
   ```

3. **Monitor resources**:
   ```bash
   # Watch memory usage
   watch 'ps aux | grep capture.py'
   
   # Check disk space
   df -h ./data/mds/
   ```

### Debugging Commands

#### View Current Configuration
```bash
python src/capture.py --help  # See all options
env | grep CAPTURE            # Show capture settings
env | grep NEKO               # Show connection settings
```

#### Test Components Individually

**Test REST Authentication**:
```bash
python -c "
import requests, os
r = requests.post(f'{os.getenv(\"NEKO_URL\")}/api/login',
                 json={'username': os.getenv('NEKO_USER'),
                       'password': os.getenv('NEKO_PASS')})
print(f'Status: {r.status_code}')
print(f'Response: {r.text}')
"
```

**Test Screenshot Endpoint**:
```bash
python -c "
import requests, os
r = requests.get(f'{os.getenv(\"NEKO_URL\")}/api/room/screen/shot.jpg')
print(f'Status: {r.status_code}')
print(f'Content-Type: {r.headers.get(\"content-type\")}')
with open('/tmp/test_screenshot.jpg', 'wb') as f:
    f.write(r.content)
print('Screenshot saved to /tmp/test_screenshot.jpg')
"
```

### Log Analysis

Enable detailed logging to diagnose issues:

```bash
export CAPTURE_LOGLEVEL=DEBUG
python src/capture.py 2>&1 | tee capture.log
```

**Key log messages to look for**:

- `REST login ok` - Authentication successful
- `Screen size: 1920x1080` - WebSocket connection established
- `Episode [id] started` - Episode recording began
- `Episode [id] committed` - Episode saved successfully
- `WS loop error` - WebSocket connection issues
- `Shot.jpg non-OK` - Screenshot endpoint problems

### Getting Help

If you're still having issues:

1. **Check the logs** with `CAPTURE_LOGLEVEL=DEBUG`
2. **Verify your environment** with `env | grep -E "(NEKO|CAPTURE|AWS)"`
3. **Test components separately** using the debug commands above
4. **Create a minimal test case**:
   ```bash
   # Simplest possible configuration
   export NEKO_URL="https://your-server.com"
   export NEKO_USER="testuser"
   export NEKO_PASS="testpass"
   export CAPTURE_OUT="/tmp/test-capture"
   export CAPTURE_LOGLEVEL="DEBUG"
   
   python src/capture.py
   ```

### Performance Tuning

For optimal performance in different scenarios:

#### High-Quality Training Data
```bash
export CAPTURE_FPS=3
export CAPTURE_JPEG_QUALITY=95
export CAPTURE_COMPRESSION="zstd:3"  # Faster compression
export CAPTURE_SHARD_SIZE="256mb"    # Smaller shards for faster upload
```

#### Long Sessions/Low Memory
```bash
export CAPTURE_FPS=1
export CAPTURE_JPEG_QUALITY=70
export CAPTURE_EPISODE_TIMEOUT=1800
export CAPTURE_COMPRESSION="zstd:9"  # Maximum compression
```

#### Local Development
```bash
unset CAPTURE_REMOTE              # No S3 upload
export CAPTURE_COMPRESSION="none" # Faster local writes
export CAPTURE_LOGLEVEL="INFO"
```

## Best Practices

### Episode Design

**Keep episodes focused**: Record single, complete tasks rather than mixing multiple workflows:
```bash
# Good: focused task
/start Complete user registration process
# ... perform registration steps ...
/stop

# Poor: mixed tasks  
/start Do registration then check email then browse products
```

**Use descriptive task names**: Help your training pipeline understand the data:
```bash
/start Handle payment form validation errors
/start Navigate product catalog with filters
/start Recover from session timeout during checkout
```

**Include error scenarios**: Capture both success and failure paths:
```bash
/start Login with invalid credentials and recover
/start Handle network interruption during file upload
/start Deal with form validation errors
```

### Action Annotation Guidelines

**Be consistent with action types**: Use standardized action schemas:
```json
{"type": "click", "coordinate": [x, y], "element": "button_id"}
{"type": "type", "text": "input text", "element": "field_id"}  
{"type": "scroll", "direction": "down", "amount": 300}
{"type": "navigate", "url": "https://example.com"}
{"type": "wait", "duration": 2.0, "reason": "page_load"}
```

**Include context in actions**: Add semantic information when possible:
```json
{"type": "click", "coordinate": [200, 100], "element": "login_button", "intent": "submit_form"}
{"type": "type", "text": "user@example.com", "element": "email_field", "intent": "enter_credentials"}
```

### Storage Management

**Choose appropriate compression**: Balance speed vs storage:
- Development: `CAPTURE_COMPRESSION="none"` (fastest)
- Production: `CAPTURE_COMPRESSION="zstd:6"` (balanced)
- Archive: `CAPTURE_COMPRESSION="zstd:9"` (smallest)

**Optimize shard sizes** for your infrastructure:
- Fast networks: `CAPTURE_SHARD_SIZE="1gb"` (fewer files)
- Slow networks: `CAPTURE_SHARD_SIZE="128mb"` (faster uploads)
- Mobile/edge: `CAPTURE_SHARD_SIZE="64mb"` (reliable transfers)

**Use appropriate retention policies**:
```bash
# Keep local copies for immediate access
export CAPTURE_KEEP_LOCAL=1

# Or upload and clean for space efficiency  
export CAPTURE_KEEP_LOCAL=0
```

## Advanced Usage

### Continuous Capture Workflows

For long-running capture sessions, use systemd or docker for reliability:

**Systemd service** (`/etc/systemd/system/neko-capture.service`):
```ini
[Unit]
Description=Neko Training Data Capture
After=network.target

[Service]
Type=simple
User=capture
WorkingDirectory=/opt/neko-capture
Environment=NEKO_URL=https://neko.example.com
Environment=NEKO_USER=capture-bot
EnvironmentFile=/etc/neko-capture/config
ExecStart=/usr/bin/python3 src/capture.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

**Docker deployment**:
```dockerfile
FROM python:3.11-slim
RUN pip install streaming requests websockets
COPY src/capture.py /app/
ENV NEKO_URL=https://neko.example.com
CMD ["python", "/app/capture.py"]
```

### Multi-instance Scaling

Run multiple capture instances for parallel data collection:

```bash
# Instance 1: UI automation tasks
export CAPTURE_OUT="./data/ui-automation"
export CAPTURE_REMOTE="s3://training/ui-automation"
python src/capture.py &

# Instance 2: Navigation tasks  
export CAPTURE_OUT="./data/navigation"
export CAPTURE_REMOTE="s3://training/navigation"  
python src/capture.py &

# Instance 3: Error handling
export CAPTURE_OUT="./data/error-recovery"
export CAPTURE_REMOTE="s3://training/error-recovery"
python src/capture.py &
```

### Integration with Training Pipelines

**Streaming data loader example**:
```python
from streaming import StreamingDataset
import torch
from torch.utils.data import DataLoader

class UIDataset(StreamingDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Extract and process episode
        payload = sample['payload']
        episode = self.process_episode(payload)
        
        return {
            'frames': episode['frames'],
            'actions': episode['actions'], 
            'task': sample['task']
        }

# Use with PyTorch
dataset = UIDataset(remote="s3://training/episodes")
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### Quality Assurance

**Validate episodes before training**:
```python
def validate_episode(episode_data):
    """Check episode quality before using for training"""
    frames = episode_data['frames']
    actions = episode_data['actions']
    
    # Check minimum length
    if len(frames) < 10:
        return False, "Too few frames"
        
    # Check action/frame ratio
    if len(actions) / len(frames) < 0.1:
        return False, "Too few actions relative to frames"
        
    # Check for valid coordinates
    for action in actions:
        if action.get('type') == 'click':
            coord = action.get('coordinate', [])
            if not (0 <= coord[0] <= 1920 and 0 <= coord[1] <= 1080):
                return False, "Invalid click coordinates"
                
    return True, "Valid episode"
```

**Monitor capture quality**:
```bash
# Check recent episodes
python -c "
from streaming import StreamingDataset
ds = StreamingDataset(local='./data/mds')
for i, sample in enumerate(ds):
    if i >= 10: break
    print(f'Episode {sample[\"episode_id\"]}: {sample[\"num_frames\"]} frames, {sample[\"num_actions\"]} actions')
"
```
