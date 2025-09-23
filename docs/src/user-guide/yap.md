# Voice Synthesis (YAP)

YAP (Yet Another Presenter) provides real-time text-to-speech capabilities for your Neko automation sessions. Convert text messages into natural-sounding speech that plays directly in the browser.

## What is YAP?

YAP transforms text into speech using advanced AI voice synthesis (F5-TTS) and streams the audio through WebRTC to your Neko browser session. This enables:

- **Voice announcements** during automation tasks
- **Interactive conversations** through chat commands  
- **Live narration** of automation steps
- **Multiple voice personas** with custom characteristics

## Quick Start

### 1. Prerequisites

Before using YAP, ensure you have:

- A running Neko server (see [Getting Started](./getting-started.md))
- GPU environment for optimal performance: `nix develop .#gpu`
- Basic voice files in the `./voices` directory

### 2. Start YAP Service

```bash
# Connect to local Neko server
export NEKO_URL="http://localhost:8080"
export NEKO_USER="user" 
export NEKO_PASS="password"

uv run src/yap.py
```

You should see:
```text
[12:34:56] yap INFO - WS connected
[12:34:56] yap INFO - RTC answer sent; audio track live.
[12:34:56] yap INFO - Voices reloaded (1 entries)
```

### 3. Test Voice Output

In your Neko browser chat, type:

```text
/yap Hello! I am your voice assistant.
```

You should hear the text spoken through the browser audio.

## Voice Commands

YAP responds to commands in the Neko chat interface:

### Immediate Speech

Speak text immediately:

```text
/yap Good morning! Ready to start automation.
/yap The task has been completed successfully.
```

### Streaming Mode

For longer conversations or live narration:

```text
/yap:begin
I'm starting the automation task now...
Navigating to the website...
Filling out the form...
Submitting the data...
/yap:end
```

In streaming mode, YAP processes text incrementally as you type, enabling natural conversation flow.

### Stop/Clear Queue

Cancel current speech and clear the queue:

```text
/yap:stop
```

## Voice Management

### Default Voice Setup

YAP needs at least one voice configured. Create a basic setup:

1. **Create voices directory:**
   ```bash
   mkdir -p voices
   ```

2. **Add a reference audio file:**
   ```bash
   # Record or copy a 3-10 second WAV file
   cp your-voice-sample.wav voices/default.wav
   ```

3. **YAP will auto-create voices.json** on first run with default settings.

### Adding New Voices

Add voices through chat commands:

```text
/yap:voice add --spk alice --ref ./voices/alice.wav --ref-text "Hello, my name is Alice" --styles "friendly,calm"
```

Parameters:
- `--spk`: Voice ID/name
- `--ref`: Path to reference audio file (WAV format, 3-10 seconds)
- `--ref-text`: Transcript of the reference audio
- `--styles`: Comma-separated style tags
- `--rate`: Speech speed (0.5-2.0, default 1.0)
- `--pitch`: Pitch shift in semitones (-12 to +12, default 0.0)

### Switching Voices

Change the active voice and parameters:

```text
/yap:voice set --spk alice
/yap:voice set --spk bob --rate 1.2 --pitch -0.5
/yap:voice set --rate 0.8
```

### Reload Voice Configuration

After manually editing `voices/voices.json`:

```text
/yap:voice reload
```

## Configuration

### Basic Settings

Set these environment variables before starting YAP:

```bash
# Connection (choose one method)
export YAP_WS="wss://demo.neko.com/api/ws?token=your_token"
# OR
export NEKO_URL="https://demo.neko.com"
export NEKO_USER="username"
export NEKO_PASS="password"

# Voice directory
export YAP_VOICES_DIR="./voices"
export YAP_SPK_DEFAULT="default"
```

### Audio Quality

```bash
# Audio format (recommended settings)
export YAP_SR=48000              # Sample rate (Hz)
export YAP_AUDIO_CHANNELS=1      # Channels (1=mono, 2=stereo)
export YAP_FRAME_MS=20           # WebRTC frame size

# Processing
export YAP_PARALLEL=2            # TTS worker threads
export YAP_MAX_CHARS=350         # Max characters per chunk
export YAP_OVERLAP_MS=30         # Audio crossfade overlap
```

### Performance Tuning

```bash
# For faster response (lower quality)
export YAP_MAX_CHARS=200
export YAP_PARALLEL=4

# For better quality (higher latency)  
export YAP_MAX_CHARS=500
export YAP_OVERLAP_MS=50

# Buffer management
export YAP_JITTER_MAX_SEC=6.0    # Audio buffer size
```

## Voice Configuration File

YAP stores voice settings in `voices/voices.json`:

```json
{
  "default": {
    "ref_audio": "./voices/default.wav",
    "ref_text": "This is my default voice sample.",
    "styles": ["neutral"],
    "params": {
      "rate": 1.0,
      "pitch": 0.0
    }
  },
  "alice": {
    "ref_audio": "./voices/alice.wav", 
    "ref_text": "Hello, my name is Alice and I sound friendly.",
    "styles": ["friendly", "energetic"],
    "params": {
      "rate": 1.1,
      "pitch": 0.2
    }
  },
  "narrator": {
    "ref_audio": "./voices/narrator.wav",
    "ref_text": "I will be narrating the automation process.",
    "styles": ["professional", "clear"],
    "params": {
      "rate": 0.9,
      "pitch": -0.3
    }
  }
}
```

### Voice Parameters

- **ref_audio**: Path to reference WAV file (3-10 seconds, clear speech)
- **ref_text**: Exact transcript of the reference audio
- **styles**: Descriptive tags (friendly, professional, calm, energetic)
- **rate**: Speech speed multiplier (0.5=slow, 1.0=normal, 2.0=fast)
- **pitch**: Pitch adjustment in semitones (negative=lower, positive=higher)

## Usage Scenarios

### Automation Announcements

Start YAP alongside your automation agent:

```bash
# Terminal 1: Start YAP
uv run src/yap.py

# Terminal 2: Run automation with voice (audio is enabled by default)
uv run src/agent.py --task "Fill out contact form"
```

The agent can announce progress:
```text
/yap Starting automation task: Fill out contact form
/yap Navigating to the website...
/yap Form submitted successfully!
```

### Interactive Sessions

Use YAP for live interaction during manual control:

```bash
# Terminal 1: Start YAP
uv run src/yap.py

# Terminal 2: Manual control
python src/manual.py
```

Then control both automation and voice through chat:
```text
!click 100 200
/yap Clicked on the submit button
!type "Hello World"
/yap Entered text in the field
```

### Multi-Voice Conversations

Set up different voices for different purposes:

```bash
# Set up voices
/yap:voice add --spk system --ref ./voices/system.wav --ref-text "System notification" --rate 0.9
/yap:voice add --spk user --ref ./voices/user.wav --ref-text "User interaction" --rate 1.1

# Use in conversation
/yap:voice set --spk system
/yap System initialization complete.

/yap:voice set --spk user  
/yap Thank you for the update!
```

## Troubleshooting

### No Audio Output

**Check browser audio permissions:**
1. Click the browser's address bar lock icon
2. Ensure "Sound" is allowed
3. Check browser volume settings

**Verify WebRTC connection:**
- Open browser developer tools (F12)
- Go to Console tab
- Look for WebRTC connection messages
- Check for audio stream indicators

**Test connection:**
```bash
uv run src/yap.py --healthcheck
```

### Poor Audio Quality

**Check reference audio:**
- Use high-quality WAV files (16-bit, 22kHz+)
- 3-10 second samples with clear speech
- No background noise or music
- Single speaker only

**Adjust processing:**
```bash
# Increase overlap for smoother transitions
export YAP_OVERLAP_MS=50

# Reduce chunk size for lower latency
export YAP_MAX_CHARS=250
```

### High Latency

**Optimize for speed:**
```bash
# Increase parallel workers
export YAP_PARALLEL=4

# Reduce chunk size
export YAP_MAX_CHARS=200

# Reduce buffer size
export YAP_JITTER_MAX_SEC=3.0
```

**Check GPU usage:**
```bash
# Monitor GPU usage while YAP is running
nvidia-smi -l 1
```

### Connection Issues

**WebSocket connection failed:**
```bash
# Test WebSocket endpoint
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  http://localhost:8080/api/ws
```

**Authentication failed:**
```bash
# Test REST login
curl -X POST http://localhost:8080/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"password"}'
```

**Check firewall/network:**
- Ensure ports 8080 (HTTP) and WebRTC ports are accessible
- Test with STUN server connectivity
- Check for corporate proxy/firewall blocking WebRTC

### Debug Mode

Enable detailed logging:

```bash
export YAP_LOGLEVEL=DEBUG
export YAP_LOG_FORMAT=json
uv run src/yap.py 2>&1 | jq .
```

Look for:
- WebSocket connection status
- WebRTC negotiation progress  
- TTS processing times
- Audio buffer status

## Advanced Usage

### Custom Voice Training

For better voice quality, record multiple reference samples:

1. **Record varied samples:**
   ```bash
   # Different emotions/styles
   voices/alice-happy.wav
   voices/alice-serious.wav
   voices/alice-excited.wav
   ```

2. **Test different samples:**
   ```text
   /yap:voice set --spk alice-happy
   /yap I'm so excited about this automation!
   
   /yap:voice set --spk alice-serious  
   /yap Please review these results carefully.
   ```

### Integration with Automation

Modify automation scripts to include voice feedback:

```python
# In your automation script
import requests

def announce(text):
    """Send voice announcement to YAP"""
    requests.post(f"{neko_url}/api/chat", json={
        "message": f"/yap {text}"
    })

# Use in automation
announce("Starting login process")
agent.click_login_button()
announce("Login successful, proceeding to dashboard")
```

### Docker Deployment

Deploy YAP as a container service:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Install Python dependencies  
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application and voices
COPY src/ /app/src/
COPY voices/ /app/voices/
WORKDIR /app

# Configuration
ENV YAP_VOICES_DIR=/app/voices
ENV YAP_SR=48000
ENV YAP_PARALLEL=2

ENTRYPOINT ["python", "src/yap.py"]
```

## Next Steps

- **Learn about [Training Data Capture](./capture.md)** to improve voice models
- **Explore [Core Agent](../developer-guide/components/agent.md)** for automation integration
- **Read [TTS Service Technical Details](../developer-guide/components/yap.md)** for advanced configuration
- **Check [Neko Integration](../developer-guide/neko.md)** for server setup options

## Related Guides

- [Getting Started](./getting-started.md) - Initial system setup
- [Training Data Capture](./capture.md) - Data collection for improvements  
- [Manual Control CLI](../developer-guide/components/manual.md) - Interactive testing
- [Architecture Overview](../developer-guide/architecture.md) - System design