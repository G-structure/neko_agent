# Agent User Guide

The Neko Agent (`agent.py`) is a production-ready AI automation system that uses computer vision and natural language processing to perform GUI automation tasks. It connects to a Neko server via WebRTC and can autonomously navigate web interfaces or mobile apps.

## Overview

The agent uses the ShowUI-2B vision-language model to:
- **See**: Capture screenshots from the target environment
- **Think**: Analyze the visual content and determine next actions
- **Act**: Execute mouse clicks, keyboard inputs, and navigation commands
- **Learn**: Adapt behavior based on task progress and feedback

## Quick Start

### Basic Command

```bash
python src/agent.py --task "Search for weather in Tokyo"
```

### With Neko Server Connection

```bash
python src/agent.py \
  --task "Navigate to GitHub and search for Python projects" \
  --neko-url "http://localhost:8080" \
  --username "user" \
  --password "password"
```

### Keep Agent Running (Online Mode)

```bash
python src/agent.py \
  --task "Check email" \
  --online
```

## Command Line Options

### Required Arguments

| Option | Environment Variable | Description |
|--------|---------------------|-------------|
| `--task` | `NEKO_TASK` | Natural language description of the automation task |

### Connection Options

| Option | Environment Variable | Description |
|--------|---------------------|-------------|
| `--ws` | `NEKO_WS` | Direct WebSocket URL (e.g., `wss://host/api/ws?token=...`) |
| `--neko-url` | `NEKO_URL` | Base URL for REST API login (e.g., `https://neko.example.com`) |
| `--username` | `NEKO_USER` | Username for REST authentication |
| `--password` | `NEKO_PASS` | Password for REST authentication |

### Behavior Options

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `--mode` | `NEKO_MODE` | `web` | Interface mode: `web` or `phone` |
| `--max-steps` | `NEKO_MAX_STEPS` | `8` | Maximum automation steps before stopping |
| `--online` | `NEKO_ONLINE` | `false` | Keep running after task completion |
| `--no-audio` | `NEKO_AUDIO` | `true` | Disable audio stream from Neko |

### Technical Options

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `--metrics-port` | `NEKO_METRICS_PORT`/`PORT` | `9000` | Prometheus metrics server port |
| `--loglevel` | `NEKO_LOGLEVEL` | `INFO` | Logging level (DEBUG, INFO, WARN, ERROR) |
| `--healthcheck` | - | - | Validate configuration and exit |

## Usage Patterns

### Single Task Execution

Run one task and exit when complete:

```bash
python src/agent.py --task "Book a flight from NYC to LAX"
```

### Continuous Operation

Stay online to handle multiple tasks via chat interface:

```bash
python src/agent.py --task "Ready for commands" --online
```

In online mode, send new tasks through the Neko chat interface.

### Phone/Mobile Automation

Switch to mobile interface mode:

```bash
python src/agent.py \
  --mode phone \
  --task "Open Instagram and check notifications"
```

### Direct WebSocket Connection

Skip REST login with direct WebSocket URL:

```bash
python src/agent.py \
  --ws "wss://neko.example.com/api/ws?token=your-token" \
  --task "Your automation task"
```

## Environment Configuration

### Core Settings

```bash
# Task and behavior
export NEKO_TASK="Default task description"
export NEKO_MODE="web"  # or "phone"
export NEKO_MAX_STEPS="15"
export NEKO_ONLINE="false"

# Connection
export NEKO_URL="http://localhost:8080"
export NEKO_USER="admin"
export NEKO_PASS="password"
export NEKO_WS="wss://direct.websocket.url/api/ws?token=..."

# Technical
export NEKO_LOGLEVEL="INFO"
export NEKO_METRICS_PORT="9000"
export NEKO_AUDIO="true"
```

### Advanced Configuration

```bash
# Model settings
export NEKO_REPO_ID="Qwen/Qwen2-VL-2B-Instruct"
export NEKO_SIZE_SHORTEST_EDGE="768"
export NEKO_SIZE_LONGEST_EDGE="1280"

# Network tuning
export NEKO_WS_TIMEOUT="30"
export NEKO_ICE_TIMEOUT="10"
export NEKO_RTCP_KEEPALIVE="1"

# Performance
export NEKO_FORCE_EXIT_GUARD_MS="5000"
export NEKO_LOG_FORMAT="json"  # or "text"
```

## Task Examples

### Web Navigation

```bash
# E-commerce
python src/agent.py --task "Add wireless headphones to Amazon cart"

# Information gathering
python src/agent.py --task "Find the latest news about AI developments"

# Form filling
python src/agent.py --task "Fill out the contact form with test data"
```

### Social Media

```bash
# Content creation
python src/agent.py --task "Post a tweet about climate change"

# Engagement
python src/agent.py --task "Like the latest 5 posts from my Twitter timeline"
```

### Productivity

```bash
# Email management
python src/agent.py --task "Check unread emails and summarize important ones"

# Calendar scheduling
python src/agent.py --task "Schedule a meeting for next Tuesday at 2pm"
```

## Monitoring and Debugging

### Health Check

Validate configuration without running tasks:

```bash
python src/agent.py --healthcheck
```

### Metrics

The agent exposes Prometheus metrics on the configured port:

```bash
curl http://localhost:9000/metrics
```

Key metrics:
- `neko_agent_tasks_total` - Total tasks processed
- `neko_agent_steps_total` - Total automation steps taken
- `neko_agent_errors_total` - Total errors encountered
- `neko_agent_inference_duration_seconds` - AI model inference time

### Debug Logging

Enable detailed logging:

```bash
python src/agent.py --loglevel DEBUG --task "Your task"
```

### Frame Capture

Screenshots and interaction data are saved to `/tmp/neko-agent/` during execution for debugging.

## Integration with Other Services

### Training Data Capture

Run alongside capture service to collect training data:

```bash
# Terminal 1: Start capture
python src/capture.py

# Terminal 2: Run agent
python src/agent.py --task "Navigate and interact for training"
```

### Voice Synthesis

Add voice feedback during automation:

```bash
# Terminal 1: Start TTS service
python src/yap.py

# Terminal 2: Run agent with voice
python src/agent.py --task "Describe actions as you perform them"
```

## Troubleshooting

### Connection Issues

**WebSocket connection failed:**
```bash
# Check Neko server status
curl http://localhost:8080/health

# Verify WebSocket endpoint
python src/agent.py --healthcheck
```

**Authentication errors:**
```bash
# Test REST login manually
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user","password":"pass"}'
```

### Performance Issues

**Slow AI inference:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi -l 1
```

**High memory usage:**
```bash
# Reduce max steps
export NEKO_MAX_STEPS="5"

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

### Automation Problems

**Agent gets stuck:**
- Reduce `--max-steps` to prevent infinite loops
- Enable `DEBUG` logging to see decision process
- Check `/tmp/neko-agent/` for captured screenshots

**Actions not working:**
- Verify `--mode` matches target interface (web vs phone)
- Check network latency between agent and Neko server
- Ensure Neko server has proper input permissions

## Best Practices

### Task Description

Write clear, specific task descriptions:

```bash
# Good
python src/agent.py --task "Navigate to reddit.com, search for 'python tutorials', and save the first 3 post titles"

# Avoid vague descriptions
python src/agent.py --task "do some web stuff"
```

### Resource Management

- Use `--max-steps` to prevent runaway automation
- Enable metrics collection for production deployments
- Monitor log output for error patterns
- Clean up `/tmp/neko-agent/` periodically

### Security

- Use environment variables for credentials
- Avoid passing passwords via command line
- Run agent with minimal required permissions
- Regularly rotate authentication tokens

## Next Steps

- **[Architecture Overview](../developer-guide/architecture.md)** - Understand the system design
- **[Training Data Capture](./capture.md)** - Collect data for model improvement  
- **[Manual Control](./manual-control.md)** - Interactive debugging and control
- **[Developer Guide](../developer-guide/components/agent.md)** - Technical implementation details