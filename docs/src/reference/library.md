# Neko Library API

The `neko` package provides reusable building blocks for command line tools in this repository.  Importing these helpers avoids copy/paste patterns between scripts and keeps behaviour consistent.

## Configuration (`neko.config`)

`Settings` is a lightweight dataclass that captures common runtime options such as the WebSocket URL and log level.  The `Settings.from_env()` classmethod reads these values from the environment so entrypoints can share the same configuration logic.

```python
from neko.config import Settings

settings = Settings.from_env()
print(settings.neko_ws)
```

## Logging (`neko.logging`)

`setup_logging()` centralises logger configuration.  It supports text or JSON output, optional log files, and mutes noisy third‑party loggers.

```python
from neko.logging import setup_logging

logger = setup_logging(level="DEBUG", fmt="json")
logger.info("ready")
```

## Utilities (`neko.utils`)

Small helpers used throughout the codebase:

- `now_iso()` – current UTC timestamp
- `safe_mkdir(path)` – create a directory if missing
- `env_bool(name, default)` – parse boolean environment variables

## WebSocket Client (`neko.websocket`)

`Signaler` handles WebSocket connections with background send/receive tasks, exponential backoff reconnects, and topic‑based fan‑out via a `Broker` instance.  The `LatestOnly` helper stores only the most recent value for a topic.

```python
from neko.websocket import Signaler

signaler = await Signaler("ws://localhost:8000/ws").connect_with_backoff()
await signaler.send({"event": "system/hello"})
```

Consumers can subscribe to queues like `signaler.broker.topic_queue("control")` to receive messages without manual routing logic.

## WebRTC Helpers (`neko.webrtc`)

`build_configuration()` constructs an `RTCConfiguration` with optional TURN support and `create_peer_connection()` returns an `RTCPeerConnection` ready for SDP/ICE handshakes.

```python
from neko.webrtc import build_configuration, create_peer_connection

config = build_configuration("stun:stun.l.google.com:19302")
pc = create_peer_connection(config)
```

These modules are intentionally small but provide a shared foundation for future refactors.
