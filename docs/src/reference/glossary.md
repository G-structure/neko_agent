# Glossary

## General Terms

**Neko Server**
: A containerized remote desktop solution that provides browser-based access to desktop environments via WebRTC.

**WebRTC**
: Web Real-Time Communication protocol used for peer-to-peer audio, video, and data transmission in web browsers.

**ICE (Interactive Connectivity Establishment)**
: Protocol for establishing peer-to-peer connections across NATs and firewalls.

**SDP (Session Description Protocol)**
: Protocol for describing media sessions and connection parameters.

## Component-Specific Terms

### Manual Control CLI

**REPL (Read-Eval-Print Loop)**
: Interactive command-line interface for real-time control and testing.

**Signaler**
: WebSocket connection manager that handles message routing and reconnection.

**Broker**
: Event distribution system that routes incoming messages to appropriate topic queues.

**Normalized Coordinates**
: Coordinate system using 0.0-1.0 range instead of pixel values for resolution independence.

**Host Control**
: Exclusive mouse and keyboard control permission on the Neko server.

### Core Agent

**Action Space**
: Set of possible actions the agent can perform (click, type, scroll, etc.).

**Observation Space**
: Visual and contextual information the agent receives from the environment.

**Training Data**
: Collected sequences of observations and actions used for model training.

### Capture Service

**MDS (MosaicML Dataset)**
: Efficient dataset format optimized for streaming and distributed training.

**Trajectory**
: Sequence of state-action pairs representing a complete task execution.

**Temporal Alignment**
: Synchronization of visual frames with corresponding action timestamps.

### TTS Service

**Voice Synthesis**
: Process of generating spoken audio from text input.

**Audio Streaming**
: Real-time transmission of audio data over WebRTC channels.

**Voice Activity Detection (VAD)**
: Technology for detecting presence of human speech in audio.

## Technical Terms

**Async/Await**
: Python asynchronous programming pattern for concurrent execution.

**Task Cancellation**
: Graceful shutdown mechanism for asyncio background tasks.

**Exponential Backoff**
: Retry strategy with increasing delays between attempts.

**Event Loop**
: Core of asyncio that manages and executes asynchronous operations.

**Queue (asyncio.Queue)**
: Thread-safe queue implementation for passing data between async tasks.

## Protocol Terms

**Event Payload**
: Data structure containing the actual content of WebSocket messages.

**Heartbeat**
: Periodic messages sent to maintain connection and detect timeouts.

**Session ID**
: Unique identifier for each client connection to the Neko server.

**Control Protocol**
: Set of message types and formats for remote desktop interaction.

## Development Terms

**mdBook**
: Documentation generator that creates books from Markdown files.

**PEP 257**
: Python Enhancement Proposal defining docstring conventions.

**PEP 287**
: Python Enhancement Proposal defining reStructuredText format for docstrings.

**Type Hints**
: Python annotations that specify expected types for function parameters and return values.