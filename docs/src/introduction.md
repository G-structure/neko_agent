# Neko Agent Documentation

Welcome to the **Neko Agent** documentation! This project provides an AI-powered automation agent that interfaces with Neko servers to perform GUI automation tasks through WebRTC connections.

## What is Neko Agent?

Neko Agent is a Python-based automation system that:

- **Connects to Neko servers** via WebRTC for real-time GUI interaction
- **Uses AI vision models** (ShowUI-2B/Qwen2VL) for visual reasoning and action planning  
- **Captures training data** automatically during automation sessions
- **Provides voice synthesis** capabilities via F5-TTS and WebRTC audio
- **Supports various actions** like clicking, typing, scrolling, and navigation

## Key Components

- **`src/neko/`** - Shared library with configuration, logging, WebSocket signaling,
  WebRTC helpers and small utility modules
- **`src/agent.py`** - Core automation agent with WebRTC integration
- **`src/capture.py`** - Training data capture service using MosaicML Streaming
- **`src/yap.py`** - Text-to-speech service with F5-TTS and voice management

## Getting Started

To get started with Neko Agent:

1. **Set up the development environment** using Nix flakes
2. **Configure a Neko server** for GUI automation targets
3. **Run the agent** to begin automation tasks
4. **Optionally enable capture** for training data collection

See the [Getting Started](./user-guide/getting-started.md) guide for detailed setup instructions.

## Architecture

The system is designed around WebRTC-based communication with Neko containers, allowing for:

- Real-time video frame analysis and action execution
- Automated training data collection in MosaicML format
- Voice feedback through WebRTC audio streams
- Scalable deployment patterns for production use

For technical details, see the [Architecture Overview](./developer-guide/architecture.md).