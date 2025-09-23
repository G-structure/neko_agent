#!/usr/bin/env python3
# src/agent_refactored.py
"""
agent_refactored.py — ShowUI-2B Neko WebRTC GUI automation agent using WebRTCNekoClient.

What it does
------------
- Connects to Neko servers using WebRTCNekoClient from neko_comms
- Processes video frames with ShowUI-2B/Qwen2VL models for visual reasoning
- Executes actions (click, type, scroll, etc.) based on AI model predictions
- Supports both offline (single-task) and online (multi-task via chat) modes
- Emits action annotations to chat for training data capture

Refactored from original agent.py to use modular neko_comms library.
"""

import asyncio
import ast
import json
import logging
import os
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import requests
from PIL import Image, ImageDraw, ImageFont
from metrics import (
    frames_received, actions_executed, parse_errors, navigation_steps,
    inference_latency, reconnects, resize_duration, start_metrics_server
)
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from neko_comms import WebRTCNekoClient, safe_parse_action
from neko_comms.types import ACTION_SPACES
from utils import setup_logging, resize_and_validate_image, save_atomic, draw_action_markers


# ----------------------
# Configuration
# ----------------------
@dataclass
class Settings:
    """Centralized configuration settings loaded from environment variables."""

    # Model configuration
    repo_id: str
    size_shortest_edge: int
    size_longest_edge: int

    # Agent behavior
    max_steps: int
    audio_default: bool
    refinement_steps: int

    # Logging configuration
    log_level: str
    log_file: Optional[str]
    log_format: str

    # Paths and storage
    frame_save_path: Optional[str]
    click_save_path: Optional[str]
    offload_folder: str

    # Metrics configuration
    metrics_port: int

    # Other settings
    run_id: Optional[str]

    @classmethod
    def from_env(cls) -> 'Settings':
        """Load settings from environment variables.

        :return: Settings instance populated from environment variables
        """
        # Handle metrics port: $PORT takes priority over NEKO_METRICS_PORT
        port = os.environ.get("PORT")
        if port is not None:
            try:
                metrics_port = int(port)
            except ValueError:
                metrics_port = int(os.environ.get("NEKO_METRICS_PORT", "9000"))
        else:
            metrics_port = int(os.environ.get("NEKO_METRICS_PORT", "9000"))

        # Handle frame/click save paths - use /tmp/neko-agent/ for relative paths
        frame_save_path = os.environ.get("FRAME_SAVE_PATH")
        click_save_path = os.environ.get("CLICK_SAVE_PATH")

        if frame_save_path and not os.path.isabs(frame_save_path):
            frame_save_path = f"/tmp/neko-agent/{frame_save_path}"

        if click_save_path and not os.path.isabs(click_save_path):
            click_save_path = f"/tmp/neko-agent/{click_save_path}"

        return cls(
            repo_id=os.environ.get("REPO_ID", "showlab/ShowUI-2B"),
            size_shortest_edge=int(os.environ.get("SIZE_SHORTEST_EDGE", "224")),
            size_longest_edge=int(os.environ.get("SIZE_LONGEST_EDGE", "1344")),
            max_steps=int(os.environ.get("NEKO_MAX_STEPS", "8")),
            audio_default=bool(int(os.environ.get("NEKO_AUDIO", "1"))),
            frame_save_path=frame_save_path,
            click_save_path=click_save_path,
            offload_folder=os.environ.get("OFFLOAD_FOLDER", "./offload"),
            refinement_steps=int(os.environ.get("REFINEMENT_STEPS", "5")),
            log_file=os.environ.get("NEKO_LOGFILE"),
            log_level=os.environ.get("NEKO_LOGLEVEL", "INFO"),
            log_format=os.environ.get("NEKO_LOG_FORMAT", "text").lower(),
            metrics_port=metrics_port,
            run_id=os.environ.get("NEKO_RUN_ID"),
        )

    def validate(self) -> List[str]:
        """Validate configuration settings and return list of errors.

        :return: List of validation error messages, empty if valid
        """
        errors = []
        if self.size_shortest_edge <= 0:
            errors.append("SIZE_SHORTEST_EDGE must be positive")
        if self.size_longest_edge <= 0:
            errors.append("SIZE_LONGEST_EDGE must be positive")
        if self.max_steps <= 0:
            errors.append("NEKO_MAX_STEPS must be positive")
        if self.refinement_steps <= 0:
            errors.append("REFINEMENT_STEPS must be positive")
        if self.metrics_port <= 0 or self.metrics_port > 65535:
            errors.append("Metrics port must be between 1 and 65535")
        if self.log_format not in ("text", "json"):
            errors.append("NEKO_LOG_FORMAT must be 'text' or 'json'")
        if self.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append("NEKO_LOGLEVEL must be valid logging level")
        return errors


# ----------------------
# Logging Setup (using utils.setup_logging)
# ----------------------
def setup_agent_logging(settings: Settings) -> logging.Logger:
    """Configure logging and setup directories for the agent.

    :param settings: Configuration settings containing log format, level, and file path
    :return: Configured logger instance
    """
    # Use common logging setup
    logger = setup_logging(settings.log_level, settings.log_format, settings.log_file)

    # Agent-specific directory setup
    agent_logger = logging.getLogger("neko_agent")

    # Log frame/click saving configuration if enabled
    if settings.frame_save_path:
        agent_logger.info("Frame saving enabled: %s", settings.frame_save_path)
        os.makedirs(os.path.dirname(settings.frame_save_path) or '/tmp/neko-agent', exist_ok=True)

    if settings.click_save_path:
        agent_logger.info("Click action saving enabled: %s", settings.click_save_path)
        dir_path = settings.click_save_path if os.path.isdir(settings.click_save_path) else os.path.dirname(settings.click_save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        else:
            os.makedirs('/tmp/neko-agent', exist_ok=True)

    return agent_logger


# ----------------------
# Metrics (imported from metrics module)
# ----------------------


# ----------------------
# Navigation prompt template
# ----------------------
_NAV_SYSTEM = (
    "You are an assistant trained to navigate the {_APP} screen. "
    "Given a task instruction, a screen observation, and an action history sequence, "
    "output the next action and wait for the next observation. "
    "Here is the action space:\n{_ACTION_SPACE}\n"
    "Format the action as a dictionary with the following keys:\n"
    "{{'action': 'ACTION_TYPE', 'value': ..., 'position': ...}}\n"
    "If value or position is not applicable, set as None. "
    "Position might be [[x1,y1],[x2,y2]] for range actions. "
    "Do NOT output extra keys or commentary."
)


# ----------------------
# Utility functions (imported from utils and neko_comms modules)
# ----------------------


# ----------------------
# Main Agent Class
# ----------------------
class NekoAgent:
    """ShowUI-2B Neko WebRTC GUI automation agent using WebRTCNekoClient."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        ws_url: str,
        nav_task: str,
        nav_mode: str,
        settings: Settings,
        logger: logging.Logger,
        max_steps: Optional[int] = None,
        metrics_port: Optional[int] = None,
        audio: Optional[bool] = None,
        online: bool = False,
    ):
        """Initialize the Neko agent.

        :param model: The vision model for inference
        :param processor: The model processor for input preparation
        :param ws_url: WebSocket URL for Neko connection
        :param nav_task: Initial navigation task
        :param nav_mode: Navigation mode ('web' or 'phone')
        :param settings: Configuration settings
        :param logger: Logger instance
        :param max_steps: Maximum navigation steps
        :param metrics_port: Prometheus metrics port
        :param audio: Enable/disable audio stream
        :param online: Keep running after task completion
        """
        self.model = model
        self.processor = processor
        self.ws_url = ws_url
        self.nav_task = nav_task
        self.nav_mode = nav_mode
        self.settings = settings
        self.logger = logger
        self.max_steps = max_steps or settings.max_steps
        self.metrics_port = metrics_port or settings.metrics_port
        self.audio_enabled = audio if audio is not None else settings.audio_default
        self.online = online

        # State
        self.running = True
        self.step_count = 0
        self.action_history: List[Dict[str, Any]] = []
        self.shutdown = asyncio.Event()

        # WebRTC client
        self.client = WebRTCNekoClient(
            ws_url=ws_url,
            auto_host=True,
            request_media=True
        )

        # Executor for model inference
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")

        # Metrics server handles (injected by main)
        self._metrics_server = None
        self._metrics_thread = None

    async def run(self) -> None:
        """Main agent run loop."""
        self.logger.info("Starting Neko agent")

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown)
            except Exception:
                pass

        try:
            # Connect to Neko server
            await self.client.connect()
            self.logger.info("Connected to Neko server")

            # Wait for first frame
            self.logger.info("Waiting for first frame...")
            frame = await self.client.frame_source.wait_for_frame(timeout=30.0)
            if not frame:
                raise RuntimeError("No frame received within 30 seconds")

            self.logger.info("First frame received, starting navigation")

            # Start main loop
            await asyncio.gather(
                self._navigation_loop(),
                self._shutdown_watcher(),
                return_exceptions=True
            )

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error("Agent error: %s", e, exc_info=True)
        finally:
            await self._cleanup()

    def _shutdown(self):
        """Signal handler for graceful shutdown."""
        self.logger.info("Shutdown signal received")
        self.running = False
        self.shutdown.set()

    async def _shutdown_watcher(self) -> None:
        """Wait for shutdown signal and trigger cancellation."""
        await self.shutdown.wait()
        raise asyncio.CancelledError

    async def _navigation_loop(self) -> None:
        """Main navigation loop handling tasks and chat commands."""
        if self.online:
            # Online mode - wait for chat commands
            await self._online_mode()
        else:
            # Offline mode - execute single task
            if self.nav_task.strip():
                await self._execute_task(self.nav_task)
            else:
                self.logger.info("No task provided, entering online mode")
                await self._online_mode()

    async def _online_mode(self) -> None:
        """Online mode - wait for tasks from chat."""
        self.logger.info("Entering online mode - waiting for chat commands")

        while self.running and self.client.is_connected():
            try:
                # Subscribe to chat messages
                msg = await self.client.subscribe_topic("chat", timeout=5.0)
                await self._process_chat_message(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error("Chat processing error: %s", e)
                break

    async def _process_chat_message(self, msg: Dict[str, Any]) -> None:
        """Process a chat message for task commands.

        :param msg: Chat message dictionary
        """
        # Extract text from message
        text = self._extract_text(msg)
        if not text:
            return

        # Check for task command
        if text.startswith("/task "):
            task = text[6:].strip()
            if task:
                self.logger.info("Received task from chat: %s", task)
                await self._execute_task(task)
        elif text.startswith("/stop"):
            self.logger.info("Received stop command from chat")
            self.running = False
        elif text.startswith("/status"):
            status = f"Agent online, step {self.step_count}/{self.max_steps}"
            await self._send_chat(status)

    def _extract_text(self, msg: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a chat message.

        :param msg: Message dictionary
        :return: Text content or None
        """
        event = msg.get("event", "")
        payload = msg.get("payload", {})

        if event == "chat/message":
            content = payload.get("content")
            if isinstance(content, dict):
                return content.get("text")
            return payload.get("text") or payload.get("message")
        elif event in ("send/broadcast", "send/unicast"):
            body = payload.get("body")
            if isinstance(body, str):
                return body
            elif isinstance(body, dict):
                return body.get("text") or body.get("message")

        return None

    async def _execute_task(self, task: str) -> None:
        """Execute a navigation task.

        :param task: Task description to execute
        """
        self.logger.info("Executing task: %s", task)
        self.step_count = 0
        self.action_history = []

        # Build system prompt
        action_space_desc = "\n".join([f"- {a}: {desc}" for a, desc in ACTION_SPACES.get(self.nav_mode, {}).items()])
        system_prompt = _NAV_SYSTEM.format(
            _APP=self.nav_mode,
            _ACTION_SPACE=action_space_desc
        )

        for step in range(self.max_steps):
            if not self.running:
                break

            self.step_count = step + 1
            self.logger.info("Step %d/%d", self.step_count, self.max_steps)

            try:
                # Get current frame
                frame = await self.client.frame_source.get()
                if not frame:
                    self.logger.warning("No frame available, skipping step")
                    continue

                frames_received.inc()

                # Resize frame if needed
                frame = resize_and_validate_image(frame, self.settings.size_longest_edge, self.logger)

                # Save frame if configured
                if self.settings.frame_save_path:
                    save_atomic(frame, self.settings.frame_save_path, self.logger)

                # Generate action via model inference
                action = await self._generate_action(frame, task, system_prompt)
                if not action:
                    self.logger.warning("Failed to generate valid action")
                    continue

                # Execute action
                await self._execute_action(action, frame)

                # Add to history
                self.action_history.append(action)
                navigation_steps.inc()

                # Check if task is complete
                if action.get("action") == "DONE":
                    self.logger.info("Task completed successfully")
                    await self._send_chat(f"Task completed: {task}")
                    break

                # Wait between steps
                await asyncio.sleep(2.0)

            except Exception as e:
                self.logger.error("Step %d failed: %s", self.step_count, e)
                continue

        if self.step_count >= self.max_steps:
            self.logger.warning("Maximum steps reached without completion")
            await self._send_chat(f"Task incomplete after {self.max_steps} steps: {task}")

    async def _generate_action(self, frame: Image.Image, task: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Generate action using the vision model.

        :param frame: Current screen frame
        :param task: Task description
        :param system_prompt: System prompt for the model
        :return: Action dictionary or None if generation fails
        """
        # Build conversation with history
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\n\nCurrent observation:"}
        ]

        # Add action history
        if self.action_history:
            history_text = "Previous actions:\n"
            for i, act in enumerate(self.action_history[-5:], 1):  # Last 5 actions
                history_text += f"{i}. {act}\n"
            conversation.append({"role": "assistant", "content": history_text})

        # Prepare inputs
        try:
            text_input = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(
                text=[text_input],
                images=[frame],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_running_loop()

            def _inference():
                with torch.no_grad():
                    t0 = time.monotonic()
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    inference_latency.observe(time.monotonic() - t0)

                    output_text = self.processor.decode(
                        output_ids[0][len(inputs['input_ids'][0]):],
                        skip_special_tokens=True
                    ).strip()
                    return output_text

            output_text = await loop.run_in_executor(self.executor, _inference)
            self.logger.debug("Model output: %s", output_text)

            # Parse action
            action = safe_parse_action(output_text, self.nav_mode, self.logger, parse_errors)
            return action

        except Exception as e:
            self.logger.error("Inference failed: %s", e)
            return None

    async def _execute_action(self, action: Dict[str, Any], frame: Image.Image) -> None:
        """Execute an action using the action executor.

        :param action: Action dictionary to execute
        :param frame: Current frame for coordinate scaling
        """
        action_type = action.get("action")
        if not action_type:
            return

        try:
            # Execute via WebRTCNekoClient's action executor
            await self.client.action_executor.execute_action(action, frame.size)

            actions_executed.labels(action_type=action_type).inc()
            self.logger.info("Executed action: %s", action)

            # Emit action annotation for training data capture
            await self._emit_action_annotation(action)

            # Save action-marked frame if configured
            if self.settings.click_save_path:
                marked_frame = draw_action_markers(frame, action, self.step_count)
                save_path = f"{self.settings.click_save_path}/step_{self.step_count:03d}_{action_type}.png"
                save_atomic(marked_frame, save_path, self.logger)

        except Exception as e:
            self.logger.error("Failed to execute action %s: %s", action, e)

    async def _emit_action_annotation(self, action: Dict[str, Any]) -> None:
        """Emit action annotation to chat for training data capture.

        :param action: Action dictionary to annotate
        """
        try:
            annotation = {
                "action": action.get("action"),
                "value": action.get("value"),
                "position": action.get("position"),
                "step": self.step_count,
                "timestamp": time.time()
            }

            message = f"Action: {json.dumps(annotation)}"
            await self._send_chat(message)

        except Exception as e:
            self.logger.error("Failed to emit action annotation: %s", e)

    async def _send_chat(self, text: str) -> None:
        """Send a chat message through the client connection.

        :param text: Message text to send
        """
        try:
            await self.client.signaler.send({
                "event": "chat/message",
                "payload": {"text": f"[agent] {text}"}
            })
        except Exception as e:
            self.logger.debug("Failed to send chat message: %s", e)

    async def _cleanup(self) -> None:
        """Clean up resources before shutdown."""
        self.logger.info("Cleaning up agent resources")

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)

        # Disconnect client
        await self.client.disconnect()

        self.logger.info("Agent cleanup complete")


# ----------------------
# Entry Point
# ----------------------
# start_metrics_server is imported from metrics module


async def main() -> None:
    """Main entry point for the Neko agent application."""
    import argparse

    p = argparse.ArgumentParser("neko_agent", description="ShowUI-2B Neko WebRTC agent")
    p.add_argument("--ws", default=os.environ.get("NEKO_WS", None), help="wss://…/api/ws?token=…")
    p.add_argument("--task", default=os.environ.get("NEKO_TASK", "Search the weather"), help="Navigation task")
    p.add_argument("--mode", default=os.environ.get("NEKO_MODE", "web"), choices=list(ACTION_SPACES.keys()), help="Mode: web or phone")
    p.add_argument("--max-steps", type=int, help="Max navigation steps")
    p.add_argument("--metrics-port", type=int, help="Prometheus metrics port")
    p.add_argument("--loglevel", default=os.environ.get("NEKO_LOGLEVEL", "INFO"), help="Logging level")
    p.add_argument("--no-audio", dest="audio", action="store_false", help="Disable audio stream")
    p.add_argument("--online", action="store_true", help="Keep running and wait for chat commands")
    p.add_argument("--neko-url", default=os.environ.get("NEKO_URL", None), help="Base https://host for REST login")
    p.add_argument("--username", default=os.environ.get("NEKO_USER", None), help="REST username")
    p.add_argument("--password", default=os.environ.get("NEKO_PASS", None), help="REST password")
    p.add_argument("--healthcheck", action="store_true", help="Validate configuration and exit")
    p.set_defaults(audio=None)

    args = p.parse_args()

    # Load configuration from environment
    settings = Settings.from_env()

    # Handle --healthcheck flag
    if args.healthcheck:
        errors = settings.validate()
        if errors:
            print("Configuration validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        else:
            print("ok")
            sys.exit(0)

    # Setup logging
    logger = setup_agent_logging(settings)

    # Override log level if specified via CLI
    if args.loglevel:
        logging.getLogger().setLevel(args.loglevel.upper())

    # In online mode, ignore any provided task and start idle
    if args.online:
        if args.task and args.task.strip():
            logger.info("--online specified; ignoring initial --task and waiting for chat tasks.")
        args.task = ""

    # Start metrics server
    metrics_port = args.metrics_port if args.metrics_port is not None else settings.metrics_port
    metrics_server, metrics_thread = start_metrics_server(metrics_port, logger)

    logger.info("Loading model/processor ...")
    device = "cpu"
    dtype = torch.float32

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("CUDA GPU detected: %s (%.1fGB)", gpu_name, gpu_memory)
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16, device="mps")
            dtype = torch.bfloat16
        except RuntimeError:
            dtype = torch.float32
        device = "mps"
        logger.info("Apple MPS detected")
        os.makedirs(settings.offload_folder, exist_ok=True)
    else:
        logger.warning("No GPU acceleration available - using CPU")

    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto"}
    if device == "mps":
        model_kwargs.update({"offload_folder": settings.offload_folder, "offload_state_dict": True})

    model = Qwen2VLForConditionalGeneration.from_pretrained(settings.repo_id, **model_kwargs).eval()
    processor = AutoProcessor.from_pretrained(
        settings.repo_id,
        size={"shortest_edge": settings.size_shortest_edge, "longest_edge": settings.size_longest_edge},
        trust_remote_code=True
    )

    logger.info("Model loaded successfully on device: %s (dtype: %s)", model.device, dtype)

    if device == "cuda":
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        logger.info("GPU memory allocated: %.2fGB", allocated_memory)

    # Determine WebSocket URL
    ws_url = args.ws
    if not ws_url:
        if not (args.neko_url and args.username and args.password):
            p.error("Provide --ws OR all of --neko-url, --username, --password")
        if any(a.startswith("--password") or a.startswith("--username") for a in sys.argv):
            print("[WARN] Consider env vars for secrets.", file=sys.stderr)
        try:
            base = args.neko_url.rstrip("/")
            r = requests.post(f"{base}/api/login", json={"username": args.username, "password": args.password}, timeout=10)
            r.raise_for_status()
            tok = r.json().get("token")
            if not tok:
                raise RuntimeError("REST login ok, but no token in response")
            host = base.split("://", 1)[-1].rstrip("/")
            scheme = "wss" if base.startswith("https") else "ws"
            ws_url = f"{scheme}://{host}/api/ws?token={tok}"
            print(f"[INFO] REST login OK, WS host={host} path=/api/ws", file=sys.stderr)
        except Exception as e:
            print(f"REST login failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif any((args.neko_url, args.username, args.password)):
        print("[WARN] --ws provided; ignoring REST args", file=sys.stderr)

    # Create agent with dependency injection
    agent = NekoAgent(
        model=model,
        processor=processor,
        ws_url=ws_url,
        nav_task=args.task,
        nav_mode=args.mode,
        settings=settings,
        logger=logger,
        max_steps=args.max_steps,
        metrics_port=metrics_port,
        audio=args.audio,
        online=args.online,
    )

    # Inject metrics server handles for clean shutdown
    agent._metrics_server = metrics_server
    agent._metrics_thread = metrics_thread

    try:
        await agent.run()
    finally:
        # Clean up metrics server
        if metrics_server and hasattr(metrics_server, 'shutdown'):
            try:
                metrics_server.shutdown()
                logger.info("Metrics server shut down")
            except Exception as e:
                logger.error("Error shutting down metrics server: %s", e)


if __name__ == "__main__":
    asyncio.run(main())


def cli() -> None:
    """Synchronous console entrypoint wrapper for packaging."""
    asyncio.run(main())