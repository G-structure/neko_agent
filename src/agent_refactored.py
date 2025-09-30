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
import contextlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from metrics import (
    frames_received, actions_executed, parse_errors, navigation_steps,
    inference_latency, reconnects, resize_duration, start_metrics_server
)

from agents import VisionAgent
from agents.parsing import safe_parse_action
from neko_comms import WebRTCNekoClient
from neko_comms.types import ACTION_SPACES, ACTION_SPACE_DESC, Action as ActionModel
from utils import setup_logging, resize_and_validate_image, save_atomic, draw_action_markers


# ----------------------
# Helpers
# ----------------------
def _coerce_bool(value: str) -> bool:
    try:
        return bool(strtobool(value))
    except (ValueError, AttributeError):
        return value.strip().lower() in {"1", "true", "yes", "on"}


# ----------------------
# Configuration
# ----------------------
@dataclass
class Settings:
    """Centralized configuration settings loaded from environment variables."""

    # Model configuration
    agent_type: str
    repo_id: str
    size_shortest_edge: int
    size_longest_edge: int

    # Network configuration
    default_ws: Optional[str]
    neko_ice_policy: str
    neko_stun_url: str
    neko_turn_url: Optional[str]
    neko_turn_user: Optional[str]
    neko_turn_pass: Optional[str]

    # Agent behavior
    max_steps: int
    audio_default: bool
    refinement_steps: int
    inference_timeout: float
    neko_rtcp_keepalive: bool
    neko_skip_initial_frames: int
    force_exit_guard_ms: int

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

    # OpenRouter configuration
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
    openrouter_reasoning_enabled: bool
    openrouter_reasoning_effort: Optional[str]

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

        audio_raw = os.environ.get("NEKO_AUDIO", "1")
        try:
            audio_default = bool(strtobool(audio_raw))
        except (AttributeError, ValueError):
            audio_default = audio_raw.strip().lower() not in {"0", "false", "off", "no", ""}

        return cls(
            agent_type=os.environ.get("NEKO_AGENT_TYPE", "showui"),
            repo_id=os.environ.get("REPO_ID", "showlab/ShowUI-2B"),
            size_shortest_edge=int(os.environ.get("SIZE_SHORTEST_EDGE", "224")),
            size_longest_edge=int(os.environ.get("SIZE_LONGEST_EDGE", "1344")),
            default_ws=os.environ.get("NEKO_WS"),
            neko_ice_policy=os.environ.get("NEKO_ICE_POLICY", "strict"),
            neko_stun_url=os.environ.get("NEKO_STUN_URL", "stun:stun.l.google.com:19302"),
            neko_turn_url=os.environ.get("NEKO_TURN_URL"),
            neko_turn_user=os.environ.get("NEKO_TURN_USER"),
            neko_turn_pass=os.environ.get("NEKO_TURN_PASS"),
            max_steps=int(os.environ.get("NEKO_MAX_STEPS", "8")),
            audio_default=audio_default,
            frame_save_path=frame_save_path,
            click_save_path=click_save_path,
            offload_folder=os.environ.get("OFFLOAD_FOLDER", "./offload"),
            refinement_steps=int(os.environ.get("REFINEMENT_STEPS", "5")),
            inference_timeout=float(os.environ.get("NEKO_INFERENCE_TIMEOUT", "120")),
            neko_rtcp_keepalive=_coerce_bool(os.environ.get("NEKO_RTCP_KEEPALIVE", "0")),
            neko_skip_initial_frames=int(os.environ.get("NEKO_SKIP_INITIAL_FRAMES", "5")),
            force_exit_guard_ms=int(os.environ.get("NEKO_FORCE_EXIT_GUARD_MS", "0")),
            log_file=os.environ.get("NEKO_LOGFILE"),
            log_level=os.environ.get("NEKO_LOGLEVEL", "INFO"),
            log_format=os.environ.get("NEKO_LOG_FORMAT", "text").lower(),
            metrics_port=metrics_port,
            run_id=os.environ.get("NEKO_RUN_ID"),
            # OpenRouter
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "qwen/qwen2.5-vl-72b-instruct"),
            openrouter_base_url=os.environ.get(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1/chat/completions"
            ),
            openrouter_site_url=os.environ.get("OPENROUTER_SITE_URL"),
            openrouter_app_name=os.environ.get("OPENROUTER_APP_NAME", "Neko Agent"),
            openrouter_max_tokens=int(os.environ.get("OPENROUTER_MAX_TOKENS", "512")),
            openrouter_temperature=float(os.environ.get("OPENROUTER_TEMPERATURE", "0.0")),
            openrouter_top_p=float(os.environ.get("OPENROUTER_TOP_P", "1.0")),
            openrouter_max_retries=int(os.environ.get("OPENROUTER_MAX_RETRIES", "3")),
            openrouter_timeout=int(os.environ.get("OPENROUTER_TIMEOUT", "60")),
            openrouter_reasoning_enabled=_coerce_bool(os.environ.get("OPENROUTER_REASONING_ENABLED", "1")),
            openrouter_reasoning_effort=os.environ.get("OPENROUTER_REASONING_EFFORT"),
        )

    def validate(self) -> List[str]:
        """Validate configuration settings and return list of errors.

        :return: List of validation error messages, empty if valid
        """
        errors = []
        if self.agent_type not in ("showui", "claude", "qwen3vl"):
            errors.append("NEKO_AGENT_TYPE must be 'showui', 'claude', or 'qwen3vl'")
        if self.size_shortest_edge <= 0:
            errors.append("SIZE_SHORTEST_EDGE must be positive")
        if self.size_longest_edge <= 0:
            errors.append("SIZE_LONGEST_EDGE must be positive")
        if self.max_steps <= 0:
            errors.append("NEKO_MAX_STEPS must be positive")
        if self.refinement_steps <= 0:
            errors.append("REFINEMENT_STEPS must be positive")
        if self.inference_timeout <= 0:
            errors.append("NEKO_INFERENCE_TIMEOUT must be positive")
        if self.neko_skip_initial_frames < 0:
            errors.append("NEKO_SKIP_INITIAL_FRAMES must be >= 0")
        if self.neko_ice_policy not in ("strict", "all"):
            errors.append("NEKO_ICE_POLICY must be 'strict' or 'all'")
        if self.metrics_port <= 0 or self.metrics_port > 65535:
            errors.append("Metrics port must be between 1 and 65535")
        if self.log_format not in ("text", "json"):
            errors.append("NEKO_LOG_FORMAT must be 'text' or 'json'")
        if self.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append("NEKO_LOGLEVEL must be valid logging level")

        # OpenRouter validation
        if self.agent_type in ("qwen3vl", "claude"):
            if not self.openrouter_api_key:
                errors.append(
                    f"OPENROUTER_API_KEY required for agent_type={self.agent_type}. "
                    f"Get your key at https://openrouter.ai/keys"
                )
            if "/" not in self.openrouter_model:
                errors.append(
                    "OPENROUTER_MODEL must be in format 'provider/model-name' "
                    "(e.g., 'qwen/qwen2.5-vl-72b-instruct')"
                )
            if self.openrouter_max_tokens <= 0:
                errors.append("OPENROUTER_MAX_TOKENS must be positive")
            if not (0.0 <= self.openrouter_temperature <= 2.0):
                errors.append("OPENROUTER_TEMPERATURE must be between 0.0 and 2.0")
            if not (0.0 <= self.openrouter_top_p <= 1.0):
                errors.append("OPENROUTER_TOP_P must be between 0.0 and 1.0")
            if self.openrouter_timeout <= 0:
                errors.append("OPENROUTER_TIMEOUT must be positive")

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
#TODO: this prompt is specific to showui and we should be able to set other _NAV_SYSTEM prompts, in fact our code should become more flexiable
#        to other prompt formats, right now it is constraned to this _NAV_SYSTEM prompt regardless of the model being used, this must be fixed.
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
    """Neko WebRTC GUI automation agent using pluggable vision models."""

    def __init__(
        self,
        vision_agent: VisionAgent,
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

        :param vision_agent: The vision agent for action generation
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
        self.vision_agent = vision_agent
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
        self.action_history: List[Dict[str, Any]] = []   #TODO: the action history across this codebase should also include FRAMES that where used to generate the action.
        self.shutdown = asyncio.Event()

        # Online mode task coordination
        self._new_task_event = asyncio.Event()
        self._is_running_task = False
        self._pending_task: Optional[str] = None

        # WebRTC client configuration mirrors the legacy agent defaults
        ice_servers: List[Any] = []
        if self.settings.neko_stun_url:
            ice_servers.append(self.settings.neko_stun_url)
        if self.settings.neko_turn_url:
            turn_entry: Dict[str, Any] = {"urls": self.settings.neko_turn_url}
            if self.settings.neko_turn_user and self.settings.neko_turn_pass:
                turn_entry["username"] = self.settings.neko_turn_user
                turn_entry["credential"] = self.settings.neko_turn_pass
            ice_servers.append(turn_entry)

        self.client = WebRTCNekoClient(
            ws_url=ws_url,
            ice_servers=ice_servers,
            ice_policy=self.settings.neko_ice_policy,
            enable_audio=self.audio_enabled,
            rtcp_keepalive=self.settings.neko_rtcp_keepalive,
            auto_host=True,
            request_media=True,
        )

        # Metrics server handles (injected by main)
        self._metrics_server = None
        self._metrics_thread = None

    async def run(self) -> None:
        """Main agent run loop."""
        self.logger.info("Starting Neko agent")
        if self.settings.run_id:
            self.logger.info("Run ID: %s", self.settings.run_id)

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

            # Wait for media to arrive before starting navigation
            self.logger.info("Waiting for first frame...")
            if not await self.client.wait_for_track(timeout=30.0):
                raise RuntimeError("No video track established within 30 seconds")

            first_frame = await self.client.wait_for_frame(timeout=20.0)
            if not first_frame:
                raise RuntimeError("No frame received within 20 seconds of video track")

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
                self.logger.info("Offline mode; requesting clean shutdown.")
                self.shutdown.set()
            else:
                self.logger.info("No task provided, entering online mode")
                await self._online_mode()

    async def _online_mode(self) -> None:
        """Online mode - wait for tasks from chat."""
        self.logger.info("Entering online mode - waiting for chat commands")

        # Start chat message consumer as background task
        async def chat_consumer():
            while self.running and self.client.is_connected():
                try:
                    msg = await self.client.subscribe_topic("chat", timeout=5.0)
                    await self._process_chat_message(msg)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error("Chat processing error: %s", e)
                    break

        # Run chat consumer in background and main task loop
        try:
            await asyncio.gather(
                chat_consumer(),
                self._task_loop(),
                return_exceptions=True
            )
        except Exception as e:
            self.logger.error("Online mode error: %s", e)

    async def _task_loop(self) -> None:
        """Main task execution loop for online mode."""
        while self.running and self.client.is_connected():
            # Wait for a task to be assigned
            ready_announced = False
            while not (self.nav_task and self.nav_task.strip()) and not self.shutdown.is_set():
                if not ready_announced:
                    await self._send_chat("Ready. Send a task in chat to begin.")
                    ready_announced = True
                try:
                    await asyncio.wait_for(self._new_task_event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    continue
                finally:
                    self._new_task_event.clear()

            if self.shutdown.is_set():
                break

            # Execute the task
            await self._send_chat(f"Starting task: {self.nav_task}")
            self._is_running_task = True

            try:
                await self._execute_task(self.nav_task)
            finally:
                self._is_running_task = False

            await self._send_chat(f"Completed task: {self.nav_task}")

            # Handle any pending task or clear and wait for next
            if self._pending_task:
                self.nav_task = self._pending_task
                self._pending_task = None
                # Continue loop to start next task immediately
            else:
                self.nav_task = ""
                # Loop will wait for next task

#TODO: there are some parts to this which is custom for the agent_refactored.py file such as the specifc slash command "task" we are looking for, but I think this logic is
#         likely duplicated in other files (and we need to minimize code duplication) such as yap_refactored.py. Think about where this should live, and how it should be flexable
#         to support all useage of similar logic found elsewhere in this codebase.
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
                if self.online and self._is_running_task:
                    self._pending_task = task
                    self.logger.info("Task queued until current run completes.")
                else:
                    self.nav_task = task
                    if self.online:
                        self._new_task_event.set()
            else:
                await self._send_chat("Usage: /task <instruction>")
        elif text.startswith("/stop"):
            self.logger.info("Received stop command from chat")
            self.running = False
            self.shutdown.set()
        elif text.startswith("/status"):
            status = f"Agent online, step {self.step_count}/{self.max_steps}"
            await self._send_chat(status)

#TODO: similar to _process_chat_message this bit of logic is likely duplicated across other files so we should find a way to refactor to minimize codeduplication and make a better library.
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
        action_space_desc = ACTION_SPACE_DESC.get(
            self.nav_mode, ACTION_SPACE_DESC.get("web", "")
        )
        system_prompt = _NAV_SYSTEM.format(
            _APP=self.nav_mode,
            _ACTION_SPACE=action_space_desc
        )

        await self._drain_initial_frames()

        # Validate screen size is initialized before starting navigation
        screen_size = self.client.frame_size
        if screen_size == (0, 0):
            self.logger.warning(
                "Screen size not yet received from server. "
                "Coordinate scaling may be incorrect until system/init or screen/updated event is received."
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

                # Log frame size vs screen size for debugging
                screen_size = self.client.frame_size
                self.logger.debug("Frame size: %s, Screen size: %s", frame.size, screen_size)

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

                # Execute action using remote screen size for proper coordinate scaling
                await self._execute_action(action)

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

    async def _drain_initial_frames(self) -> None:
        """Discard initial frames to mirror legacy warmup behaviour."""

        skip = self.settings.neko_skip_initial_frames
        if skip <= 0:
            return

        for _ in range(skip):
            try:
                frame = await self.client.frame_source.wait_for_frame(timeout=2.0)
            except Exception:
                break
            if not frame:
                break

    async def _generate_action(self, frame: Image.Image, task: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Generate action using the vision agent with iterative refinement for clicks.

        Note: Refinement works in frame coordinate space. The normalized coordinates [0-1]
        returned by this method are resolution-independent and will be scaled to screen
        coordinates during action execution.
        """

        # Use frame size for cropping - we work in frame coordinate space here
        full_size = frame.size
        crop_box = (0, 0, full_size[0], full_size[1])
        final_action: Optional[Dict[str, Any]] = None

        for iteration in range(self.settings.refinement_steps):
            current_frame = frame.crop(crop_box)

            # Save refinement iteration frame if configured
            if self.settings.click_save_path and iteration > 0:
                try:
                    refine_path = f"{self.settings.click_save_path}/step_{self.step_count:03d}_refine_{iteration}.png"
                    save_atomic(current_frame, refine_path, self.logger)
                    self.logger.debug("Saved refinement frame %d: %s", iteration, refine_path)
                except Exception as e:
                    self.logger.warning("Failed to save refinement frame: %s", e)

            # Delegate to vision agent for inference
            output_text = await self.vision_agent.generate_action(
                image=current_frame,
                task=task,
                system_prompt=system_prompt,
                action_history=self.action_history,
                crop_box=crop_box,
                iteration=iteration,
                full_size=full_size,
            )

            if not output_text:
                break

            self.logger.debug("Model output: %s", output_text)
            action = safe_parse_action(output_text, self.nav_mode, self.logger, parse_errors)
            if not action:
                self.logger.warning("Failed to generate valid action")
                break

            final_action = action
            action_type = action.get("action")

            #TODO this needs to parse or make sure we are parsing all DONE style actions
            if action_type == "DONE":
                break

            #TODO I would like to use another prompt for refinement so the vllm agent knows what is going on. We should matain a full chain so that the agent see the refiment frames and messages in the history (action history)
            #        maybe this means extending action history to support sub actions such as refinement steps.
            # Refinement only applies to actions that produce a point position
            if self._action_has_point_position(action):
                normalized = self._normalize_point_position(action, crop_box, full_size)
                if normalized:
                    final_action["position"] = normalized
                    self.logger.info(
                        "Refinement iteration %d/%d for %s: normalized position = %s",
                        iteration + 1,
                        self.settings.refinement_steps,
                        action_type,
                        normalized,
                    )

                    # Continue refining if we have more iterations
                    if iteration < self.settings.refinement_steps - 1:
                        next_box = self._compute_refinement_box(normalized, crop_box, full_size)
                        if next_box:
                            crop_box = next_box
                            self.logger.debug("Next crop box for refinement: %s", next_box)
                            continue
                break

            # Actions without point coordinates do not use refinement
            break

        # Save final action-marked frame if configured
        if final_action and self.settings.click_save_path:
            try:
                marked_frame = draw_action_markers(frame, final_action, self.step_count)
                action_type = final_action.get("action", "unknown")
                save_path = f"{self.settings.click_save_path}/step_{self.step_count:03d}_{action_type}_final.png"
                save_atomic(marked_frame, save_path, self.logger)
                self.logger.debug("Saved final action frame: %s", save_path)
            except Exception as e:
                self.logger.warning("Failed to save action frame: %s", e)

        return final_action

    @staticmethod
    def _action_has_point_position(action: Dict[str, Any]) -> bool:
        """Return True if the action carries a single [x, y] style position."""

        position = action.get("position")
        if not isinstance(position, list) or len(position) != 2:
            return False

        try:
            float(position[0])
            float(position[1])
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    #TODO: Check this logic later, making sure we are not normailizng the poitns more times than nessiary
    def _normalize_point_position(
        action: Dict[str, Any],
        crop_box: Tuple[int, int, int, int],
        full_size: Tuple[int, int],
    ) -> Optional[List[float]]:
        """Convert crop-local coordinates to full-frame normalized coordinates."""

        position = action.get("position")
        if not (isinstance(position, list) and len(position) == 2):
            return None

        try:
            local_x = float(position[0])
            local_y = float(position[1])
        except (TypeError, ValueError):
            return None

        left, top, right, bottom = crop_box
        crop_w = max(right - left, 1)
        crop_h = max(bottom - top, 1)
        full_w, full_h = full_size
        if full_w <= 0 or full_h <= 0:
            return None

        local_x = min(max(local_x, 0.0), 1.0)
        local_y = min(max(local_y, 0.0), 1.0)

        abs_x = left + local_x * crop_w
        abs_y = top + local_y * crop_h

        norm_x = min(max(abs_x / full_w, 0.0), 1.0)
        norm_y = min(max(abs_y / full_h, 0.0), 1.0)
        return [norm_x, norm_y]

    @staticmethod
    def _compute_refinement_box(
        normalized: List[float],
        current_box: Tuple[int, int, int, int],
        full_size: Tuple[int, int],
        crop_ratio: float = 0.5,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute the next crop box centered around the normalized coordinate."""

        full_w, full_h = full_size
        if full_w <= 0 or full_h <= 0:
            return None

        curr_w = current_box[2] - current_box[0]
        curr_h = current_box[3] - current_box[1]
        if curr_w <= 32 and curr_h <= 32:
            return None

        new_w = max(int(curr_w * crop_ratio), 32)
        new_h = max(int(curr_h * crop_ratio), 32)

        center_x = min(max(normalized[0], 0.0), 1.0) * full_w
        center_y = min(max(normalized[1], 0.0), 1.0) * full_h

        left = int(max(0, min(center_x - new_w / 2, full_w - new_w)))
        top = int(max(0, min(center_y - new_h / 2, full_h - new_h)))
        right = min(full_w, left + new_w)
        bottom = min(full_h, top + new_h)

        if right - left <= 0 or bottom - top <= 0:
            return None

        return (left, top, int(right), int(bottom))

    async def _execute_action(self, action: Dict[str, Any]) -> None:
        """Execute an action using the action executor.

        Uses the remote browser's screen size (self.client.frame_size) for coordinate
        scaling, not the video frame dimensions. This ensures clicks are positioned
        correctly even when the video stream resolution differs from the browser resolution.

        :param action: Action dictionary to execute
        """
        action_type = action.get("action")
        if not action_type:
            return

        try:
            executor = getattr(self.client, 'action_executor', None)
            if executor is None:
                raise RuntimeError('Action executor not ready')

            # Get remote screen size from client
            screen_size = self.client.frame_size

            # Validate screen size before executing actions
            if screen_size == (0, 0):
                self.logger.error("Screen size not yet initialized, cannot execute action")
                return

            action_payload = ActionModel(
                action=action.get("action"),
                value=action.get("value"),
                position=action.get("position"),
                amount=action.get("amount"),
            )

            await executor.execute_action(action_payload, screen_size)

            actions_executed.labels(action_type=action_type).inc()
            self.logger.info("Executed action: %s (screen_size=%s)", action, screen_size)

            # Emit action annotation for training data capture
            await self._emit_action_annotation(action)

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

#TODO: this can proably be refactored out of the agent_refactored.py file as it is proably used in other files such as yap_refactorted.py or manual_refactored.py
    async def _send_chat(self, text: str) -> None:
        """Send a chat message through the client connection.

        :param text: Message text to send
        """
        try:
            signaler = getattr(self.client, "signaler", None)
            if signaler is None:
                return
            await signaler.send({
                "event": "chat/message",
                "payload": {"text": f"[agent] {text}"}
            })
        except Exception as e:
            self.logger.debug("Failed to send chat message: %s", e)

    async def _cleanup(self) -> None:
        """Clean up resources before shutdown."""
        self.logger.info("Cleaning up agent resources")

        # Clean up vision agent
        with contextlib.suppress(Exception):
            await self.vision_agent.cleanup()

        # Disconnect client
        with contextlib.suppress(Exception):
            await self.client.disconnect()

        # Stop metrics server like the legacy agent
        ms = getattr(self, "_metrics_server", None)
        mt = getattr(self, "_metrics_thread", None)
        if ms is not None:
            with contextlib.suppress(Exception):
                if hasattr(ms, "shutdown"):
                    ms.shutdown()
                if hasattr(ms, "server_close"):
                    ms.server_close()
        if mt is not None:
            with contextlib.suppress(Exception):
                mt.join(timeout=1.0)
        self._metrics_server = None
        self._metrics_thread = None

        await asyncio.sleep(0.2)
        if self.settings.force_exit_guard_ms > 0:
            await asyncio.sleep(self.settings.force_exit_guard_ms / 1000.0)
            import os as _os, threading as _threading
            active = [t for t in _threading.enumerate()
                      if t.is_alive() and t is not _threading.current_thread() and not t.daemon]
            if active:
                names = [t.name for t in active]
                self.logger.warning("Non-daemon threads still active: %s", names)
                self.logger.info("Forcing process exit now to avoid hang")
                _os._exit(0)

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

    # Create vision agent using factory
    logger.info("Creating vision agent: %s", settings.agent_type)

    from agents import create_vision_agent
    vision_agent = create_vision_agent(settings.agent_type, settings, logger)

    # Log device info
    device_info = vision_agent.get_device_info()
    logger.info("Vision agent initialized: %s", device_info)

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

    #TODO the we need more customizablity on the prompt we are sending to the vllm, currently this file agent_refactored.py is mostly written for showui
    #      we need to support other prompting stratgies.
    # Create agent with dependency injection
    agent = NekoAgent(
        vision_agent=vision_agent,
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
