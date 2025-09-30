"""Pluggable prompt strategies for vision agents.

Provides strategy classes that can render prompts/messages for different
vision-language models and parse their outputs back into structured actions.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from PIL import Image

from neko_comms.types import ACTION_SPACE_DESC

from .parsing import safe_parse_action
from .reasoning import strip_think_segments


HistoryEntry = Dict[str, Any]


class PromptStrategy(ABC):
    """Base prompt strategy for formatting prompts and parsing actions."""

    name: str = "base"

    def __init__(self, settings: Any, logger: logging.Logger):
        self.settings = settings
        self.logger = logger

    @abstractmethod
    def get_system_prompt(self, nav_mode: str) -> str:
        """Return the system prompt string for the active model."""

    @abstractmethod
    def build_messages(
        self,
        task: str,
        action_history: List[HistoryEntry],
        current_image: Image.Image,
        nav_mode: str,
        is_refinement: bool = False,
        crop_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Create an OpenAI-style message list for the current step."""

    def parse_output(
        self,
        response: str,
        nav_mode: str,
        logger: Optional[logging.Logger] = None,
    ) -> Optional[Dict[str, Any]]:
        """Parse raw response text into an action dict using shared utilities."""
        cleaned = strip_think_segments(response)
        return safe_parse_action(cleaned, nav_mode=nav_mode, logger=logger)


class SimpleCoTPromptStrategy(PromptStrategy):
    """Single-turn chain-of-thought prompt strategy producing JSON actions."""

    name = "simple_cot"

    def get_system_prompt(self, nav_mode: str) -> str:
        action_space_desc = ACTION_SPACE_DESC.get(nav_mode, ACTION_SPACE_DESC.get("web", ""))
        return (
            "You are an expert AI assistant that controls a {_APP} interface. "
            "Review the task, previous actions, and current screenshot, then respond "
            "with the next action as JSON.\n\nAction space:\n{_ACTION_SPACE}\n"
            "Respond strictly with a JSON object containing keys: action, value, position."
        ).format(_APP=nav_mode.upper(), _ACTION_SPACE=action_space_desc)

    def build_messages(
        self,
        task: str,
        action_history: List[HistoryEntry],
        current_image: Image.Image,
        nav_mode: str,
        is_refinement: bool = False,
        crop_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if crop_info is None:
            crop_info = {}

        history_lines: List[str] = []
        if action_history:
            for idx, entry in enumerate(action_history[-5:], 1):
                action_json = json.dumps(entry.get("action"), ensure_ascii=False)
                frame_ref = entry.get("frame_ref")
                reasoning = entry.get("reasoning")
                line = f"{idx}. action={action_json}"
                if frame_ref:
                    line += f" (frame={frame_ref})"
                if reasoning:
                    line += f"\n   reasoning: {reasoning}"
                for sub_idx, sub in enumerate(entry.get("sub_actions", []), 1):
                    sub_json = json.dumps(sub.get("action"), ensure_ascii=False)
                    sub_reason = sub.get("reasoning")
                    line += f"\n   refine {sub_idx}: {sub_json}"
                    if sub_reason:
                        line += f"\n      reasoning: {sub_reason}"
                history_lines.append(line)

        crop_desc = ""
        if is_refinement and crop_info:
            norm_center = crop_info.get("normalized_center")
            span = crop_info.get("normalized_span")
            if norm_center and span:
                crop_desc = (
                    f"Refinement pass focused near coords ({norm_center[0]:.2f}, {norm_center[1]:.2f}) "
                    f"with span ({span[0]:.2f}, {span[1]:.2f}).\n"
                )

        history_text = "\nPrevious actions:\n" + "\n".join(history_lines) if history_lines else ""
        observation_prefix = "Refinement observation:" if is_refinement else "Current observation:"
        user_text = (
            f"Task: {task}\n{crop_desc}{observation_prefix}\n"
            f"{history_text}"
        ).strip()

        return [
            {
                "role": "system",
                "content": self.get_system_prompt(nav_mode),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": current_image},
                ],
            },
        ]


class ConversationalChainPromptStrategy(PromptStrategy):
    """Multi-turn conversational strategy optimized for tool-calling agents."""

    name = "conversational_chain"

    def get_system_prompt(self, nav_mode: str) -> str:
        action_space_desc = ACTION_SPACE_DESC.get(nav_mode, ACTION_SPACE_DESC.get("web", ""))
        return (
            "You are a vision-language agent controlling a {_APP} computer.\n"
            "You have a tool named `computer` with actions CLICK, INPUT, SCROLL, ENTER, HOVER, SELECT.\n"
            "Always think in <think> tags before calling the tool.\n"
            "Return DONE when the task is complete.\n"
            "Action space:\n{_ACTION_SPACE}"
        ).format(_APP=nav_mode.upper(), _ACTION_SPACE=action_space_desc)

    def build_messages(
        self,
        task: str,
        action_history: List[HistoryEntry],
        current_image: Image.Image,
        nav_mode: str,
        is_refinement: bool = False,
        crop_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.get_system_prompt(nav_mode),
            }
        ]

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task: {task}. Use the computer tool to interact with the screen."},
                ],
            }
        )

        for entry in action_history:
            frame_ref = entry.get("frame_ref", "previous")
            frame_image = entry.get("image")
            reasoning = entry.get("reasoning")
            action = entry.get("action")

            observe_text = f"Observe previous state ({frame_ref})."
            content = [{"type": "text", "text": observe_text}]
            if frame_image is not None:
                content.append({"type": "image", "image": frame_image})
            messages.append({"role": "user", "content": content})

            if reasoning:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"<think>Reasoning: {reasoning}</think>"}],
                    }
                )

            if action:
                tool_payload = json.dumps({"tool": "computer", "args": action}, ensure_ascii=False)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"<think>Executed tool call:</think> {tool_payload}",
                            }
                        ],
                    }
                )

            for sub in entry.get("sub_actions", []):
                sub_frame = sub.get("image")
                crop = sub.get("crop_info", {})
                norm_center = crop.get("normalized_center")
                span = crop.get("normalized_span")
                zoom_text = "Zoom Task: Refine click."
                if norm_center and span:
                    zoom_text = (
                        "Zoom Task: Refine click at normalized coords "
                        f"({norm_center[0]:.2f}, {norm_center[1]:.2f}) with span "
                        f"({span[0]:.2f}, {span[1]:.2f})."
                    )
                sub_content = [{"type": "text", "text": zoom_text}]
                if sub_frame is not None:
                    sub_content.append({"type": "image", "image": sub_frame})
                messages.append({"role": "user", "content": sub_content})

                sub_reason = sub.get("reasoning")
                if sub_reason:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"<think>{sub_reason}</think>"}],
                        }
                    )
                sub_action = sub.get("action")
                if sub_action:
                    sub_payload = json.dumps({"tool": "computer", "args": sub_action}, ensure_ascii=False)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"<think>Executed tool call:</think> {sub_payload}",
                                }
                            ],
                        }
                    )

        continuation_text = "Continue based on previous action."
        if is_refinement:
            continuation_text = "Continue refinement with the zoomed observation."

        final_content = [
            {"type": "text", "text": continuation_text},
            {"type": "image", "image": current_image},
        ]
        messages.append({"role": "user", "content": final_content})
        return messages


STRATEGIES: Dict[str, Any] = {
    SimpleCoTPromptStrategy.name: SimpleCoTPromptStrategy,
    ConversationalChainPromptStrategy.name: ConversationalChainPromptStrategy,
}


def create_prompt_strategy(name: Optional[str], settings: Any, logger: logging.Logger) -> PromptStrategy:
    """Instantiate a prompt strategy by name, falling back to simple_cot."""
    selected = (name or SimpleCoTPromptStrategy.name).lower()
    strategy_cls = STRATEGIES.get(selected, SimpleCoTPromptStrategy)
    return strategy_cls(settings, logger)
