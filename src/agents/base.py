"""Abstract base class for vision agents with prompt strategies."""

from __future__ import annotations

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .prompt_strategies import PromptStrategy, create_prompt_strategy
from .reasoning import extract_think_segments


@dataclass
class ModelResponse:
    """Container for model responses returned by concrete agents."""

    text: Optional[str]
    reasoning: Optional[str] = None
    raw: Any = None
    reasoning_emitted: bool = False


@dataclass
class HistoryNode:
    """Represents a single action (or refinement) in history."""

    action: Dict[str, Any]
    frame_ref: Optional[str] = None
    reasoning: Optional[str] = None
    image: Optional[Image.Image] = None
    crop_info: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    sub_actions: List['HistoryNode'] = field(default_factory=list)


class EnhancedActionHistory:
    """Manages action history with refinement sub-actions and frame references."""

    def __init__(self) -> None:
        self._entries: List[HistoryNode] = []

    def reset(self) -> None:
        self._entries.clear()

    def add(self, entry: HistoryNode, *, is_refinement: bool) -> None:
        if is_refinement and self._entries:
            self._entries[-1].sub_actions.append(entry)
            return
        self._entries.append(entry)

    def as_list(self) -> List[Dict[str, Any]]:
        def serialize(node: HistoryNode) -> Dict[str, Any]:
            return {
                "action": node.action,
                "frame_ref": node.frame_ref,
                "reasoning": node.reasoning,
                "image": node.image,
                "crop_info": node.crop_info,
                "timestamp": node.timestamp,
                "sub_actions": [serialize(sub) for sub in node.sub_actions],
            }

        return [serialize(entry) for entry in self._entries]

    def __len__(self) -> int:
        return len(self._entries)


class VisionAgent(ABC):
    """Abstract interface for vision-language models that generate GUI actions."""

    chat_callback: Optional[Callable[[str], Any]] = None

    def __init__(
        self,
        settings: Any,
        logger: Any,
        *,
        default_prompt_strategy: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.history = EnhancedActionHistory()

        strategy_name = getattr(settings, "prompt_strategy", None) or default_prompt_strategy
        self.prompt_strategy: PromptStrategy = create_prompt_strategy(strategy_name, settings, logger)

    async def generate_action(
        self,
        image: Image.Image,
        task: str,
        nav_mode: str,
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
        *,
        frame_ref: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a structured action dictionary using the configured strategy."""

        is_refinement = iteration > 0
        crop_info = self._build_crop_metadata(crop_box, full_size)
        crop_info["nav_mode"] = nav_mode

        history_payload = self.history.as_list()
        system_prompt = self.prompt_strategy.get_system_prompt(nav_mode)
        messages = self.prompt_strategy.build_messages(
            task=task,
            action_history=history_payload,
            current_image=image,
            nav_mode=nav_mode,
            is_refinement=is_refinement,
            crop_info=crop_info,
        )

        response = await self._invoke_model(
            system_prompt=system_prompt,
            messages=messages,
            image=image,
            task=task,
            nav_mode=nav_mode,
            crop_box=crop_box,
            iteration=iteration,
            full_size=full_size,
            is_refinement=is_refinement,
        )

        if not response or not response.text:
            return None

        action_dict = self.prompt_strategy.parse_output(
            response.text,
            nav_mode=nav_mode,
            logger=self.logger,
        )

        if action_dict:
            reasoning_text = self._resolve_reasoning_text(response)
            if reasoning_text:
                self._log_reasoning(reasoning_text)
                if not getattr(response, "reasoning_emitted", False):
                    await self._emit_reasoning(reasoning_text)
            self._update_history(
                action=action_dict,
                image=image,
                frame_ref=frame_ref,
                crop_info=crop_info,
                reasoning=reasoning_text,
                is_refinement=is_refinement,
            )
        return action_dict

    @abstractmethod
    async def _invoke_model(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        image: Image.Image,
        task: str,
        nav_mode: str,
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
        is_refinement: bool,
    ) -> ModelResponse:
        """Concrete agents must implement model invocation and return raw text."""

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return device/model information for logging."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Release model resources."""

    def reset_history(self) -> None:
        """Clear accumulated action history (e.g., when starting a new task)."""
        self.history.reset()

    def _resolve_reasoning_text(self, response: ModelResponse) -> Optional[str]:
        """Derive reasoning text from model response, preferring API fields."""
        if not response:
            return None

        if response.reasoning and response.reasoning.strip():
            return response.reasoning.strip()

        # Try raw payloads returned by APIs
        raw = response.raw
        if isinstance(raw, str):
            extracted = extract_think_segments(raw)
            if extracted:
                return extracted
        elif isinstance(raw, dict):
            try:
                choice = raw.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content")
                if isinstance(content, list):
                    combined = chr(10).join(
                        segment.get("text", "")
                        for segment in content
                        if isinstance(segment, dict) and segment.get("type") == "text"
                    )
                    extracted = extract_think_segments(combined)
                    if extracted:
                        return extracted
                elif isinstance(content, str):
                    extracted = extract_think_segments(content)
                    if extracted:
                        return extracted
            except Exception:
                pass

        # Fallback to scanning primary text output
        if response.text:
            extracted = extract_think_segments(response.text)
            if extracted:
                return extracted
        return None

    def _log_reasoning(self, text: str) -> None:
        """Log model reasoning content for observability."""
        preview = " ".join(segment.strip() for segment in text.splitlines() if segment.strip())
        self.logger.info("[thinking] %s", preview or text.strip())

    async def _emit_reasoning(self, text: str) -> None:
        """Send reasoning to an attached chat callback if available."""
        callback = self.chat_callback
        if not callback:
            return
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(text)
            else:
                callback(text)
        except Exception as exc:
            self.logger.debug("Reasoning callback failed: %s", exc, exc_info=True)

    def _build_crop_metadata(
        self,
        crop_box: Tuple[int, int, int, int],
        full_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        left, top, right, bottom = crop_box
        width = max(right - left, 1)
        height = max(bottom - top, 1)
        full_w, full_h = full_size

        if full_w <= 0 or full_h <= 0:
            normalized_center = [0.5, 0.5]
            normalized_span = [1.0, 1.0]
        else:
            center_x = left + width / 2
            center_y = top + height / 2
            normalized_center = [center_x / full_w, center_y / full_h]
            normalized_span = [width / full_w, height / full_h]

        return {
            "box": crop_box,
            "full_size": full_size,
            "normalized_center": normalized_center,
            "normalized_span": normalized_span,
        }

    def _update_history(
        self,
        *,
        action: Dict[str, Any],
        image: Image.Image,
        frame_ref: Optional[str],
        crop_info: Dict[str, Any],
        reasoning: Optional[str],
        is_refinement: bool,
    ) -> None:
        entry = HistoryNode(
            action=action,
            frame_ref=frame_ref,
            reasoning=reasoning,
            image=image.copy() if image is not None else None,
            crop_info=crop_info,
        )
        self.history.add(entry, is_refinement=is_refinement)
