"""Vision agent abstractions for Neko automation.

This module provides pluggable vision model interfaces for GUI automation.
Supports local models (ShowUI/Qwen2VL) and remote APIs (Claude, Qwen3VL).
"""

import logging
from typing import TYPE_CHECKING

from .base import VisionAgent

if TYPE_CHECKING:
    from ..agent_refactored import Settings

__all__ = ["VisionAgent", "create_vision_agent"]


def create_vision_agent(agent_type: str, settings: 'Settings', logger: logging.Logger) -> VisionAgent:
    """Factory function to create vision agents based on type.

    :param agent_type: Agent type identifier ('showui', 'claude', 'qwen3vl')
    :param settings: Configuration settings
    :param logger: Logger instance
    :return: Configured VisionAgent instance
    :raises ValueError: If agent_type is not recognized
    """
    if agent_type == "showui":
        from .showui_agent import ShowUIAgent
        return ShowUIAgent(settings, logger)
    elif agent_type == "claude":
        from .remote_agent import ClaudeComputerUseAgent
        return ClaudeComputerUseAgent(settings, logger)
    elif agent_type == "qwen3vl":
        from .remote_agent import Qwen3VLAgent
        return Qwen3VLAgent(settings, logger)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Supported: showui, claude, qwen3vl")