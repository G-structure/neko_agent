"""Abstract base class for vision agents."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image


class VisionAgent(ABC):
    """Abstract interface for vision-language models that generate GUI automation actions.

    Vision agents process screen images and task descriptions to generate structured
    action commands for GUI automation. This interface abstracts the model implementation,
    allowing for local models (ShowUI, Qwen2VL) or remote APIs (Claude Computer Use, etc).

    Attributes:
        chat_callback: Optional callback for sending chat messages (e.g., thinking tokens)
    """

    chat_callback: Optional[Callable[[str], Any]] = None

    @abstractmethod
    async def generate_action(
        self,
        image: Image.Image,
        task: str,
        system_prompt: str,
        action_history: List[Dict[str, Any]],
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
    ) -> Optional[str]:
        """Generate action string from screen image and task context.

        This method takes a screen image, task description, and context information
        to generate a raw action string. The action string should follow the format
        defined by the system_prompt (typically JSON-formatted action commands).

        :param image: Current screen image (may be cropped for refinement)
        :param task: Task description to accomplish
        :param system_prompt: System prompt defining action space and format
        :param action_history: List of previous actions (for context)
        :param crop_box: Crop coordinates (left, top, right, bottom) in full_size coordinates
        :param iteration: Current refinement iteration (0 = first pass)
        :param full_size: Original full frame size (width, height)
        :return: Raw action string from model, or None if generation fails
        """
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return device and model information for logging.

        :return: Dictionary containing device type, memory info, model details, etc.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up model resources (GPU memory, connections, etc.).

        Called during agent shutdown to release resources gracefully.
        """
        pass