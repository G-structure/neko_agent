"""Base class for GPU-accelerated local vision models."""

import asyncio
import logging
import os
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

from .base import VisionAgent

if TYPE_CHECKING:
    from ..agent_refactored import Settings


class GPUAgent(VisionAgent):
    """Abstract base class for local GPU-accelerated vision models.

    Provides common functionality for:
    - Device detection (CUDA/MPS/CPU)
    - Dtype selection (bfloat16/float32)
    - Executor management for async inference
    - GPU memory tracking and cleanup

    Subclasses must implement:
    - _load_model(): Model-specific loading logic
    - _run_inference(): Model-specific inference logic
    """

    def __init__(self, settings: 'Settings', logger: logging.Logger):
        """Initialize GPU agent with device detection and model loading.

        :param settings: Configuration settings
        :param logger: Logger instance
        """
        super().__init__(settings, logger, default_prompt_strategy="simple_cot")

        # Device and dtype will be set by _detect_device
        self.device: str = "cpu"
        self.dtype: torch.dtype = torch.float32

        # Detect and configure device/dtype
        self._detect_device()
        self._configure_dtype()

        # Create executor for async inference
        self.executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="gpu-inference"
        )

        # Call subclass to load model
        self._load_model()

    def _detect_device(self) -> None:
        """Detect available compute device (CUDA/MPS/CPU)."""
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info("CUDA GPU detected: %s (%.1fGB)", gpu_name, gpu_memory)
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.logger.info("Apple MPS detected")
            # Ensure offload folder exists for MPS
            os.makedirs(self.settings.offload_folder, exist_ok=True)
        else:
            self.device = "cpu"
            self.logger.warning("No GPU acceleration available - using CPU")

    def _configure_dtype(self) -> None:
        """Configure dtype based on device capabilities."""
        if self.device == "cuda":
            self.dtype = torch.bfloat16
        elif self.device == "mps":
            # Test if MPS supports bfloat16
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            try:
                _ = torch.zeros(1, dtype=torch.bfloat16, device="mps")
                self.dtype = torch.bfloat16
                self.logger.debug("MPS supports bfloat16")
            except RuntimeError:
                self.dtype = torch.float32
                self.logger.debug("MPS does not support bfloat16, using float32")
        else:
            self.dtype = torch.float32

    @abstractmethod
    def _load_model(self) -> None:
        """Load model and processor (implementation-specific).

        Subclasses should:
        1. Load the model with appropriate device_map and dtype
        2. Load the processor/tokenizer
        3. Store as instance attributes (self.model, self.processor, etc.)
        """
        pass

    @abstractmethod
    async def _run_inference(
        self,
        inputs: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.0
    ) -> str:
        """Run model inference (implementation-specific).

        :param inputs: Processed model inputs (format depends on model)
        :param max_new_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature
        :return: Generated text output
        """
        pass

    def get_device_info(self) -> Dict[str, Any]:
        """Return device and model information.

        :return: Dictionary with device, dtype, and model details
        """
        info = {
            "device": self.device,
            "dtype": str(self.dtype),
            "model": getattr(self.settings, "repo_id", "unknown"),
        }

        if self.device == "cuda":
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total_gb"] = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
                info["gpu_memory_allocated_gb"] = (
                    torch.cuda.memory_allocated(0) / 1e9
                )
            except Exception as e:
                self.logger.debug("Failed to get CUDA info: %s", e)

        return info

    async def cleanup(self) -> None:
        """Clean up GPU resources and executor.

        Shuts down the thread pool executor and clears CUDA cache if applicable.
        """
        self.logger.info("Cleaning up GPU agent resources")

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.logger.debug("Executor shut down")

        # Clear CUDA cache
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                self.logger.debug("Cleared CUDA cache")
            except Exception as e:
                self.logger.debug("Failed to clear CUDA cache: %s", e)
