"""ShowUI-2B vision agent implementation using Qwen2VL."""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from .base import VisionAgent

if TYPE_CHECKING:
    from ..agent_refactored import Settings


class ShowUIAgent(VisionAgent):
    """ShowUI-2B vision agent using Qwen2VL for local GPU inference.

    Implements the VisionAgent interface for ShowUI-2B model, handling:
    - Model loading and device management (CUDA/MPS/CPU)
    - Chat template formatting
    - Image preprocessing and tokenization
    - Inference execution with timeout handling
    """

    def __init__(self, settings: 'Settings', logger: logging.Logger):
        """Initialize ShowUI agent with model loading.

        :param settings: Configuration settings
        :param logger: Logger instance
        """
        self.settings = settings
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="showui-inference")

        # Device and dtype detection
        self.device = "cpu"
        self.dtype = torch.float32

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info("CUDA GPU detected: %s (%.1fGB)", gpu_name, gpu_memory)
        elif torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            try:
                _ = torch.zeros(1, dtype=torch.bfloat16, device="mps")
                self.dtype = torch.bfloat16
            except RuntimeError:
                self.dtype = torch.float32
            self.device = "mps"
            self.logger.info("Apple MPS detected")
            os.makedirs(settings.offload_folder, exist_ok=True)
        else:
            self.logger.warning("No GPU acceleration available - using CPU")

        # Model loading
        self.logger.info("Loading ShowUI-2B model from %s...", settings.repo_id)
        model_kwargs: Dict[str, Any] = {"torch_dtype": self.dtype, "device_map": "auto"}
        if self.device == "mps":
            model_kwargs.update({
                "offload_folder": settings.offload_folder,
                "offload_state_dict": True
            })

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            settings.repo_id, **model_kwargs
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            settings.repo_id,
            size={
                "shortest_edge": settings.size_shortest_edge,
                "longest_edge": settings.size_longest_edge
            },
            trust_remote_code=True
        )

        self.logger.info("Model loaded successfully on device: %s (dtype: %s)", self.model.device, self.dtype)

        if self.device == "cuda":
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            self.logger.info("GPU memory allocated: %.2fGB", allocated_memory)

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
        """Generate action using ShowUI-2B model inference.

        :param image: Current screen image
        :param task: Task description
        :param system_prompt: System prompt defining action space
        :param action_history: Previous actions for context
        :param crop_box: Crop coordinates for refinement
        :param iteration: Refinement iteration number
        :param full_size: Original frame size
        :return: Raw action string or None on failure
        """
        # Build prompt components
        user_text, history_text = self._build_prompt_components(
            task, action_history, crop_box, iteration, full_size
        )

        # Construct multimodal message
        segments: List[Dict[str, Any]] = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": user_text},
        ]
        if history_text:
            segments.append({"type": "text", "text": history_text})
        segments.append({
            "type": "image",
            "image": image,
            "size": {
                "shortest_edge": self.settings.size_shortest_edge,
                "longest_edge": self.settings.size_longest_edge,
            },
        })

        messages = [{"role": "user", "content": segments}]

        # Run inference
        future: Optional[asyncio.Future[str]] = None
        try:
            text_input = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            loop = asyncio.get_running_loop()

            def _inference() -> str:
                with torch.no_grad():
                    t0 = time.monotonic()
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    elapsed = time.monotonic() - t0
                    self.logger.debug("Inference completed in %.2fs", elapsed)

                    output_text = self.processor.decode(
                        output_ids[0][len(inputs["input_ids"][0]):],
                        skip_special_tokens=True,
                    ).strip()
                    return output_text

            future = loop.run_in_executor(self.executor, _inference)
            output_text = await asyncio.wait_for(
                future, timeout=self.settings.inference_timeout
            )
            return output_text

        except asyncio.TimeoutError:
            if future:
                future.cancel()
            self.logger.error(
                "Model inference timed out after %.1fs",
                self.settings.inference_timeout,
            )
            return None
        except Exception as e:
            self.logger.error("Inference failed: %s", e, exc_info=True)
            return None

    def _build_prompt_components(
        self,
        task: str,
        action_history: List[Dict[str, Any]],
        crop_box: Tuple[int, int, int, int],
        iteration: int,
        full_size: Tuple[int, int],
    ) -> Tuple[str, Optional[str]]:
        """Build textual prompt components for the model.

        :param task: Task description
        :param action_history: Previous actions
        :param crop_box: Current crop box
        :param iteration: Refinement iteration
        :param full_size: Full frame size
        :return: Tuple of (user_text, history_text)
        """
        if iteration == 0:
            user_text = f"Task: {task}\n\nCurrent observation:"
        else:
            full_w, full_h = full_size
            crop_w = max(crop_box[2] - crop_box[0], 1)
            crop_h = max(crop_box[3] - crop_box[1], 1)
            cx = ((crop_box[0] + crop_box[2]) / 2) / full_w if full_w else 0.5
            cy = ((crop_box[1] + crop_box[3]) / 2) / full_h if full_h else 0.5
            span_x = crop_w / full_w if full_w else 1.0
            span_y = crop_h / full_h if full_h else 1.0
            user_text = (
                f"Task: {task}\n\n"
                f"Refinement pass {iteration + 1} of {self.settings.refinement_steps} "
                f"zoomed near normalized coords ({cx:.2f}, {cy:.2f}) "
                f"with approx span ({span_x:.2f}, {span_y:.2f}).\n\nCurrent observation:"
            )

        history_text: Optional[str] = None
        if action_history:
            history_lines = [
                f"{idx}. {json.dumps(act, ensure_ascii=False)}"
                for idx, act in enumerate(action_history[-5:], 1)
            ]
            history_text = "Previous actions:\n" + "\n".join(history_lines)

        return user_text, history_text

    def get_device_info(self) -> Dict[str, Any]:
        """Return device and model information.

        :return: Dictionary with device, dtype, model info
        """
        info = {
            "device": self.device,
            "dtype": str(self.dtype),
            "model": self.settings.repo_id,
        }

        if self.device == "cuda":
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9
            except Exception:
                pass

        return info

    async def cleanup(self) -> None:
        """Clean up model resources.

        Shuts down the inference executor and releases GPU memory.
        """
        self.logger.info("Cleaning up ShowUI agent resources")

        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)

        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
                self.logger.debug("Cleared CUDA cache")
            except Exception:
                pass