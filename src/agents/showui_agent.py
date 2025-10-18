"""ShowUI-2B vision agent implementation using Qwen2VL."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from .base import ModelResponse
from .gpu_agent import GPUAgent

if TYPE_CHECKING:
    from ..agent import Settings


class ShowUIAgent(GPUAgent):
    """ShowUI-2B vision agent using Qwen2VL for local GPU inference.

    Inherits device management and executor handling from GPUAgent.
    Implements ShowUI-specific model loading, prompt formatting, and inference.
    """

    def _load_model(self) -> None:
        """Load ShowUI-2B/Qwen2VL model and processor.

        Implements the abstract method from GPUAgent to load model-specific
        components with appropriate device_map and dtype configuration.
        """
        self.logger.info("Loading ShowUI-2B model from %s...", self.settings.repo_id)

        # Build model kwargs with device-specific settings
        model_kwargs: Dict[str, Any] = {"torch_dtype": self.dtype, "device_map": "auto"}

        # MPS requires offload configuration
        if self.device == "mps":
            model_kwargs.update(
                {
                    "offload_folder": self.settings.offload_folder,
                    "offload_state_dict": True,
                }
            )

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.settings.repo_id, **model_kwargs
        ).eval()

        # Load processor with image size configuration
        self.processor = AutoProcessor.from_pretrained(
            self.settings.repo_id,
            size={
                "shortest_edge": self.settings.size_shortest_edge,
                "longest_edge": self.settings.size_longest_edge,
            },
            trust_remote_code=True,
        )

        self.logger.info(
            "Model loaded successfully on device: %s (dtype: %s)",
            self.model.device,
            self.dtype,
        )

        # Log GPU memory usage
        if self.device == "cuda":
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            self.logger.info("GPU memory allocated: %.2fGB", allocated_memory)

    async def _run_inference(
        self, inputs: Any, max_new_tokens: int = 256, temperature: float = 0.0
    ) -> str:
        """Run ShowUI-2B model inference.

        :param inputs: Processed inputs from processor
        :param max_new_tokens: Maximum tokens to generate
        :param temperature: Sampling temperature
        :return: Generated text output
        """

        def _inference() -> str:
            with torch.no_grad():
                t0 = time.monotonic()
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )
                elapsed = time.monotonic() - t0
                self.logger.debug("Inference completed in %.2fs", elapsed)

                output_text = self.processor.decode(
                    output_ids[0][len(inputs["input_ids"][0]) :],
                    skip_special_tokens=True,
                ).strip()
                return output_text

        loop = asyncio.get_running_loop()
        output_text = await loop.run_in_executor(self.executor, _inference)
        return output_text

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
        """Marshal strategy messages into ShowUI processor inputs and run inference."""

        # Collect images referenced in the chat messages preserving order
        image_refs: List[Image.Image] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image"
                    and item.get("image") is not None
                ):
                    # Attach expected size metadata for Qwen processor
                    item.setdefault(
                        "size",
                        {
                            "shortest_edge": self.settings.size_shortest_edge,
                            "longest_edge": self.settings.size_longest_edge,
                        },
                    )
                    image_refs.append(item["image"])

        try:
            chat_template = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat_template],
                images=image_refs or [image],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            output_text = await asyncio.wait_for(
                self._run_inference(inputs),
                timeout=self.settings.inference_timeout,
            )
            return ModelResponse(text=output_text)

        except asyncio.TimeoutError:
            self.logger.error(
                "Model inference timed out after %.1fs",
                self.settings.inference_timeout,
            )
            return ModelResponse(text=None)
        except Exception as exc:
            self.logger.error("Inference failed: %s", exc, exc_info=True)
            return ModelResponse(text=None)
