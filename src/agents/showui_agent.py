"""ShowUI-2B vision agent implementation using Qwen2VL."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from .gpu_agent import GPUAgent

if TYPE_CHECKING:
    from ..agent_refactored import Settings


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
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "device_map": "auto"
        }

        # MPS requires offload configuration
        if self.device == "mps":
            model_kwargs.update({
                "offload_folder": self.settings.offload_folder,
                "offload_state_dict": True
            })

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.settings.repo_id, **model_kwargs
        ).eval()

        # Load processor with image size configuration
        self.processor = AutoProcessor.from_pretrained(
            self.settings.repo_id,
            size={
                "shortest_edge": self.settings.size_shortest_edge,
                "longest_edge": self.settings.size_longest_edge
            },
            trust_remote_code=True
        )

        self.logger.info(
            "Model loaded successfully on device: %s (dtype: %s)",
            self.model.device,
            self.dtype
        )

        # Log GPU memory usage
        if self.device == "cuda":
            allocated_memory = torch.cuda.memory_allocated(0) / 1e9
            self.logger.info("GPU memory allocated: %.2fGB", allocated_memory)

    async def _run_inference(
        self,
        inputs: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.0
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
                    output_ids[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True,
                ).strip()
                return output_text

        loop = asyncio.get_running_loop()
        output_text = await loop.run_in_executor(self.executor, _inference)
        return output_text

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

        # Run inference with timeout
        try:
            # Apply chat template and process inputs
            text_input = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Run inference through base class method
            output_text = await asyncio.wait_for(
                self._run_inference(inputs),
                timeout=self.settings.inference_timeout
            )
            return output_text

        except asyncio.TimeoutError:
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