from __future__ import annotations

import gc
from threading import Lock

import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionXLInstructPix2PixPipeline,
)
from PIL import Image

from app.config import Settings

EditPipeline = StableDiffusionInstructPix2PixPipeline | StableDiffusionXLInstructPix2PixPipeline


class ImageEditor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: EditPipeline | None = None
        self._lock = Lock()

    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _dtype(self) -> torch.dtype:
        if self.device in {"mps", "cuda"}:
            return torch.float16
        return torch.float32

    @property
    def is_sdxl_edit_model(self) -> bool:
        return "sdxl" in self.settings.edit_model_id.lower()

    def _load_pipeline(self) -> EditPipeline:
        if self._pipeline is None:
            pipeline_class = (
                StableDiffusionXLInstructPix2PixPipeline
                if self.is_sdxl_edit_model
                else StableDiffusionInstructPix2PixPipeline
            )
            pipeline_kwargs = {"torch_dtype": self._dtype()}
            if not self.is_sdxl_edit_model:
                pipeline_kwargs["safety_checker"] = None

            pipeline = pipeline_class.from_pretrained(
                self.settings.edit_model_id,
                **pipeline_kwargs,
            )
            pipeline = pipeline.to(self.device)
            pipeline.enable_attention_slicing()
            self._pipeline = pipeline
        return self._pipeline

    # Negative prompt steers the model away from common face and anatomy artifacts.
    _DEFAULT_NEGATIVE = (
        "blurry, deformed, distorted face, asymmetric face, bad eyes, bad teeth, "
        "extra limbs, missing limbs, bad hands, fused fingers, bad anatomy, low quality, duplicate"
    )

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        steps: int = 20,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 1.5,
        seed: int | None = None,
        negative_prompt: str = _DEFAULT_NEGATIVE,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        with self._lock:
            pipeline = self._load_pipeline()
            pipeline_kwargs: dict[str, object] = {"negative_prompt": negative_prompt}
            if self.is_sdxl_edit_model:
                pipeline_kwargs["height"] = image.height
                pipeline_kwargs["width"] = image.width

            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    generator=generator,
                    **pipeline_kwargs,
                ).images[0]

            if self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            return result
