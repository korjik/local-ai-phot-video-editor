from __future__ import annotations

import gc
from threading import Lock

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

from app.config import Settings


class ImageEditor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: StableDiffusionInstructPix2PixPipeline | None = None
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

    def _load_pipeline(self) -> StableDiffusionInstructPix2PixPipeline:
        if self._pipeline is None:
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.settings.edit_model_id,
                torch_dtype=self._dtype(),
                safety_checker=None,
            )
            pipeline = pipeline.to(self.device)
            pipeline.enable_attention_slicing()
            self._pipeline = pipeline
        return self._pipeline

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        steps: int = 20,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        seed: int | None = None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        with self._lock:
            pipeline = self._load_pipeline()
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    generator=generator,
                ).images[0]

            if self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            return result
