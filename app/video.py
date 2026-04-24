from __future__ import annotations

import gc
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

from app.config import Settings


class VideoGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: StableVideoDiffusionPipeline | None = None
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

    def _load_pipeline(self) -> StableVideoDiffusionPipeline:
        if self._pipeline is None:
            kwargs = {"torch_dtype": self._dtype()}
            if self.device in {"mps", "cuda"}:
                kwargs["variant"] = "fp16"
            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.settings.video_model_id,
                **kwargs,
            )
            pipeline = pipeline.to(self.device)
            pipeline.enable_attention_slicing()
            if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "enable_forward_chunking"):
                pipeline.unet.enable_forward_chunking()
            self._pipeline = pipeline
        return self._pipeline

    def generate(
        self,
        image: Image.Image,
        output_path: Path | None = None,
        steps: int = 25,
        frames: int = 14,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        seed: int | None = None,
    ) -> Path:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        with self._lock:
            pipeline = self._load_pipeline()
            with torch.inference_mode():
                result = pipeline(
                    image=image,
                    num_inference_steps=steps,
                    num_frames=frames,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    decode_chunk_size=min(8, frames),
                    generator=generator,
                )

            path = output_path
            if path is None:
                temp = NamedTemporaryFile(suffix=".mp4", delete=False)
                path = Path(temp.name)
                temp.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(result.frames[0], str(path), fps=fps)

            if self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            return path
