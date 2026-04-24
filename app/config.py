from functools import lru_cache
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    edit_model_id: str = "diffusers/sdxl-instructpix2pix-768"
    video_model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    prompt_planner: str = "rules"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    max_image_side: int = 768


@lru_cache
def get_settings() -> Settings:
    return Settings(
        edit_model_id=os.getenv("EDIT_MODEL_ID", "diffusers/sdxl-instructpix2pix-768"),
        video_model_id=os.getenv(
            "VIDEO_MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt"
        ),
        prompt_planner=os.getenv("PROMPT_PLANNER", "rules").strip().lower(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        max_image_side=int(os.getenv("MAX_IMAGE_SIDE", "768")),
    )
