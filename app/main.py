from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.editor import ImageEditor
from app.image_utils import load_image, png_bytes
from app.prompt_planner import plan_prompt
from app.video import VideoGenerator


settings = get_settings()
editor = ImageEditor(settings)
video_generator = VideoGenerator(settings)

app = FastAPI(title="Mac Local Image Editor")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    with open("app/static/index.html", encoding="utf-8") as file:
        return file.read()


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "device": editor.device,
        "model": settings.edit_model_id,
        "video_model": settings.video_model_id,
        "prompt_planner": settings.prompt_planner,
    }


@app.post("/api/plan")
async def plan(prompt: str = Form(...)) -> dict[str, str]:
    try:
        planned = await plan_prompt(prompt, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"prompt": planned}


@app.post("/api/edit")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    steps: int = Form(20),
    guidance_scale: float = Form(7.5),
    image_guidance_scale: float = Form(1.5),
    seed: int | None = Form(None),
) -> Response:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")
    if steps < 1 or steps > 80:
        raise HTTPException(status_code=400, detail="Steps must be between 1 and 80.")

    try:
        source = load_image(await image.read(), settings.max_image_side)
        planned = await plan_prompt(prompt, settings)
        result = editor.edit(
            image=source,
            prompt=planned,
            steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            seed=seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=png_bytes(result),
        media_type="image/png",
        headers={"X-Planned-Prompt": planned},
    )


@app.post("/api/video")
async def generate_video(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    edit_first: bool = Form(False),
    edit_steps: int = Form(20),
    video_steps: int = Form(25),
    frames: int = Form(14),
    fps: int = Form(7),
    motion_bucket_id: int = Form(127),
    noise_aug_strength: float = Form(0.02),
    seed: int | None = Form(None),
) -> Response:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")
    if video_steps < 1 or video_steps > 80:
        raise HTTPException(status_code=400, detail="Video steps must be between 1 and 80.")
    if frames < 2 or frames > 25:
        raise HTTPException(status_code=400, detail="Frames must be between 2 and 25.")
    if fps < 1 or fps > 30:
        raise HTTPException(status_code=400, detail="FPS must be between 1 and 30.")
    if motion_bucket_id < 1 or motion_bucket_id > 255:
        raise HTTPException(status_code=400, detail="Motion bucket must be between 1 and 255.")

    planned = ""
    temp_path: Path | None = None
    try:
        source = load_image(await image.read(), settings.max_image_side)
        if edit_first:
            planned = await plan_prompt(prompt, settings)
            source = editor.edit(image=source, prompt=planned, steps=edit_steps, seed=seed)

        temp_path = video_generator.generate(
            image=source,
            steps=video_steps,
            frames=frames,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            seed=seed,
        )
        content = temp_path.read_bytes()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return Response(
        content=content,
        media_type="video/mp4",
        headers={"X-Planned-Prompt": planned},
    )
