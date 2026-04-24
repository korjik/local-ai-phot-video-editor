from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.editor import ImageEditor
from app.image_utils import load_image, png_bytes
from app.prompt_planner import mentions_clothing, mentions_full_outfit_swap, plan_prompt
from app.segmenter import ALL_CLOTHING_LABELS, HumanSegmenter
from app.video import VideoGenerator
from app.vton import VirtualTryOnPipeline


settings = get_settings()
editor = ImageEditor(settings)
video_generator = VideoGenerator(settings)
segmenter = HumanSegmenter()
vton_pipeline = VirtualTryOnPipeline(settings)

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
        "segment_model": HumanSegmenter.MODEL_ID,
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
    guidance_scale: float = Form(3.0),
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

        if mentions_clothing(prompt):
            # Detect clothing region and restrict the edit to that mask.
            # For full outfit swaps (upper + lower garments both mentioned) we:
            #   - fill the waist gap between upper and lower clothing masks
            #   - use a higher guidance_scale so the model follows the prompt
            #   - use a lower image_guidance_scale so it actually replaces the outfit
            #     instead of just tinting it (1.2 preserves too much for a full swap)
            is_full_swap = mentions_full_outfit_swap(prompt)
            mask = segmenter.clothing_mask(source, fill_gap=is_full_swap)
            if mask is not None:
                result = editor.edit_masked(
                    image=source,
                    mask=mask,
                    prompt=planned,
                    steps=max(steps, 25) if is_full_swap else steps,
                    guidance_scale=max(guidance_scale, 8.5 if is_full_swap else 7.0),
                    image_guidance_scale=min(image_guidance_scale, 0.9 if is_full_swap else 1.2),
                    seed=seed,
                )
            else:
                # No clothing detected — fall back to full-image edit
                result = editor.edit(
                    image=source,
                    prompt=planned,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    seed=seed,
                )
        else:
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


@app.post("/api/vton")
async def virtual_try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    prompt: str = Form(""),
    steps: int = Form(30),
    guidance_scale: float = Form(2.0),
    seed: int | None = Form(None),
) -> Response:
    """Virtual try-on: place the garment from the collection onto the person.

    person_image  — photo of the person (wearing anything).
    garment_image — product photo of the garment (e.g. bra + panties from collection).
    prompt        — optional text description of the garment for better accuracy.
    """
    for upload in (person_image, garment_image):
        if not upload.content_type or not upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Both uploads must be images.")
    if steps < 1 or steps > 80:
        raise HTTPException(status_code=400, detail="Steps must be between 1 and 80.")

    try:
        person = load_image(await person_image.read(), settings.max_image_side)
        garment = load_image(await garment_image.read(), settings.max_image_side)

        mask = segmenter.clothing_mask(person, fill_gap=True)
        if mask is None:
            raise ValueError("No clothing detected on the person image.")

        result = vton_pipeline.try_on(
            person_image=person,
            garment_image=garment,
            mask=mask,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=png_bytes(result), media_type="image/png")


@app.post("/api/segment")
async def segment_image(
    image: UploadFile = File(...),
    all_clothing: bool = Form(False),
) -> Response:
    """Return a visualization of the detected clothing regions.

    The response is a PNG with clothing areas highlighted in green.
    If no person is detected, the image is tinted red.
    all_clothing=true includes hats and shoes in addition to main garments.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    try:
        source = load_image(await image.read(), settings.max_image_side)
        label_ids = ALL_CLOTHING_LABELS if all_clothing else None
        kwargs = {"label_ids": label_ids} if label_ids is not None else {}
        vis = segmenter.visualize(source, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(content=png_bytes(vis), media_type="image/png")


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
