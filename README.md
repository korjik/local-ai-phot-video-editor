# Mac Local Image Editor

Local generative image editing for Apple Silicon Macs. The app takes an input image and a plain-language edit prompt, optionally expands the prompt through a local LLM, then runs a local Diffusers image-editing model on the Mac GPU through PyTorch MPS.

## Architecture

```text
Browser UI
  -> FastAPI /api/edit
  -> Prompt planner
      -> rule-based cleanup
      -> optional Ollama expansion
  -> Diffusers image editor
      -> StableDiffusionXLInstructPix2PixPipeline by default
      -> StableDiffusionInstructPix2PixPipeline for legacy SD 1.x models
      -> torch device: mps on Apple Silicon, cpu fallback otherwise
  -> Diffusers video generator
      -> StableVideoDiffusionPipeline
      -> MP4 preview in browser
  -> edited PNG response
```

The default model is `diffusers/sdxl-instructpix2pix-768`, which gives higher-quality prompt-guided edits than the older SD 1.x InstructPix2Pix model and generally handles people, faces, and body detail better. First run downloads model weights from Hugging Face.
For a smaller legacy model, set `EDIT_MODEL_ID=timbrooks/instruct-pix2pix`.
The default video model is `stabilityai/stable-video-diffusion-img2vid-xt`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

If your Python is newer than 3.12, install Python 3.11 or 3.12 first. PyTorch MPS support is only available on Apple Silicon builds of PyTorch.

## CLI Usage

Plan the model-facing prompt:

```bash
mac-image-edit plan "make the sky dramatic but keep the buildings unchanged"
```

Generate an edit:

```bash
mac-image-edit edit input.jpg "make the sky dramatic but keep the buildings unchanged" \
  --output outputs/edited.png \
  --steps 20 \
  --guidance-scale 3.0 \
  --image-guidance-scale 1.5 \
  --show-planned-prompt
```

The command prints the output path when generation finishes.

Generate a video from an image:

```bash
mac-image-edit video input.jpg \
  --output outputs/video.mp4 \
  --video-steps 25 \
  --frames 14 \
  --fps 7
```

Generate a prompt-edited image first, then animate it:

```bash
mac-image-edit video input.jpg "make the sky dramatic but keep the buildings unchanged" \
  --edit-first \
  --output outputs/edited-video.mp4 \
  --show-planned-prompt
```

## Web App

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Docker

Docker Desktop on macOS does not expose the Apple Silicon MPS GPU to Linux containers. Use the native Python setup above for Mac GPU acceleration. The Docker path is useful for a reproducible CPU-only API/CLI environment.

Build and run the web app:

```bash
docker compose up --build web
```

Open `http://127.0.0.1:8000`.

Run the CLI in Docker:

```bash
mkdir -p inputs outputs
cp /path/to/input.jpg inputs/input.jpg

docker compose run --rm cli edit inputs/input.jpg \
  "make the sky dramatic but keep the buildings unchanged" \
  --output outputs/edited.png \
  --steps 20 \
  --show-planned-prompt
```

Run video generation in Docker:

```bash
docker compose run --rm cli video inputs/input.jpg \
  --output outputs/video.mp4 \
  --frames 14 \
  --fps 7
```

Plan a prompt without loading the image model:

```bash
docker compose run --rm cli plan "make the product photo brighter"
```

Compose keeps model downloads in named volumes:

```bash
docker volume ls | grep mac-local-image-editor
```

## Optional Prompt Expansion With Ollama

Install and run Ollama locally, then set:

```bash
export PROMPT_PLANNER=ollama
export OLLAMA_MODEL=llama3.1
```

Without Ollama, the app uses a deterministic local planner that rewrites short user requests into editing-focused instructions.
The rule-based planner also adds anatomy-aware guidance when prompts mention faces or body parts, helping the image model localize edits while preserving identity, pose, proportions, hands, limbs, skin texture, and untargeted features.

In Docker Compose, Ollama running on the Mac host is available at `http://host.docker.internal:11434`, which is the default Compose `OLLAMA_BASE_URL`.

## Runtime Configuration

Environment variables:

```bash
EDIT_MODEL_ID=diffusers/sdxl-instructpix2pix-768
VIDEO_MODEL_ID=stabilityai/stable-video-diffusion-img2vid-xt
PROMPT_PLANNER=rules
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1
MAX_IMAGE_SIDE=768
```

For smaller Macs, reduce `MAX_IMAGE_SIDE` to `512`. Larger images need more unified memory.
The app automatically uses the SDXL InstructPix2Pix pipeline when `EDIT_MODEL_ID` contains `sdxl`; otherwise it falls back to the legacy Stable Diffusion InstructPix2Pix pipeline.

## API

```bash
curl -X POST http://127.0.0.1:8000/api/edit \
  -F "image=@input.jpg" \
  -F "prompt=make the sky dramatic but keep the buildings unchanged" \
  -F "steps=20" \
  -F "guidance_scale=3.0" \
  --output edited.png
```

Generate an MP4 video:

```bash
curl -X POST http://127.0.0.1:8000/api/video \
  -F "image=@input.jpg" \
  -F "frames=14" \
  -F "fps=7" \
  --output video.mp4
```

## Notes

- The first request is slow because model weights are loaded lazily.
- Video generation uses a separate model and has its own first-run download.
- The model runs locally; no image is sent to a hosted AI API.
- Hugging Face model downloads still require internet on first use unless the model is already cached.
- Docker runs CPU-only on macOS; run natively for PyTorch MPS acceleration.
