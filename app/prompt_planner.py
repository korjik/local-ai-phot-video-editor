from __future__ import annotations

import re

from app.config import Settings


_SPACE_RE = re.compile(r"\s+")


def clean_prompt(prompt: str) -> str:
    cleaned = _SPACE_RE.sub(" ", prompt.strip())
    if not cleaned:
        raise ValueError("Prompt cannot be empty.")
    return cleaned


def rules_plan(prompt: str) -> str:
    cleaned = clean_prompt(prompt)
    lower = cleaned.lower()

    preservation = "Preserve the original composition, identity, geometry, and important details."
    quality = "Make the edit natural, coherent, and photorealistic unless the user asks otherwise."

    if any(word in lower for word in ("cartoon", "anime", "painting", "sketch", "illustration")):
        quality = "Apply the requested visual style consistently while preserving the main subject."

    return f"{cleaned}. {preservation} {quality}"


async def ollama_plan(prompt: str, settings: Settings) -> str:
    import httpx

    cleaned = clean_prompt(prompt)
    instruction = (
        "Rewrite this image-editing request into one concise instruction for an image-to-image "
        "diffusion editor. Preserve user intent. Include what should stay unchanged. "
        "Do not add unrelated objects or explanations.\n\n"
        f"User request: {cleaned}"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={"model": settings.ollama_model, "prompt": instruction, "stream": False},
        )
        response.raise_for_status()
        planned = response.json().get("response", "")
    return rules_plan(planned or cleaned)


async def plan_prompt(prompt: str, settings: Settings) -> str:
    if settings.prompt_planner == "ollama":
        return await ollama_plan(prompt, settings)
    return rules_plan(prompt)
