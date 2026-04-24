from __future__ import annotations

import re

from app.config import Settings


_SPACE_RE = re.compile(r"\s+")

_FACE_TERMS = (
    "face",
    "faces",
    "facial",
    "head",
    "hair",
    "eye",
    "eyes",
    "eyebrow",
    "eyebrows",
    "eyelash",
    "eyelashes",
    "nose",
    "mouth",
    "lip",
    "lips",
    "teeth",
    "smile",
    "jaw",
    "chin",
    "cheek",
    "cheeks",
    "forehead",
    "ear",
    "ears",
    "beard",
    "mustache",
)

_BODY_TERMS = (
    "body",
    "bodies",
    "person",
    "people",
    "human",
    "pose",
    "posture",
    "torso",
    "shoulder",
    "shoulders",
    "arm",
    "arms",
    "elbow",
    "elbows",
    "wrist",
    "wrists",
    "hand",
    "hands",
    "finger",
    "fingers",
    "thumb",
    "thumbs",
    "leg",
    "legs",
    "knee",
    "knees",
    "ankle",
    "ankles",
    "foot",
    "feet",
    "skin",
    "neck",
    "waist",
    "hip",
    "hips",
)


def _contains_term(prompt: str, terms: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", prompt) for term in terms)


def clean_prompt(prompt: str) -> str:
    cleaned = _SPACE_RE.sub(" ", prompt.strip())
    if not cleaned:
        raise ValueError("Prompt cannot be empty.")
    return cleaned


def rules_plan(prompt: str) -> str:
    """Build a compact prompt that fits within CLIP's 77-token limit.

    The user instruction comes first (highest priority). A short preservation
    suffix follows so the anatomy/composition hints are never truncated.
    """
    cleaned = clean_prompt(prompt)
    lower = cleaned.lower()

    is_style = any(word in lower for word in ("cartoon", "anime", "painting", "sketch", "illustration"))
    mentions_face = _contains_term(lower, _FACE_TERMS)
    mentions_body = _contains_term(lower, _BODY_TERMS)

    if is_style:
        suffix = ", apply style consistently, preserve subject"
    elif mentions_face and mentions_body:
        suffix = ", preserve identity, other facial features, body proportions, and pose"
    elif mentions_face:
        suffix = ", preserve identity, expression, and all other facial features"
    elif mentions_body:
        suffix = ", preserve body proportions, pose, and natural anatomy"
    else:
        suffix = ", preserve composition, subject, and details"

    return f"{cleaned}{suffix}"


async def ollama_plan(prompt: str, settings: Settings) -> str:
    import httpx

    cleaned = clean_prompt(prompt)
    instruction = (
        "Rewrite this image-editing request into one SHORT instruction (under 15 words) for an "
        "image-to-image diffusion editor. State what to change. If a face or body part is mentioned, "
        "name it. Do not add explanations or preservation hints — those are added separately.\n\n"
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
