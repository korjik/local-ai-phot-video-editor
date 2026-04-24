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
    cleaned = clean_prompt(prompt)
    lower = cleaned.lower()

    preservation = "Preserve the original composition, identity, geometry, and important details."
    quality = "Make the edit natural, coherent, and photorealistic unless the user asks otherwise."
    anatomy = ""

    if any(word in lower for word in ("cartoon", "anime", "painting", "sketch", "illustration")):
        quality = "Apply the requested visual style consistently while preserving the main subject."

    mentions_face = _contains_term(lower, _FACE_TERMS)
    mentions_body = _contains_term(lower, _BODY_TERMS)
    if mentions_face or mentions_body:
        anatomy_parts = [
            "Recognize people, faces, and body parts as structured anatomy.",
            "Apply the requested change only to the named face or body part.",
            "Keep untargeted facial features, limbs, hands, skin texture, pose, proportions, and clothing unchanged.",
        ]
        if mentions_face:
            anatomy_parts.append("Preserve the person's identity, expression, gaze, and facial symmetry.")
        if mentions_body:
            anatomy_parts.append("Preserve natural joints, fingers, limb count, posture, and body proportions.")
        anatomy = " " + " ".join(anatomy_parts)

    return f"{cleaned}. {preservation} {quality}{anatomy}"


async def ollama_plan(prompt: str, settings: Settings) -> str:
    import httpx

    cleaned = clean_prompt(prompt)
    instruction = (
        "Rewrite this image-editing request into one concise instruction for an image-to-image "
        "diffusion editor. Preserve user intent. Include what should stay unchanged. "
        "When the request mentions a face or body part, name that part explicitly and preserve "
        "identity, pose, proportions, skin texture, hands, limbs, and untargeted anatomy. "
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
