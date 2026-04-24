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

_CLOTHING_TERMS = (
    "shirt",
    "t-shirt",
    "tshirt",
    "blouse",
    "top",
    "jacket",
    "coat",
    "blazer",
    "sweater",
    "sweatshirt",
    "hoodie",
    "cardigan",
    "vest",
    "pants",
    "jeans",
    "trousers",
    "leggings",
    "shorts",
    "skirt",
    "dress",
    "gown",
    "suit",
    "uniform",
    "outfit",
    "clothes",
    "clothing",
    "wear",
    "wearing",
    "fabric",
    "sleeve",
    "collar",
    "undershirt",
    "tank",
    "crop",
    "turtleneck",
    "scarf",
    "hat",
    "cap",
    "boot",
    "boots",
    "sneakers",
    "shoes",
    "shoe",
    # lingerie / underwear / swimwear
    "bra",
    "bralette",
    "panties",
    "panty",
    "underwear",
    "lingerie",
    "bikini",
    "thong",
    "briefs",
    "camisole",
    "bodysuit",
    "swimsuit",
    "swimwear",
    "corset",
    "nightgown",
    "negligee",
)

# Terms indicating upper-body garments
_UPPER_CLOTHING_TERMS = (
    "shirt", "t-shirt", "tshirt", "blouse", "top", "jacket", "coat", "blazer",
    "sweater", "sweatshirt", "hoodie", "cardigan", "vest", "undershirt", "tank",
    "crop", "turtleneck", "bra", "bralette", "camisole", "bodysuit", "corset",
    "negligee", "nightgown",
)

# Terms indicating lower-body garments
_LOWER_CLOTHING_TERMS = (
    "pants", "jeans", "trousers", "leggings", "shorts", "skirt", "thong",
    "panties", "panty", "underwear", "briefs", "swimsuit", "swimwear",
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
    mentions_cloth = _contains_term(lower, _CLOTHING_TERMS)

    if is_style:
        suffix = ", apply style consistently, preserve subject"
    elif mentions_face and (mentions_body or mentions_cloth):
        suffix = ", preserve identity, other facial features, body proportions, and pose"
    elif mentions_face:
        suffix = ", preserve identity, expression, and all other facial features"
    elif mentions_cloth:
        suffix = ", preserve face, skin, hair, pose, and body proportions"
    elif mentions_body:
        suffix = ", preserve body proportions, pose, and natural anatomy"
    else:
        suffix = ", preserve composition, subject, and details"

    return f"{cleaned}{suffix}"


def mentions_clothing(prompt: str) -> bool:
    """Return True if the prompt refers to clothing items or actions."""
    return _contains_term(prompt.lower(), _CLOTHING_TERMS)


def mentions_full_outfit_swap(prompt: str) -> bool:
    """Return True if the prompt mentions both upper- and lower-body garments.

    Used to detect requests like 'blazer and pants → bra and panties' so the
    edit pipeline can apply more aggressive parameters and gap-filling masks.
    """
    lower = prompt.lower()
    return _contains_term(lower, _UPPER_CLOTHING_TERMS) and _contains_term(lower, _LOWER_CLOTHING_TERMS)


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
