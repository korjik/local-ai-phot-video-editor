from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None

if register_heif_opener is not None:
    register_heif_opener()


def load_image(data: bytes, max_side: int) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(
            "Could not decode the uploaded image. Use PNG, JPEG, WebP, or HEIC/HEIF."
        ) from exc

    width, height = rgb.size
    longest = max(width, height)
    if longest <= max_side:
        return rgb

    scale = max_side / longest
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return rgb.resize(size, Image.Resampling.LANCZOS)


def png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
