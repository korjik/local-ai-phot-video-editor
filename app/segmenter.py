from __future__ import annotations

from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoModelForSemanticSegmentation

# SegFormer / ImageNet preprocessing constants — hardcoded so we don't need
# torchvision (which SegformerImageProcessor pulls in transitively).
_SEGFORMER_INPUT_SIZE = 512
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ATR dataset label IDs used by mattmdjaga/segformer_b2_clothes
_LABEL_BACKGROUND = 0
_LABEL_HAT = 1
_LABEL_HAIR = 2
_LABEL_SUNGLASSES = 3
_LABEL_UPPER_CLOTHES = 4
_LABEL_SKIRT = 5
_LABEL_PANTS = 6
_LABEL_DRESS = 7
_LABEL_BELT = 8
_LABEL_LEFT_SHOE = 9
_LABEL_RIGHT_SHOE = 10
_LABEL_FACE = 11
_LABEL_LEFT_LEG = 12
_LABEL_RIGHT_LEG = 13
_LABEL_LEFT_ARM = 14
_LABEL_RIGHT_ARM = 15
_LABEL_BAG = 16
_LABEL_SCARF = 17

# Labels that confirm a person is present in the image
_PERSON_LABELS = frozenset({
    _LABEL_HAIR, _LABEL_FACE,
    _LABEL_LEFT_ARM, _LABEL_RIGHT_ARM,
    _LABEL_LEFT_LEG, _LABEL_RIGHT_LEG,
    _LABEL_UPPER_CLOTHES, _LABEL_SKIRT, _LABEL_PANTS, _LABEL_DRESS,
})

# Used to detect full-outfit swaps (blazer+pants → bra+panties etc.)
_UPPER_CLOTHING_IDS = frozenset({_LABEL_UPPER_CLOTHES, _LABEL_DRESS})
_LOWER_CLOTHING_IDS = frozenset({_LABEL_PANTS, _LABEL_SKIRT})

# Default set of clothing labels included in the edit mask
CLOTHING_LABELS: tuple[int, ...] = (
    _LABEL_UPPER_CLOTHES,
    _LABEL_SKIRT,
    _LABEL_PANTS,
    _LABEL_DRESS,
    _LABEL_BELT,
    _LABEL_SCARF,
)

# Extended set that includes accessories
ALL_CLOTHING_LABELS: tuple[int, ...] = (
    _LABEL_HAT,
    _LABEL_UPPER_CLOTHES,
    _LABEL_SKIRT,
    _LABEL_PANTS,
    _LABEL_DRESS,
    _LABEL_BELT,
    _LABEL_LEFT_SHOE,
    _LABEL_RIGHT_SHOE,
    _LABEL_SCARF,
)


def _fill_body_gap(mask_arr: np.ndarray) -> np.ndarray:
    """Fill the vertical gap between upper and lower clothing regions column-by-column.

    When a blazer and pants are both masked, there can be an unmasked strip at
    the waist/midriff. This closes that gap so the diffusion model can generate
    the full new outfit (e.g. bra + bare midriff + panties) without seams.
    """
    filled = mask_arr.copy()
    for col in range(mask_arr.shape[1]):
        rows = np.where(mask_arr[:, col] > 0)[0]
        if len(rows) >= 2:
            filled[rows[0]: rows[-1] + 1, col] = 255
    return filled


class HumanSegmenter:
    """Segments humans and their clothing using a SegFormer model.

    The model (mattmdjaga/segformer_b2_clothes) is fine-tuned on the ATR
    dataset and classifies each pixel into 18 clothing/body categories.
    It is loaded lazily on first use and cached for the lifetime of the object.
    """

    MODEL_ID = "mattmdjaga/segformer_b2_clothes"

    def __init__(self) -> None:
        self._model: AutoModelForSemanticSegmentation | None = None
        self._lock = Lock()

    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self) -> AutoModelForSemanticSegmentation:
        if self._model is None:
            model = AutoModelForSemanticSegmentation.from_pretrained(self.MODEL_ID)
            self._model = model.to(self.device).eval()
        assert self._model is not None
        return self._model

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Resize to the model input size and apply ImageNet normalization."""
        resized = image.convert("RGB").resize(
            (_SEGFORMER_INPUT_SIZE, _SEGFORMER_INPUT_SIZE),
            Image.Resampling.BILINEAR,
        )
        arr = np.asarray(resized, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    def _segment_labels(self, image: Image.Image) -> np.ndarray:
        """Return an (H, W) uint8 array of label IDs at full image resolution."""
        model = self._load()
        pixel_values = self._preprocess(image)

        with torch.inference_mode():
            logits = model(pixel_values=pixel_values).logits  # (1, num_classes, H/4, W/4)

        # Upsample to original image size
        upsampled = F.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        labels: np.ndarray = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return labels

    def has_person(self, image: Image.Image) -> bool:
        """Return True if at least one person-related label is found."""
        with self._lock:
            labels = self._segment_labels(image)
        return bool(set(np.unique(labels).tolist()) & _PERSON_LABELS)

    def clothing_mask(
        self,
        image: Image.Image,
        label_ids: tuple[int, ...] = CLOTHING_LABELS,
        dilate_px: int = 8,
        feather_px: int = 6,
        fill_gap: bool = False,
    ) -> Image.Image | None:
        """Return a grayscale PIL mask (255=clothing, 0=other), or None if no clothing found.

        dilate_px expands the mask to cover seam edges.
        feather_px blurs the boundary for smooth compositing.
        fill_gap=True fills the vertical strip between upper and lower clothing regions,
        which is needed for full-outfit swaps (e.g. blazer+pants → bra+panties) where
        the midriff/waist gap would otherwise be left untouched by the diffusion model.
        When fill_gap is True and both upper and lower garments are detected, dilate_px
        is also bumped to 20 to handle any residual edge pixels.
        """
        with self._lock:
            labels = self._segment_labels(image)

        mask_arr = np.zeros(labels.shape, dtype=np.uint8)
        for lid in label_ids:
            mask_arr[labels == lid] = 255

        if not mask_arr.any():
            return None

        if fill_gap:
            detected_ids = set(np.unique(labels).tolist())
            has_upper = bool(detected_ids & _UPPER_CLOTHING_IDS)
            has_lower = bool(detected_ids & _LOWER_CLOTHING_IDS)
            if has_upper and has_lower:
                mask_arr = _fill_body_gap(mask_arr)
                dilate_px = max(dilate_px, 20)

        mask_img = Image.fromarray(mask_arr, mode="L")

        if dilate_px > 0:
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=dilate_px * 2 + 1))
        if feather_px > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_px))

        return mask_img

    def visualize(
        self,
        image: Image.Image,
        label_ids: tuple[int, ...] = CLOTHING_LABELS,
    ) -> Image.Image:
        """Return the image with detected clothing regions highlighted in green."""
        with self._lock:
            labels = self._segment_labels(image)

        mask_arr = np.zeros(labels.shape, dtype=np.uint8)
        for lid in label_ids:
            mask_arr[labels == lid] = 200

        mask_img = Image.fromarray(mask_arr, mode="L")
        has_clothing = bool(mask_arr.any())

        base = image.convert("RGBA")
        # Green tint for clothing regions, red tint if no person/clothing found
        color = (0, 200, 80, 160) if has_clothing else (200, 40, 40, 100)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay.paste(color, mask=mask_img if has_clothing else Image.new("L", image.size, 60))
        return Image.alpha_composite(base, overlay).convert("RGB")
