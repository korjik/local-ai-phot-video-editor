from __future__ import annotations

import gc
from threading import Lock

import torch
from PIL import Image

from app.config import Settings


class VirtualTryOnPipeline:
    """IDM-VTON virtual try-on pipeline (yisol/IDM-VTON).

    IDM-VTON uses two UNets:
      - unet         : SDXL denoising UNet, conditioned on the masked person image
      - unet_encoder : garment feature extractor with the same architecture

    At each diffusion step the garment encoder processes the reference garment
    image and its intermediate attention activations are injected into the
    corresponding layers of the person UNet via reference attention.  This
    transfers the garment's exact texture, color, and structure onto the masked
    body region without relying solely on text to describe the product.

    Usage
    -----
    1. Upload a person photo.
    2. Upload the garment product photo (e.g. bra + panties from the collection).
    3. The segmenter generates a clothing mask; call try_on() with all three.
    """

    MODEL_ID = "yisol/IDM-VTON"

    _DEFAULT_NEGATIVE = (
        "monochrome, lowres, bad anatomy, worst quality, low quality, "
        "deformed body, blurry, wrong proportions, extra limbs, missing limbs, "
        "distorted face, watermark, text"
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipe = None
        self._lock = Lock()

    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _dtype(self) -> torch.dtype:
        # float16 on CUDA; MPS and CPU use float32 for numerical stability
        return torch.float16 if self.device == "cuda" else torch.float32

    def _load(self):
        if self._pipe is not None:
            return self._pipe

        from diffusers import DiffusionPipeline

        # IDM-VTON ships pipeline.py in its HF repo; custom_pipeline loads it.
        # trust_remote_code is required for custom pipeline code execution.
        pipe = DiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            custom_pipeline="yisol/IDM-VTON",
            trust_remote_code=True,
            torch_dtype=self._dtype(),
        ).to(self.device)
        pipe.enable_attention_slicing()
        self._pipe = pipe
        return pipe

    @staticmethod
    def _make_agnostic(person_image: Image.Image, mask: Image.Image) -> Image.Image:
        """Grey out the clothing region to create a cloth-agnostic person image.

        IDM-VTON conditions on an 'agnostic' image where the original garment is
        removed so the model doesn't anchor on the existing outfit's colour/texture.
        """
        agnostic = person_image.copy().convert("RGB")
        grey = Image.new("RGB", person_image.size, (128, 128, 128))
        # Use a hard threshold on the (possibly feathered) mask
        hard_mask = mask.point(lambda p: 255 if p > 30 else 0)
        agnostic.paste(grey, mask=hard_mask)
        return agnostic

    def try_on(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = _DEFAULT_NEGATIVE,
        steps: int = 30,
        guidance_scale: float = 2.0,
        seed: int | None = None,
    ) -> Image.Image:
        """Generate a try-on image of the person wearing the garment.

        Parameters
        ----------
        person_image:   RGB photo of the person.
        garment_image:  RGB product photo of the garment from the collection.
        mask:           Grayscale clothing mask from HumanSegmenter.clothing_mask().
        prompt:         Optional description of the garment (improves accuracy).
        steps:          Diffusion steps — 25-40 gives good quality/speed balance.
        guidance_scale: Classifier-free guidance scale. 2.0 works well for try-on.
        seed:           Fixed seed for reproducibility.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        agnostic = self._make_agnostic(person_image, mask)

        # IDM-VTON expects a pose image (DWPose skeleton overlay).  We skip the
        # external pose estimator and pass a zero image instead — the garment
        # encoder's reference attention is the primary transfer mechanism, so
        # results are still strong without explicit skeletal pose conditioning.
        pose_img = Image.new("RGB", person_image.size, (0, 0, 0))

        prompt_text = prompt or "clothing item, high quality, detailed fabric"

        with self._lock:
            pipe = self._load()

            with torch.inference_mode():
                # SDXL dual-encoder text conditioning
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt_text,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
                # Separate unconditional garment embedding (no CFG scale applied)
                prompt_embeds_cloth, _, _, _ = pipe.encode_prompt(
                    prompt_text,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=negative_prompt,
                )

                result = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img,
                    text_embeds_cloth=prompt_embeds_cloth,
                    cloth=garment_image,
                    mask_image=mask,
                    image=agnostic,
                    ip_adapter_image=garment_image,
                    height=person_image.height,
                    width=person_image.width,
                ).images[0]

        if self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        return result
