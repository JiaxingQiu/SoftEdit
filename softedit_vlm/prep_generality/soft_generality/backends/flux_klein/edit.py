"""FLUX.2-klein-9B wrapper: one edit + latent interpolation for strength spectrum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import numpy as np
from PIL import Image

from .config import DEFAULT_MODEL_ID, DEFAULT_SEED, DEFAULT_SIZE, DEFAULT_STEPS, DEFAULT_STRENGTH_SWEEP


@dataclass
class EditResult:
    """Uniform result container across backends."""
    image: Image.Image
    prompt: str
    strength: float
    num_inference_steps: int
    seed: int
    height: int
    width: int


class FluxKleinEditor:
    """FLUX.2-klein-9B backend.

    Generates one full edit per label, then creates a strength spectrum
    via latent-space interpolation between original and edited image.
    """

    name = "flux_klein"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        offload: bool = True,
    ):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        from diffusers import Flux2KleinPipeline

        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def _vae_device(self) -> torch.device:
        """Get the device the VAE weights are actually on."""
        return next(self.pipe.vae.parameters()).device

    def _ensure_vae_on_gpu(self):
        """Move VAE to GPU if it was offloaded to CPU."""
        vae = self.pipe.vae
        if next(vae.parameters()).device.type == "cpu":
            vae.to(self.device)

    def _encode_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        """Encode a PIL image to VAE latent space."""
        self._ensure_vae_on_gpu()
        vae = self.pipe.vae
        processor = self.pipe.image_processor
        img_tensor = processor.preprocess(image, height=height, width=width)
        img_tensor = img_tensor.to(device=self._vae_device(), dtype=vae.dtype)
        with torch.inference_mode():
            latent = vae.encode(img_tensor).latent_dist.sample()
        return latent

    def _decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode a latent tensor back to PIL image."""
        self._ensure_vae_on_gpu()
        vae = self.pipe.vae
        processor = self.pipe.image_processor
        latent = latent.to(device=self._vae_device(), dtype=vae.dtype)
        with torch.inference_mode():
            image_tensor = vae.decode(latent, return_dict=False)[0]
        return processor.postprocess(image_tensor, output_type="pil")[0]

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        *,
        num_inference_steps: int = DEFAULT_STEPS,
        height: int = DEFAULT_SIZE,
        width: int = DEFAULT_SIZE,
        seed: int = DEFAULT_SEED,
    ) -> EditResult:
        """Generate one full edit (t=1.0)."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            out = self.pipe(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=1.0,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
            )
        return EditResult(
            image=out.images[0],
            prompt=prompt,
            strength=1.0,
            num_inference_steps=num_inference_steps,
            seed=seed,
            height=height,
            width=width,
        )

    def interpolate(
        self,
        original: Image.Image,
        edited: Image.Image,
        t: float,
        height: int = DEFAULT_SIZE,
        width: int = DEFAULT_SIZE,
    ) -> Image.Image:
        """Interpolate between original and edited in VAE latent space."""
        z_orig = self._encode_image(original, height, width)
        z_edit = self._encode_image(edited, height, width)
        z_interp = (1.0 - t) * z_orig + t * z_edit
        return self._decode_latent(z_interp)

    def strength_sweep(
        self,
        image: Image.Image,
        prompt: str,
        *,
        strengths: Sequence[float] = DEFAULT_STRENGTH_SWEEP,
        num_inference_steps: int = DEFAULT_STEPS,
        height: int = DEFAULT_SIZE,
        width: int = DEFAULT_SIZE,
        seed: int = DEFAULT_SEED,
        _log_prefix: str = "",
    ) -> list[EditResult]:
        """One generation + latent interpolation at each strength level."""
        print(f"{_log_prefix}  generating full edit...")
        full_edit = self.edit(
            image,
            prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            seed=seed,
        )

        original_resized = image.resize((width, height), Image.Resampling.LANCZOS)
        results = []
        for i, t in enumerate(strengths, 1):
            if t >= 1.0:
                interp_img = full_edit.image
            else:
                print(f"{_log_prefix}  interpolating t={t}...")
                interp_img = self.interpolate(original_resized, full_edit.image, t, height, width)
            results.append(
                EditResult(
                    image=interp_img,
                    prompt=prompt,
                    strength=t,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    height=height,
                    width=width,
                )
            )
        print(f"{_log_prefix}  done ({len(results)} images).")
        return results

    def strength_titles(self, results: list[EditResult]) -> list[str]:
        return [f"t={r.strength}" for r in results]
