"""Inference: generate images from a trained Flux LoRA using diffusers.

The FluxPipeline is heavy to load (~24 GB base model), so we keep a single
process-wide instance alive and just swap LoRA adapters in/out between requests.
On a 5090 (32 GB) the bf16 pipeline fits with model-CPU-offload disabled; we
enable offload only as a safety net if CUDA OOMs.
"""
from __future__ import annotations

import argparse
import gc
import textwrap
import time
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from .config import paths, settings

_pipe = None
_current_lora: Optional[str] = None


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch

    dtype = torch.bfloat16
    if settings.inference_backend == "sdturbo":
        # Small, fast model to validate the real inference path on a modest GPU.
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            cache_dir=str(paths.models),
        )
    else:
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            settings.base_model,
            torch_dtype=dtype,
            token=settings.hf_token or None,
            cache_dir=str(paths.models),
        )

    if torch.cuda.is_available():
        try:
            pipe = pipe.to("cuda")
        except torch.cuda.OutOfMemoryError:
            # Fall back to sequential offload on tighter memory.
            pipe.enable_model_cpu_offload()
    _pipe = pipe
    return _pipe


def _mock_image(prompt: str, width: int, height: int, lora_path: Optional[str]) -> Image.Image:
    """A lightweight placeholder so the full UI flow works with no model/GPU."""
    # Diagonal gradient background (deterministic-ish from the prompt).
    seed = sum(ord(c) for c in prompt)
    c1 = ((seed * 7) % 200 + 30, (seed * 13) % 200 + 30, (seed * 17) % 200 + 30)
    c2 = ((seed * 5) % 120 + 10, (seed * 11) % 120 + 10, (seed * 19) % 120 + 10)
    img = Image.new("RGB", (width, height), c1)
    top = Image.new("RGB", (width, height), c2)
    mask = Image.new("L", (width, height))
    mask.putdata([int(255 * (x + y) / (width + height))
                  for y in range(height) for x in range(width)])
    img = Image.composite(top, img, mask)

    draw = ImageDraw.Draw(img)
    draw.text((24, 24), "MOCK MODE - no real model", fill=(255, 255, 255))
    lora_label = Path(lora_path).stem if lora_path else "base model"
    draw.text((24, 48), f"LoRA: {lora_label}", fill=(220, 220, 220))
    y = height // 2 - 40
    for line in textwrap.wrap(prompt, width=max(20, width // 14))[:8]:
        draw.text((24, y), line, fill=(255, 255, 255))
        y += 20
    return img


def _ensure_lora(lora_path: Optional[str]) -> None:
    """Load the requested LoRA, unloading any previously-active one first."""
    global _current_lora
    # Flux LoRAs only apply to the Flux pipeline; the sdturbo backend ignores them.
    if settings.inference_backend != "flux":
        return
    if lora_path == _current_lora:
        return

    pipe = _get_pipe()
    try:
        pipe.unload_lora_weights()
    except Exception:  # noqa: BLE001 — nothing loaded yet is fine
        pass

    if lora_path:
        p = Path(lora_path)
        if not p.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
        pipe.load_lora_weights(str(p.parent), weight_name=p.name)
    _current_lora = lora_path


def generate(
    prompt: str,
    *,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
) -> dict:
    """Generate one image and save it under DATA_DIR/generated. Returns metadata."""
    paths.ensure()

    if settings.mock:
        # No torch, no model — just a placeholder so the UI flow is exercisable.
        time.sleep(0.6)  # tiny delay so the loading overlay is visible
        image = _mock_image(prompt, width, height, lora_path)
    else:
        import torch

        pipe = _get_pipe()
        _ensure_lora(lora_path)

        generator = None
        if seed is not None:
            generator = torch.Generator(
                device="cuda" if torch.cuda.is_available() else "cpu"
            ).manual_seed(seed)

        if settings.inference_backend == "sdturbo":
            # SD-Turbo is distilled: no CFG, very few steps.
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=max(1, min(steps, 4)),
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
        else:
            cross_attention_kwargs = {"scale": lora_scale} if lora_path else None
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                joint_attention_kwargs=cross_attention_kwargs,
            ).images[0]

    image_id = uuid.uuid4().hex
    out_path = paths.generated / f"{image_id}.png"
    image.save(out_path)

    return {
        "image_id": image_id,
        "path": str(out_path),
        "filename": out_path.name,
        "prompt": prompt,
        "lora": lora_path,
        "lora_scale": lora_scale,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "created": time.time(),
    }


def free_memory() -> None:
    """Drop the pipeline and reclaim VRAM (used when switching tasks)."""
    global _pipe, _current_lora
    _pipe = None
    _current_lora = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an image from a Flux LoRA.")
    ap.add_argument("prompt")
    ap.add_argument("--lora", help="Path to a trained LoRA .safetensors")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=3.5)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    result = generate(
        args.prompt, lora_path=args.lora, lora_scale=args.scale,
        width=args.width, height=args.height, steps=args.steps,
        guidance_scale=args.guidance, seed=args.seed,
    )
    print(f"Saved -> {result['path']}")


if __name__ == "__main__":
    main()
