"""Inference: generate images from a trained Flux LoRA using diffusers.

The FluxPipeline is heavy to load (~24 GB base model), so we keep a single
process-wide instance alive and just swap LoRA adapters in/out between requests.
On a 5090 (32 GB) the bf16 pipeline fits with model-CPU-offload disabled; we
enable offload only as a safety net if CUDA OOMs.
"""
from __future__ import annotations

import argparse
import gc
import time
import uuid
from pathlib import Path
from typing import Optional

from .config import paths, settings

_pipe = None
_current_lora: Optional[str] = None


def _get_pipe():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        settings.base_model,
        torch_dtype=torch.bfloat16,
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


def _ensure_lora(lora_path: Optional[str]) -> None:
    """Load the requested LoRA, unloading any previously-active one first."""
    global _current_lora
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
    import torch

    pipe = _get_pipe()
    _ensure_lora(lora_path)

    generator = None
    if seed is not None:
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)

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

    paths.ensure()
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
