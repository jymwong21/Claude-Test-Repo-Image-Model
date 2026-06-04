"""Auto-captioning for Flux LoRA datasets using Florence-2.

For a personal/character LoRA the recommended caption format is:
    "<trigger_word>, <natural description of the rest of the scene>"
The trigger word (e.g. "ohwx person") is what you'll later put in generation
prompts to invoke your subject. Florence-2 describes the scene; we prepend the
trigger and write one `<image>.txt` sidecar per image, which is exactly what
ai-toolkit consumes.

This uses the NATIVE transformers Florence-2 integration (transformers >= 5,
repo `florence-community/Florence-2-large`): no `trust_remote_code`, so it stays
compatible with the transformers 5.x that ai-toolkit pins (the old remote-code
path breaks on transformers >= 4.50). Florence-2-large is small (~0.7 B) and
loads quickly even alongside Flux.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .config import paths, settings

_MODEL_ID = "florence-community/Florence-2-large"
_TASK = "<DETAILED_CAPTION>"

# Deterministic stub descriptions used in mock mode (no Florence-2 download).
_MOCK_DESCRIPTIONS = [
    "a close-up portrait of a person, soft natural lighting, neutral background",
    "a person smiling at the camera, outdoor daylight, shallow depth of field",
    "a half-body photo of a person wearing casual clothing, indoor setting",
    "a candid photo of a person, side profile, warm tones",
    "a portrait of a person with a plain background, even studio lighting",
]

# Lazily-loaded singletons so repeated calls (e.g. from the web backend) reuse
# the model instead of reloading it every time.
_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    # Native integration — no trust_remote_code. `dtype=` is the transformers 5.x
    # kwarg (it replaces the old `torch_dtype`).
    _model = Florence2ForConditionalGeneration.from_pretrained(
        _MODEL_ID, dtype=dtype
    ).to(device)
    _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    _model.eval()
    return _model, _processor


def _caption_one(image: Image.Image) -> str:
    import torch

    model, processor = _load_model()
    device = model.device
    dtype = next(model.parameters()).dtype

    inputs = processor(text=_TASK, images=image, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=_TASK, image_size=(image.width, image.height)
    )
    return parsed[_TASK].strip()


def caption_project(
    project: str,
    *,
    trigger_word: str = "",
    overwrite: bool = False,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Generate `.txt` captions for every image in a project's dataset folder.

    `trigger_word` is prepended to each caption. `progress(done, total, name)` is
    called after each image if provided (used by the web backend for live UI).
    """
    dataset_dir = paths.project_dataset(project)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No dataset for project '{project}' at {dataset_dir}")

    images = sorted(
        p for p in dataset_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    total = len(images)
    written, skipped = 0, 0

    for i, img_path in enumerate(images, start=1):
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            skipped += 1
            if progress:
                progress(i, total, img_path.name)
            continue

        if settings.mock:
            # No Florence-2 — deterministic stub so the flow works offline.
            time.sleep(0.3)
            description = _MOCK_DESCRIPTIONS[(i - 1) % len(_MOCK_DESCRIPTIONS)]
        else:
            with Image.open(img_path) as im:
                description = _caption_one(im.convert("RGB"))

        caption = f"{trigger_word}, {description}" if trigger_word else description
        txt_path.write_text(caption, encoding="utf-8")
        written += 1
        if progress:
            progress(i, total, img_path.name)

    return {
        "project": project,
        "total": total,
        "written": written,
        "skipped": skipped,
        "trigger_word": trigger_word,
    }


def read_captions(project: str) -> list[dict]:
    """Return [{image, caption}] for the project's dataset (for UI review/edit)."""
    dataset_dir = paths.project_dataset(project)
    if not dataset_dir.exists():
        return []
    out = []
    for img_path in sorted(dataset_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        txt = img_path.with_suffix(".txt")
        out.append({
            "image": img_path.name,
            "caption": txt.read_text(encoding="utf-8").strip() if txt.exists() else "",
        })
    return out


def write_caption(project: str, image_name: str, caption: str) -> None:
    """Persist a (manually edited) caption for one image."""
    dataset_dir = paths.project_dataset(project)
    txt_path = (dataset_dir / image_name).with_suffix(".txt")
    txt_path.write_text(caption.strip(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-caption a Flux LoRA dataset.")
    ap.add_argument("project", help="Project name")
    ap.add_argument("--trigger", default="", help="Trigger word, e.g. 'ohwx person'")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    def _print(done, total, name):
        print(f"  [{done}/{total}] {name}")

    result = caption_project(
        args.project, trigger_word=args.trigger,
        overwrite=args.overwrite, progress=_print,
    )
    print(f"Captioned {result['written']} image(s), skipped {result['skipped']}.")


if __name__ == "__main__":
    main()
