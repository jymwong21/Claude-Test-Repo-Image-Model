"""Dataset preparation for Flux LoRA training.

Takes raw user uploads and turns them into a clean training folder:
  • converts to RGB, strips EXIF/orientation issues
  • downscales oversized images (keeps long edge <= max_edge) so latent caching
    is fast and consistent
  • re-encodes to high-quality PNG with sequential names
  • reports anything too small to be useful

ai-toolkit reads images + sidecar `<name>.txt` caption files from one folder, so
that's exactly what we produce here. Captions are added separately by caption.py.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from .config import paths

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MIN_EDGE = 512        # Flux trains at 512–1024; smaller than this hurts quality
DEFAULT_MAX_EDGE = 1536


def _iter_images(src: Path):
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            yield p


def prepare(
    source: Path,
    project: str,
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    clear_existing: bool = False,
) -> dict:
    """Process every image in `source` into the project's dataset folder.

    Returns a summary dict: {processed, skipped_small, warnings, dataset_dir}.
    """
    dest = paths.project_dataset(project)
    if clear_existing and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    processed, skipped_small, warnings = 0, 0, []

    # Continue numbering after any images already in the folder so repeated
    # uploads append instead of clobbering 001.png.
    existing = [
        int(p.stem) for p in dest.glob("*.png") if p.stem.isdigit()
    ]
    index = (max(existing) + 1) if existing else 1

    for img_path in _iter_images(Path(source)):
        try:
            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im)   # honour camera rotation
                im = im.convert("RGB")

                w, h = im.size
                if min(w, h) < MIN_EDGE:
                    skipped_small += 1
                    warnings.append(
                        f"{img_path.name}: {w}x{h} is below {MIN_EDGE}px — skipped"
                    )
                    continue

                longest = max(w, h)
                if longest > max_edge:
                    scale = max_edge / longest
                    im = im.resize(
                        (round(w * scale), round(h * scale)), Image.LANCZOS
                    )

                out = dest / f"{index:03d}.png"
                im.save(out, format="PNG")
                processed += 1
                index += 1
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the batch
            warnings.append(f"{img_path.name}: failed to process ({exc})")

    if processed < 5:
        warnings.append(
            f"Only {processed} usable image(s). 15–30 varied photos give the "
            f"best LoRA — consider adding more."
        )

    return {
        "project": project,
        "dataset_dir": str(dest),
        "processed": processed,
        "skipped_small": skipped_small,
        "warnings": warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a Flux LoRA training dataset.")
    ap.add_argument("source", type=Path, help="Folder of raw images")
    ap.add_argument("project", help="Project name (training folder)")
    ap.add_argument("--max-edge", type=int, default=DEFAULT_MAX_EDGE)
    ap.add_argument("--clear", action="store_true", help="Wipe existing dataset first")
    args = ap.parse_args()

    paths.ensure()
    result = prepare(
        args.source, args.project, max_edge=args.max_edge, clear_existing=args.clear
    )
    print(f"Processed {result['processed']} image(s) -> {result['dataset_dir']}")
    if result["skipped_small"]:
        print(f"Skipped {result['skipped_small']} too-small image(s).")
    for w in result["warnings"]:
        print(f"  ⚠ {w}")


if __name__ == "__main__":
    main()
