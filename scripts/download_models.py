"""Pre-download the base model (and optionally Florence-2) into the data cache.

Optional, but handy: run this once after setup so the big FLUX.1-dev download
happens up front instead of during your first training/generation.

    python -m scripts.download_models
"""
from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from pipeline.config import paths, settings


def main() -> None:
    paths.ensure()
    if not settings.hf_token:
        print("WARNING: HF_TOKEN not set — FLUX.1-dev is gated and will fail.",
              file=sys.stderr)

    print(f"Downloading {settings.base_model} -> {paths.models} …")
    snapshot_download(
        repo_id=settings.base_model,
        cache_dir=str(paths.models),
        token=settings.hf_token or None,
    )
    print("Downloading microsoft/Florence-2-large (captioning) …")
    snapshot_download(
        repo_id="microsoft/Florence-2-large",
        cache_dir=str(paths.models),
    )
    print("Done.")


if __name__ == "__main__":
    main()
