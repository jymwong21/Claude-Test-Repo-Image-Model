"""Shared configuration & filesystem layout for Flux LoRA Studio.

Everything the app reads/writes lives under DATA_DIR so it can sit on a RunPod
persistent volume (/workspace). Import `settings` and `paths` from here rather
than hard-coding directories anywhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo root (…/pipeline/config.py -> repo root is two levels up).
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    hf_token: str = os.environ.get("HF_TOKEN", "")
    base_model: str = os.environ.get("BASE_MODEL", "black-forest-labs/FLUX.1-dev")
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("DATA_DIR", str(REPO_ROOT / "data"))
        ).expanduser()
    )
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    caption_backend: str = os.environ.get("CAPTION_BACKEND", "florence2")

    # Path to the cloned ai-toolkit checkout (created by runpod/setup.sh).
    ai_toolkit_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AI_TOOLKIT_DIR", str(REPO_ROOT / "ai-toolkit"))
        ).expanduser()
    )


settings = Settings()


@dataclass(frozen=True)
class Paths:
    """Concrete directories derived from DATA_DIR."""

    root: Path

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"  # one subdir per project (the training images + .txt captions)

    @property
    def output(self) -> Path:
        return self.root / "output"  # ai-toolkit writes LoRA checkpoints here, per project

    @property
    def models(self) -> Path:
        return self.root / "models"  # HF cache / base model

    @property
    def generated(self) -> Path:
        return self.root / "generated"  # inference results

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"  # training job metadata + logs

    @property
    def configs(self) -> Path:
        return self.root / "configs"  # generated ai-toolkit yaml configs

    def project_dataset(self, project: str) -> Path:
        return self.datasets / project

    def project_output(self, project: str) -> Path:
        return self.output / project

    def ensure(self) -> None:
        for p in (self.datasets, self.output, self.models, self.generated,
                  self.jobs, self.configs):
            p.mkdir(parents=True, exist_ok=True)


paths = Paths(settings.data_dir)

# The bundled ai-toolkit config template used as the base for every run.
TRAIN_CONFIG_TEMPLATE = REPO_ROOT / "pipeline" / "configs" / "flux_lora_5090.yaml"
