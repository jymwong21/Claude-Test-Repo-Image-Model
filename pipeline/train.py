"""Training orchestration: build an ai-toolkit config and launch a run.

This module is deliberately thin. The actual training is done by ai-toolkit's
`run.py`; we just translate the user's choices into a config YAML and hand it
over. `backend/jobs.py` owns the subprocess lifecycle (so it can stream logs and
track status); a CLI `main()` is provided for running directly on the pod.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .config import TRAIN_CONFIG_TEMPLATE, paths, settings


@dataclass
class TrainParams:
    project: str
    trigger_word: str = "ohwx person"
    steps: int = 2000
    learning_rate: float = 1e-4
    rank: int = 16                 # LoRA rank (linear + alpha)
    save_every: int = 250
    sample_every: int = 250
    base_model: str = ""           # defaults to settings.base_model
    sample_prompts: list[str] | None = None

    def resolved_base_model(self) -> str:
        return self.base_model or settings.base_model

    def resolved_sample_prompts(self) -> list[str]:
        if self.sample_prompts:
            return self.sample_prompts
        t = self.trigger_word or "the subject"
        return [
            f"{t}, professional headshot, soft studio lighting, sharp focus",
            f"{t} walking through a sunlit city street, candid photo, 35mm",
            f"{t} as an oil painting portrait, dramatic chiaroscuro lighting",
        ]


def build_training_config(params: TrainParams) -> Path:
    """Render the ai-toolkit YAML for this project and return its path."""
    with open(TRAIN_CONFIG_TEMPLATE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_dir = paths.project_dataset(params.project)
    output_dir = paths.output  # ai-toolkit nests <output>/<name>/ itself

    proc = cfg["config"]["process"][0]
    cfg["config"]["name"] = params.project
    cfg["meta"]["name"] = params.project

    proc["training_folder"] = str(output_dir)
    proc["trigger_word"] = params.trigger_word or None
    proc["network"]["linear"] = params.rank
    proc["network"]["linear_alpha"] = params.rank
    proc["save"]["save_every"] = params.save_every
    proc["datasets"][0]["folder_path"] = str(dataset_dir)
    proc["train"]["steps"] = params.steps
    proc["train"]["lr"] = params.learning_rate
    proc["model"]["name_or_path"] = params.resolved_base_model()
    proc["sample"]["sample_every"] = params.sample_every
    proc["sample"]["prompts"] = params.resolved_sample_prompts()

    paths.ensure()
    out_path = paths.configs / f"{params.project}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def build_command(config_path: Path) -> list[str]:
    """The command that runs ai-toolkit against a generated config."""
    run_py = settings.ai_toolkit_dir / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(
            f"ai-toolkit not found at {run_py}. Run runpod/setup.sh first."
        )
    return [sys.executable, str(run_py), str(config_path)]


def lora_output_path(project: str) -> Path:
    """Where the final LoRA safetensors is expected after training."""
    return paths.project_output(project) / f"{project}.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Flux LoRA via ai-toolkit.")
    ap.add_argument("project")
    ap.add_argument("--trigger", default="ohwx person")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--print-config", action="store_true",
                    help="Only build & print the config path, don't train")
    args = ap.parse_args()

    params = TrainParams(
        project=args.project, trigger_word=args.trigger,
        steps=args.steps, learning_rate=args.lr, rank=args.rank,
    )
    config_path = build_training_config(params)
    print(f"Wrote config: {config_path}")
    print(f"Params: {asdict(params)}")
    if args.print_config:
        return

    import subprocess
    cmd = build_command(config_path)
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(settings.ai_toolkit_dir)))


if __name__ == "__main__":
    main()
