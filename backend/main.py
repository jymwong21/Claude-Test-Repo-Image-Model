"""Flux LoRA Studio — FastAPI backend.

Serves the web UI and exposes the pipeline as a small JSON API:
  projects → dataset upload → caption → train (live progress) → generate.

Run on the pod with:  python -m backend.main
"""
from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import caption as caption_mod
from pipeline import inference
from pipeline import prepare_dataset
from pipeline import train as train_mod
from pipeline.config import REPO_ROOT, paths, settings

from .jobs import Status, caption_manager, training_manager

app = FastAPI(title="Flux LoRA Studio")

FRONTEND_DIR = REPO_ROOT / "frontend"
_PROJECT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _valid_project(name: str) -> str:
    if not _PROJECT_RE.match(name):
        raise HTTPException(400, "Project name must be 1–64 chars: letters, digits, - or _")
    return name


@app.on_event("startup")
async def _startup() -> None:
    paths.ensure()


# ── Models ────────────────────────────────────────────────────────────────────

class CreateProject(BaseModel):
    name: str


class CaptionRequest(BaseModel):
    trigger_word: str = "ohwx person"
    overwrite: bool = False


class EditCaption(BaseModel):
    caption: str


class TrainRequest(BaseModel):
    trigger_word: str = "ohwx person"
    steps: int = Field(2000, ge=200, le=6000)
    learning_rate: float = Field(1e-4, gt=0, le=1e-2)
    rank: int = Field(16, ge=4, le=128)


class GenerateRequest(BaseModel):
    prompt: str
    lora: Optional[str] = None        # path to a .safetensors, or None for base model
    lora_scale: float = Field(1.0, ge=0, le=2)
    width: int = Field(1024, ge=512, le=1536)
    height: int = Field(1024, ge=512, le=1536)
    steps: int = Field(28, ge=4, le=50)
    guidance_scale: float = Field(3.5, ge=0, le=10)
    seed: Optional[int] = None


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    gpu = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            gpu = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"sm_{cap[0]}{cap[1]}",
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "torch": torch.__version__,
            }
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok",
        "mock": settings.mock,
        "inference_backend": settings.inference_backend,
        "gpu": gpu,
        "hf_token_configured": bool(settings.hf_token),
        "ai_toolkit_ready": (settings.ai_toolkit_dir / "run.py").exists(),
        "base_model": settings.base_model,
    }


# ── Projects & dataset ────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects() -> list[dict]:
    out = []
    if paths.datasets.exists():
        for d in sorted(paths.datasets.iterdir()):
            if not d.is_dir():
                continue
            images = [p for p in d.iterdir() if p.suffix.lower() in
                      {".png", ".jpg", ".jpeg", ".webp"}]
            captioned = sum(1 for p in images if p.with_suffix(".txt").exists())
            out.append({
                "name": d.name,
                "image_count": len(images),
                "captioned_count": captioned,
                "lora_count": len(_list_project_loras(d.name)),
            })
    return out


@app.post("/api/projects")
async def create_project(body: CreateProject) -> dict:
    name = _valid_project(body.name)
    paths.project_dataset(name).mkdir(parents=True, exist_ok=True)
    return {"name": name}


@app.post("/api/projects/{project}/images")
async def upload_images(project: str, files: list[UploadFile] = File(...)) -> dict:
    _valid_project(project)
    tmp = Path(tempfile.mkdtemp(prefix="flux-upload-"))
    try:
        for f in files:
            if not f.content_type or not f.content_type.startswith("image/"):
                continue
            dest = tmp / Path(f.filename or "img").name
            async with aiofiles.open(dest, "wb") as out:
                await out.write(await f.read())
        result = prepare_dataset.prepare(tmp, project)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


@app.get("/api/projects/{project}/dataset")
async def get_dataset(project: str) -> dict:
    _valid_project(project)
    return {"project": project, "items": caption_mod.read_captions(project)}


@app.delete("/api/projects/{project}/images/{image_name}")
async def delete_image(project: str, image_name: str) -> dict:
    _valid_project(project)
    img = paths.project_dataset(project) / Path(image_name).name
    if img.exists():
        img.unlink()
        img.with_suffix(".txt").unlink(missing_ok=True)
        return {"deleted": image_name}
    raise HTTPException(404, "Image not found")


# ── Captioning ────────────────────────────────────────────────────────────────

@app.post("/api/projects/{project}/caption")
async def start_caption(project: str, body: CaptionRequest) -> dict:
    _valid_project(project)
    try:
        job = caption_manager.start(project, body.trigger_word, body.overwrite)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return job.to_dict()


@app.get("/api/caption/status")
async def caption_status() -> dict:
    job = caption_manager.job
    return job.to_dict() if job else {"status": Status.IDLE.value}


@app.put("/api/projects/{project}/caption/{image_name}")
async def edit_caption(project: str, image_name: str, body: EditCaption) -> dict:
    _valid_project(project)
    caption_mod.write_caption(project, Path(image_name).name, body.caption)
    return {"image": image_name, "caption": body.caption}


# ── Training ──────────────────────────────────────────────────────────────────

@app.post("/api/projects/{project}/train")
async def start_train(project: str, body: TrainRequest) -> dict:
    _valid_project(project)
    items = caption_mod.read_captions(project)
    if not items:
        raise HTTPException(400, "No images in dataset — upload photos first.")
    if any(not it["caption"] for it in items):
        raise HTTPException(400, "Some images are uncaptioned — run captioning first.")

    params = train_mod.TrainParams(
        project=project,
        trigger_word=body.trigger_word,
        steps=body.steps,
        learning_rate=body.learning_rate,
        rank=body.rank,
    )
    try:
        job = training_manager.start(params)
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(409, str(exc))
    return job.to_dict()


@app.get("/api/train/status")
async def train_status() -> dict:
    job = training_manager.job
    if not job:
        return {"status": Status.IDLE.value}
    data = job.to_dict()
    data["log_tail"] = job.tail(60)
    return data


@app.post("/api/train/cancel")
async def train_cancel() -> dict:
    return {"cancelled": training_manager.cancel()}


# ── LoRAs & generation ────────────────────────────────────────────────────────

def _list_project_loras(project: str) -> list[dict]:
    out_dir = paths.project_output(project)
    if not out_dir.exists():
        return []
    loras = []
    for sf in sorted(out_dir.rglob("*.safetensors")):
        loras.append({
            "project": project,
            "name": sf.stem,
            "path": str(sf),
            "size_mb": round(sf.stat().st_size / 1e6, 1),
            "modified": sf.stat().st_mtime,
        })
    return loras


@app.get("/api/loras")
async def list_loras() -> list[dict]:
    out = []
    if paths.output.exists():
        for d in sorted(paths.output.iterdir()):
            if d.is_dir():
                out.extend(_list_project_loras(d.name))
    out.sort(key=lambda x: x["modified"], reverse=True)
    return out


@app.post("/api/generate")
async def generate(body: GenerateRequest) -> dict:
    if training_manager.is_busy():
        raise HTTPException(409, "Training is using the GPU — wait for it to finish.")
    if body.lora and not Path(body.lora).exists():
        raise HTTPException(404, "LoRA file not found.")
    try:
        result = await asyncio.to_thread(
            inference.generate,
            body.prompt,
            lora_path=body.lora,
            lora_scale=body.lora_scale,
            width=body.width,
            height=body.height,
            steps=body.steps,
            guidance_scale=body.guidance_scale,
            seed=body.seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Generation failed: {exc}")
    result["url"] = f"/generated/{result['filename']}"
    return result


@app.get("/api/generated")
async def list_generated() -> list[dict]:
    if not paths.generated.exists():
        return []
    items = [
        {"filename": p.name, "url": f"/generated/{p.name}", "modified": p.stat().st_mtime}
        for p in paths.generated.glob("*.png")
    ]
    items.sort(key=lambda x: x["modified"], reverse=True)
    return items


# ── Static & frontend ─────────────────────────────────────────────────────────

# Ensure dirs exist before mounting (StaticFiles requires them to be present).
paths.ensure()
app.mount("/datasets", StaticFiles(directory=str(paths.datasets)), name="datasets")
app.mount("/generated", StaticFiles(directory=str(paths.generated)), name="generated")
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


def run() -> None:
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
