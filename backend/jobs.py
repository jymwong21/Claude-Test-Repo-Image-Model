"""Background job management for long-running pipeline tasks.

A single RTX 5090 means one GPU-heavy job at a time, so this keeps a simple
singleton model: one active training job and one active captioning job, each run
in a daemon thread with its log/progress polled by the web UI.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pipeline import caption as caption_mod
from pipeline import train as train_mod
from pipeline.config import paths, settings


class Status(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Training ──────────────────────────────────────────────────────────────────

# ai-toolkit logs progress lines; pull "<step>/<total>" out of them.
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


@dataclass
class TrainingJob:
    params: train_mod.TrainParams
    status: Status = Status.IDLE
    step: int = 0
    total_steps: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    error: str = ""
    log_path: str = ""
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _lines: list[str] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict:
        elapsed = (self.ended_at or time.time()) - self.started_at if self.started_at else 0
        pct = (self.step / self.total_steps * 100) if self.total_steps else 0
        return {
            "project": self.params.project,
            "status": self.status.value,
            "step": self.step,
            "total_steps": self.total_steps,
            "percent": round(pct, 1),
            "elapsed_sec": round(elapsed),
            "error": self.error,
        }

    def tail(self, n: int = 80) -> list[str]:
        with self._lock:
            return self._lines[-n:]


class TrainingManager:
    def __init__(self) -> None:
        self._job: Optional[TrainingJob] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def job(self) -> Optional[TrainingJob]:
        return self._job

    def is_busy(self) -> bool:
        return self._job is not None and self._job.status == Status.RUNNING

    def start(self, params: train_mod.TrainParams) -> TrainingJob:
        if self.is_busy():
            raise RuntimeError("A training job is already running.")

        config_path = train_mod.build_training_config(params)
        # build_command() raises if ai-toolkit isn't installed (e.g. in Codespaces),
        # so only resolve it for the real path.
        cmd = None if settings.mock else train_mod.build_command(config_path)

        paths.ensure()
        log_path = paths.jobs / f"{params.project}-{int(time.time())}.log"
        job = TrainingJob(
            params=params,
            status=Status.RUNNING,
            total_steps=params.steps,
            started_at=time.time(),
            log_path=str(log_path),
        )
        self._job = job

        def _emit(logf, line: str) -> None:
            logf.write(line + "\n")
            logf.flush()
            with job._lock:
                job._lines.append(line)
                if len(job._lines) > 2000:
                    del job._lines[:1000]

        def _run_mock() -> None:
            """Simulate a training run end-to-end (no GPU, no ai-toolkit)."""
            try:
                with open(log_path, "w", encoding="utf-8") as logf:
                    _emit(logf, f"[mock] starting training for '{params.project}'")
                    _emit(logf, f"[mock] trigger='{params.trigger_word}' rank={params.rank} lr={params.learning_rate}")
                    _emit(logf, "[mock] loading FLUX.1-dev (simulated)…")
                    _emit(logf, "[mock] caching latents to disk…")
                    ticks = 24
                    for t in range(1, ticks + 1):
                        if job.status == Status.CANCELLED:
                            _emit(logf, "[mock] cancelled by user")
                            return
                        time.sleep(0.8)
                        job.step = round(job.total_steps * t / ticks)
                        loss = max(0.04, 0.6 - 0.5 * t / ticks)
                        _emit(logf, f"[mock] step {job.step}/{job.total_steps} loss={loss:.3f}")
                        if t % 6 == 0:
                            _emit(logf, f"[mock] saved checkpoint at step {job.step}")
                    # Write a placeholder LoRA so it appears in the gallery/selector.
                    out_dir = paths.project_output(params.project)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    placeholder = out_dir / f"{params.project}.safetensors"
                    placeholder.write_text(
                        "MOCK LoRA placeholder — train on a real GPU to produce weights.\n",
                        encoding="utf-8",
                    )
                    _emit(logf, f"[mock] wrote {placeholder.name}")
                    _emit(logf, "[mock] training complete ✓")
                job.status = Status.COMPLETED
                job.step = job.total_steps
            except Exception as exc:  # noqa: BLE001
                job.status = Status.FAILED
                job.error = str(exc)
            finally:
                job.ended_at = time.time()

        def _run_real() -> None:
            try:
                with open(log_path, "w", encoding="utf-8") as logf:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(settings.ai_toolkit_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    job._proc = proc
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        logf.write(line)
                        logf.flush()
                        with job._lock:
                            job._lines.append(line.rstrip("\n"))
                            if len(job._lines) > 2000:
                                del job._lines[:1000]
                        m = _STEP_RE.search(line)
                        if m and int(m.group(2)) == job.total_steps:
                            job.step = int(m.group(1))
                    code = proc.wait()
                if job.status == Status.CANCELLED:
                    pass
                elif code == 0:
                    job.status = Status.COMPLETED
                    job.step = job.total_steps
                else:
                    job.status = Status.FAILED
                    job.error = f"ai-toolkit exited with code {code}"
            except Exception as exc:  # noqa: BLE001
                job.status = Status.FAILED
                job.error = str(exc)
            finally:
                job.ended_at = time.time()

        self._thread = threading.Thread(
            target=_run_mock if settings.mock else _run_real, daemon=True)
        self._thread.start()
        return job

    def cancel(self) -> bool:
        if not (self._job and self._job.status == Status.RUNNING):
            return False
        self._job.status = Status.CANCELLED  # mock loop polls this flag
        if self._job._proc:
            self._job._proc.terminate()
        return True


# ── Captioning ────────────────────────────────────────────────────────────────

@dataclass
class CaptionJob:
    project: str
    status: Status = Status.IDLE
    done: int = 0
    total: int = 0
    current: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        pct = (self.done / self.total * 100) if self.total else 0
        return {
            "project": self.project,
            "status": self.status.value,
            "done": self.done,
            "total": self.total,
            "percent": round(pct, 1),
            "current": self.current,
            "error": self.error,
        }


class CaptionManager:
    def __init__(self) -> None:
        self._job: Optional[CaptionJob] = None

    @property
    def job(self) -> Optional[CaptionJob]:
        return self._job

    def is_busy(self) -> bool:
        return self._job is not None and self._job.status == Status.RUNNING

    def start(self, project: str, trigger_word: str, overwrite: bool) -> CaptionJob:
        if self.is_busy():
            raise RuntimeError("A captioning job is already running.")
        job = CaptionJob(project=project, status=Status.RUNNING)
        self._job = job

        def _progress(done: int, total: int, name: str) -> None:
            job.done, job.total, job.current = done, total, name

        def _run() -> None:
            try:
                caption_mod.caption_project(
                    project, trigger_word=trigger_word,
                    overwrite=overwrite, progress=_progress,
                )
                job.status = Status.COMPLETED
            except Exception as exc:  # noqa: BLE001
                job.status = Status.FAILED
                job.error = str(exc)

        threading.Thread(target=_run, daemon=True).start()
        return job


training_manager = TrainingManager()
caption_manager = CaptionManager()
