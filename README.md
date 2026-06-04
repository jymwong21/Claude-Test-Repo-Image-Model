# Flux LoRA Studio

Train a personal **Flux.1-dev LoRA** on your own photos and generate images with
it — end to end, self-hosted on a rented **RunPod RTX 5090**, driven by a clean
web UI.

Upload photos → auto-caption → train → generate. No third-party image APIs; the
models run on your GPU and the trained `.safetensors` are yours.

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 1 Dataset│ → │ 2 Caption│ → │ 3 Train  │ → │4 Generate│
│  upload  │   │Florence-2│   │ai-toolkit│   │diffusers │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## Try it now (mock mode — no GPU, no downloads)

Want to click through the entire flow before paying for a GPU or downloading 24 GB
of model? Run in **mock mode**: captioning, training, and generation are stubbed
with placeholders, so the whole UI works on a plain laptop or in **GitHub
Codespaces** with ~30 MB of dependencies.

**GitHub Codespaces (easiest):** open this repo → **Code ▸ Codespaces ▸ Create
codespace**. The dev container installs the light deps and sets `MOCK=true`
automatically. Then in the terminal:

```bash
python -m backend.main          # opens the forwarded port 8000
```

**Local (laptop, no Docker):**

```bash
pip install -r requirements-dev.txt
MOCK=true python -m backend.main     # http://localhost:8000
```

You'll see a **MOCK MODE** badge in the top bar. Create a project, drop in a few
photos, auto-caption, "train" (simulated progress), and generate placeholder
images — exercising every screen. The same `.devcontainer/` opens locally via
VS Code **Dev Containers: Reopen in Container**, so moving off Codespaces is a
one-click affair. When you're on a real GPU box, set `MOCK=false` and install
`requirements.txt` — same UI, real models. See [`runpod/README.md`](runpod/README.md)
for renting a 5090 (incl. billing) or running on your own PC.

## What's in here

| Path | Purpose |
|------|---------|
| `runpod/setup.sh` | One-shot pod setup — Blackwell-correct PyTorch (cu128), ai-toolkit, deps |
| `runpod/README.md` | How to rent & launch a 5090 on RunPod |
| `pipeline/prepare_dataset.py` | Clean/resize uploads into a training folder |
| `pipeline/caption.py` | Auto-caption images with Florence-2 (+ trigger word) |
| `pipeline/train.py` | Build an ai-toolkit config and launch a LoRA run |
| `pipeline/inference.py` | Generate images from a trained LoRA via `diffusers` |
| `pipeline/configs/flux_lora_5090.yaml` | ai-toolkit base config, tuned for a 5090 |
| `backend/main.py` | FastAPI app — serves the UI + JSON API |
| `backend/jobs.py` | Background training/captioning job runners with live progress |
| `frontend/` | The 4-step web UI |
| `scripts/download_models.py` | Pre-fetch FLUX.1-dev + Florence-2 |

## Why a 5090 (and the one big gotcha)

The RTX 5090 is **Blackwell, sm_120, 32 GB** — plenty for a Flux.1-dev LoRA. But
Blackwell is **only** supported by **CUDA 12.8+ / PyTorch ≥ 2.7 (cu128)**. Older
wheels install fine then crash with *"no kernel image is available for execution
on the device"*. `runpod/setup.sh` installs the correct stack for you — just
start from a CUDA 12.8 base template.

## Quick start

```bash
# On a RunPod 5090 pod (see runpod/README.md for renting it):
cd /workspace
git clone <your-fork-url> flux-lora-studio && cd flux-lora-studio

cp .env.example .env        # add your HF_TOKEN (accept the FLUX.1-dev license first)
bash runpod/setup.sh        # installs everything (~10–15 min)

source .venv/bin/activate
python -m backend.main      # serves the UI on :8000
```

Open the pod's **Connect → HTTP Service [8000]** link and follow the 4 steps.

## The workflow in detail

**1 · Dataset** — Drop 15–30 photos of your subject. Variety wins: different
angles, lighting, expressions, backgrounds. Images are auto-converted to RGB,
EXIF-rotated, and downscaled to a 1536px long edge.

**2 · Caption** — Choose a **trigger word** (a rare token like `ohwx person`).
Florence-2 writes a description per image and the trigger is prepended, giving
captions like `ohwx person, a man standing in a park wearing a blue jacket`.
Review and edit any caption inline before training.

**3 · Train** — Tune steps / learning rate / LoRA rank, then launch. The run uses
`ostris/ai-toolkit` with Flux.1-dev quantized to 8-bit + gradient checkpointing,
which fits comfortably in 32 GB. Watch live progress and logs; sample images are
written every N steps. ~15–30 images at 2000 steps ≈ 30–50 min.

**4 · Generate** — Pick your trained LoRA, write a prompt **using the trigger
word**, set strength/steps/guidance/aspect, and generate. Results land in a
gallery with a download button.

## Using it from the command line

Every stage is scriptable without the UI:

```bash
python -m pipeline.prepare_dataset ./my_photos myproject
python -m pipeline.caption myproject --trigger "ohwx person"
python -m pipeline.train myproject --trigger "ohwx person" --steps 2000 --rank 16
python -m pipeline.inference "ohwx person on a snowy mountain, golden hour" \
    --lora data/output/myproject/myproject.safetensors
```

## Configuration

All config is via `.env` (see `.env.example`):

- `HF_TOKEN` — **required**; FLUX.1-dev is gated.
- `BASE_MODEL` — defaults to `black-forest-labs/FLUX.1-dev`.
- `DATA_DIR` — where datasets/LoRAs/outputs live; point at `/workspace/...` on
  RunPod so it persists across restarts.
- `HOST` / `PORT` — web server bind (default `0.0.0.0:8000`).

## Notes & tips

- **Keep your LoRAs.** They're in `data/output/<project>/`. Download them before
  deleting the pod (the UI lists them; or `scp` them off).
- **Trigger word in prompts.** If your subject doesn't show up, make sure the
  prompt contains the exact trigger word you trained with.
- **Overfitting.** Too many steps → stiff, identical outputs. Start at 2000 and
  lower if results look "burned in."
- **One GPU, one job.** Training and generation share the GPU; the backend
  blocks generation while a training run is active.

## License & responsible use

Respect the **FLUX.1-dev non-commercial license** and only train on images you
have the right to use. Don't create deceptive or non-consensual imagery of real
people.
