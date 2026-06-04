# Renting & launching a RunPod RTX 5090

A step-by-step for getting Flux LoRA Studio running on a rented 5090.

## 1. Create the pod

1. Go to **runpod.io → Pods → Deploy**.
2. **GPU:** pick **RTX 5090** (32 GB VRAM). A single 5090 trains a Flux LoRA comfortably.
3. **Template:** choose a **CUDA 12.8+** PyTorch image. Good choices:
   - `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04`
   - or any "PyTorch 2.7/2.8 / CUDA 12.8" community template.

   > ⚠️ **Do not** use a CUDA 12.4 / torch 2.3–2.6 template. Blackwell (sm_120)
   > needs cu128. `runpod/setup.sh` installs the right torch regardless, but
   > starting from a 12.8 base avoids a long reinstall.
4. **Disk:** Container disk ≥ 50 GB. Add a **Network Volume** (≥ 60 GB) mounted at
   `/workspace` so the ~24 GB FLUX.1-dev download and your datasets/LoRAs persist
   across pod restarts.
5. **Ports:** expose HTTP port **8000** (for the web UI) and TCP **22** (SSH).
6. Deploy.

## 2. Connect & install

```bash
# SSH into the pod (or use the RunPod web terminal), then:
cd /workspace
git clone <your-fork-url> flux-lora-studio
cd flux-lora-studio

cp .env.example .env
nano .env          # paste your HF_TOKEN, accept FLUX.1-dev license on HF first

bash runpod/setup.sh
```

`setup.sh` installs Blackwell-correct PyTorch, clones `ai-toolkit`, installs the
app deps, logs into Hugging Face, and creates the data directories. It takes
~10–15 min on a fresh pod (most of it downloading wheels).

## 3. Launch the UI

```bash
source .venv/bin/activate
python -m backend.main
```

Then in the RunPod dashboard open **Connect → HTTP Service [Port 8000]**. You'll
land on the Flux LoRA Studio UI. From there: upload photos → caption → train →
generate.

## 4. Costs & tips

- A 5090 on RunPod runs roughly **$0.7–1.0/hr** (community vs secure cloud).
- First training run downloads FLUX.1-dev (~24 GB) once; it's cached on the
  `/workspace` volume after that.
- A typical personal LoRA (15–25 images, ~2000 steps) trains in **~30–50 min**.
- **Stop the pod** when you're done to avoid idle charges. With a network volume,
  your models and LoRAs are still there when you restart.
- The trained LoRA `.safetensors` files live in `data/output/<name>/` — download
  them from the UI or via `scp` to keep them after you delete the pod.
