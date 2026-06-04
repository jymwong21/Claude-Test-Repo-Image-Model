# Renting & launching a RunPod RTX 5090

A step-by-step for getting Flux LoRA Studio running on a rented 5090.

## 0. Add credits (paying for compute)

RunPod runs on **prepaid credits** — you load a balance, then pods bill against it
per second while running.

1. Create an account and sign in at **runpod.io → Console**.
2. Go to **Billing** (left sidebar) → **Add Credits**.
3. Pay by **credit/debit card** (via Stripe) or **crypto**. The usual **minimum is
   $10**. You can optionally enable **auto-reload** so a pod never dies mid-run.
4. (Optional) Set a **spend limit** so you can't be surprised.

**What you'll actually spend:**
- An RTX 5090 runs roughly **$0.7–1.0/hr** (Community vs Secure Cloud).
- A typical personal LoRA (15–30 images, ~2000 steps) trains in **~30–50 min**, so a
  full session — setup + first model download + a training run + some generation — is
  usually **$2–5**. $10 of credit is plenty to start.
- Billing is **per-second while the pod is RUNNING**. **Stop the pod when idle.**
- A **network volume** (recommended) costs a few **cents/GB-month** and is billed even
  while the pod is stopped — but it keeps your ~24 GB model + LoRAs so you don't
  re-download them next time.

> **Community Cloud** (cheaper, hosted by individuals) vs **Secure Cloud** (datacenter,
> more reliable). Check **RTX 5090** availability under each and pick whichever has stock.

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

## 4. Tips

- First training run downloads FLUX.1-dev (~24 GB) once; it's cached on the
  `/workspace` volume after that.
- **Stop the pod** when you're done to avoid idle charges. With a network volume,
  your models and LoRAs are still there when you restart.
- The trained LoRA `.safetensors` files live in `data/output/<name>/` — download
  them from the UI or via `scp` to keep them after you delete the pod.
- Want to try the app's flow *before* paying for a GPU? Run it in **mock mode** first
  (see the main `README.md` → "Try it now") — same UI, no GPU, no downloads.
