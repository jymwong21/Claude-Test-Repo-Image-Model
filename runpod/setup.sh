#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Flux LoRA Studio — RunPod RTX 5090 setup
#
# The RTX 5090 is a Blackwell GPU (compute capability sm_120). It is ONLY
# supported by CUDA 12.8+ and PyTorch built against cu128 (torch >= 2.7).
# Older wheels (cu121/cu124, torch <= 2.6) will install fine but then crash at
# runtime with:  "no kernel image is available for execution on the device".
#
# This script installs the correct stack, clones ai-toolkit, and wires up the
# app. Run it ONCE per fresh pod, from the repo root:
#
#     bash runpod/setup.sh
#
# Recommended RunPod template: any CUDA 12.8+ base image, e.g.
#   "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04"
# Pick a pod with >= 48 GB container disk and a /workspace network volume so the
# ~24 GB FLUX.1-dev download survives restarts.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "──────────────────────────────────────────────────────────"
echo "  Flux LoRA Studio — RunPod 5090 setup"
echo "  Repo: $REPO_ROOT"
echo "──────────────────────────────────────────────────────────"

# 1. ── Sanity: is a Blackwell GPU visible? ────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[1/7] GPU detected:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[1/7] WARNING: nvidia-smi not found — are you on a GPU pod?" >&2
fi

# 2. ── System packages ────────────────────────────────────────────────────────
echo "[2/7] Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git git-lfs python3-venv build-essential >/dev/null
git lfs install --skip-repo >/dev/null 2>&1 || true

# 3. ── Python venv ────────────────────────────────────────────────────────────
echo "[3/7] Creating virtual environment (.venv)..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools >/dev/null

# 4. ── PyTorch for Blackwell (cu128) ──────────────────────────────────────────
# This MUST come before any other install so nothing pulls a wrong torch wheel.
echo "[4/7] Installing PyTorch (cu128 / Blackwell sm_120)..."
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio

echo "      Verifying CUDA can see the GPU..."
python - <<'PY'
import torch
print(f"      torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"      device: {torch.cuda.get_device_name(0)} | sm_{cap[0]}{cap[1]}")
    assert cap[0] >= 12, "Expected Blackwell sm_120+ — wrong torch build?"
else:
    print("      WARNING: CUDA not available — check the pod/driver.")
PY

# 5. ── ai-toolkit (the trainer) ───────────────────────────────────────────────
echo "[5/7] Cloning ostris/ai-toolkit..."
if [ ! -d ai-toolkit ]; then
  git clone https://github.com/ostris/ai-toolkit.git
fi
pushd ai-toolkit >/dev/null
git submodule update --init --recursive
# Install ai-toolkit deps but DO NOT let it override our torch.
grep -ivE '^torch(vision|audio)?([=<>!~ ].*)?$' requirements.txt > /tmp/ait-reqs.txt || cp requirements.txt /tmp/ait-reqs.txt
pip install -r /tmp/ait-reqs.txt
popd >/dev/null

# 6. ── App dependencies ───────────────────────────────────────────────────────
echo "[6/7] Installing app dependencies (backend + inference + captioning)..."
pip install -r requirements.txt

# 7. ── Data dirs + HF login ───────────────────────────────────────────────────
echo "[7/7] Preparing data directories & Hugging Face auth..."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
DATA_DIR="${DATA_DIR:-/workspace/flux-lora-studio/data}"
mkdir -p "$DATA_DIR"/{datasets,captions,output,models,generated,jobs}
echo "      Data dir: $DATA_DIR"

if [ -n "${HF_TOKEN:-}" ]; then
  python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 \
    || huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true
  echo "      Logged into Hugging Face."
else
  echo "      NOTE: HF_TOKEN not set in .env — set it before training (FLUX.1-dev is gated)."
fi

echo "──────────────────────────────────────────────────────────"
echo "  ✅ Setup complete."
echo
echo "  Next:"
echo "    source .venv/bin/activate"
echo "    python -m backend.main          # starts the web UI on :\${PORT:-8000}"
echo
echo "  On RunPod, expose the HTTP port (8000) and open the pod's"
echo "  'Connect → HTTP Service' link to reach the UI."
echo "──────────────────────────────────────────────────────────"
