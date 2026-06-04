#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Flux LoRA Studio — RunPod RTX 5090 setup
#
# Recommended base image (what this script is tuned for):
#   runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   → CUDA 12.8.1 · PyTorch 2.8.0 · Ubuntu 24.04 · Python 3.11
#
# The RTX 5090 is a Blackwell GPU (compute capability sm_120), supported ONLY by
# CUDA 12.8 + PyTorch >= 2.7 built against cu128. Older wheels (cu124, torch <= 2.6)
# install fine then crash at runtime with "no kernel image is available for
# execution on the device".
#
# Heads-up: some RunPod "torch 2.8" images have historically shipped torch
# 2.4.1+cu124 by mistake (see runpod/containers#114). So this script does NOT
# blindly trust the base image — it VERIFIES the installed torch and only
# reinstalls the correct cu128 build when it's wrong. Fast when the image is
# already correct; self-healing when it isn't.
#
# Run ONCE per fresh pod, from the repo root:
#     bash runpod/setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_BREAK_SYSTEM_PACKAGES=1   # Ubuntu 24.04 system Python is PEP-668 managed

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY=python3
PIP="$PY -m pip"

# Pinned, mutually-compatible CUDA 12.8 torch stack (matches the base image label).
CU128_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PINS=(torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0)

echo "──────────────────────────────────────────────────────────"
echo "  Flux LoRA Studio — RunPod 5090 setup"
echo "  Repo: $REPO_ROOT"
echo "  Python: $($PY --version 2>&1)"
echo "──────────────────────────────────────────────────────────"

# 1. ── GPU sanity ─────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[1/7] GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[1/7] WARNING: nvidia-smi not found — are you on a GPU pod?" >&2
fi

# 2. ── System packages ────────────────────────────────────────────────────────
echo "[2/7] Installing system packages (git, git-lfs, build tools)…"
apt-get update -qq
apt-get install -y -qq git git-lfs build-essential >/dev/null
git lfs install --skip-repo >/dev/null 2>&1 || true

# 3. ── pip + torchvision runtime deps ─────────────────────────────────────────
echo "[3/7] Upgrading pip…"
$PIP install --upgrade pip wheel setuptools >/dev/null
# Make sure numpy/pillow exist so a cu128-ONLY torch install can resolve its deps.
$PIP install --quiet "numpy>=1.26" "pillow>=10.4" >/dev/null

# Reusable check: is the importable torch Blackwell-capable (cu128, torch>=2.7)?
torch_adequate() {
  $PY - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("   torch not importable:", e); sys.exit(1)
base = torch.__version__.split('+')[0]
mm = tuple(int(x) for x in base.split('.')[:2])
cuda = torch.version.cuda or ""
print(f"   torch {torch.__version__} (cuda {cuda or 'none'})")
if mm < (2, 7) or not cuda.startswith("12.8"):
    sys.exit(2)
try:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        x = torch.randn(16, 16, device="cuda"); (x @ x).sum().item()
        print(f"   GPU kernel OK: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
    else:
        print("   no GPU visible at setup time — accepting on version/build")
except Exception as e:
    print("   GPU kernel test FAILED:", e); sys.exit(3)
sys.exit(0)
PY
}

# 4. ── PyTorch (verify, fix only if needed) ───────────────────────────────────
echo "[4/7] Verifying PyTorch is Blackwell-capable…"
if torch_adequate; then
  echo "      ✓ existing torch is good — keeping it (no reinstall)."
else
  echo "      ✗ inadequate/missing — installing pinned cu128 build…"
  $PIP install --index-url "$CU128_INDEX" "${TORCH_PINS[@]}"
  torch_adequate || { echo "ERROR: torch still not Blackwell-capable after install." >&2; exit 1; }
fi

# 5. ── ai-toolkit (the trainer) ───────────────────────────────────────────────
echo "[5/7] Cloning ostris/ai-toolkit + installing its pinned stack…"
if [ ! -d ai-toolkit ]; then
  git clone https://github.com/ostris/ai-toolkit.git
fi
pushd ai-toolkit >/dev/null
git submodule update --init --recursive
# Install in place so ai-toolkit's nested `-r requirements_base.txt` resolves.
$PIP install -r requirements.txt
popd >/dev/null
# Guard: if ai-toolkit's deps ever pull a non-cu128 torch, restore ours.
if ! torch_adequate; then
  echo "      ai-toolkit changed torch — restoring cu128 build…"
  $PIP install --index-url "$CU128_INDEX" "${TORCH_PINS[@]}"
fi

# 6. ── App + inference + captioning deps ──────────────────────────────────────
# requirements.txt MIRRORS ai-toolkit's ML pins (transformers 5.5.3, hf_hub 1.10.1,
# peft 0.18.1, diffusers @ the same commit, …), so this is a no-op for the ML libs
# and just adds the web server + Florence-2/T5 extras. No version fighting.
echo "[6/7] Installing app dependencies…"
$PIP install -r requirements.txt

# 7. ── Data dirs + HF login ───────────────────────────────────────────────────
echo "[7/7] Preparing data directories & Hugging Face auth…"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
DATA_DIR="${DATA_DIR:-/workspace/flux-lora-studio/data}"
mkdir -p "$DATA_DIR"/{datasets,captions,output,models,generated,jobs}
echo "      Data dir: $DATA_DIR"
if [ -n "${HF_TOKEN:-}" ]; then
  $PY - <<'PY'
import os
from huggingface_hub import login
try:
    login(token=os.environ["HF_TOKEN"])
    print("      Logged into Hugging Face.")
except Exception as e:
    print("      HF login skipped:", e)
PY
else
  echo "      NOTE: HF_TOKEN not set in .env — set it before training (FLUX.1-dev is gated)."
fi

echo "──────────────────────────────────────────────────────────"
echo "  ✅ Setup complete."
echo
echo "  Start the app:"
echo "    python3 -m backend.main          # web UI on :\${PORT:-8000}"
echo
echo "  On RunPod, expose port 8000 and open 'Connect → HTTP Service'."
echo "──────────────────────────────────────────────────────────"
