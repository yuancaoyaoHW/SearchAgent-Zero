#!/usr/bin/env bash
# =============================================================================
# SearchAgent-Zero: Training environment for 8×V100-32GB (CUDA 12.x)
# =============================================================================
# This script creates a conda environment with all dependencies adapted for
# NVIDIA V100 (sm_70). Key differences from the default install:
#   - PyTorch 2.4 + CUDA 12.1 (last stable combo for V100 + vLLM 0.6.x)
#   - vLLM 0.6.6.post1 (last version with solid sm_70 support)
#   - xformers (replaces FlashAttention-2 / FlashInfer which need sm_80+)
#   - No flash-attn, no flashinfer, no sglang
#   - Python 3.11 (vLLM 0.6.x has better 3.11 support than 3.12)
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-verl-v100}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=============================================="
echo " SearchAgent-Zero V100 Training Environment"
echo "=============================================="
echo "  Env name:    ${ENV_NAME}"
echo "  Python:      ${PYTHON_VERSION}"
echo "  Repo root:   ${REPO_ROOT}"
echo "=============================================="

# -----------------------------------------------
# 1. Create conda environment
# -----------------------------------------------
echo ""
echo "[1/6] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# -----------------------------------------------
# 2. Install PyTorch 2.4 + CUDA 12.1
# -----------------------------------------------
echo ""
echo "[2/6] Installing PyTorch 2.4.0 + CUDA 12.4"
pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------------
# 3. Install vLLM 0.6.6.post1 (V100-compatible)
# -----------------------------------------------
echo ""
echo "[3/6] Installing vLLM 0.6.6.post1 (last sm_70 stable release)"
pip install --no-cache-dir "vllm==0.6.6.post1"

# -----------------------------------------------
# 4. Install xformers (attention backend for V100)
# -----------------------------------------------
echo ""
echo "[4/6] Installing xformers 0.0.28.post3 (V100 attention backend)"
pip install --no-cache-dir "xformers==0.0.28.post3"

# -----------------------------------------------
# 5. Install core dependencies
# -----------------------------------------------
echo ""
echo "[5/6] Installing core Python packages"
pip install --no-cache-dir \
    "transformers>=4.45.0,<4.52.0" \
    accelerate \
    datasets \
    peft \
    hf-transfer \
    "numpy<2.0.0" \
    "pyarrow>=15.0.0" \
    pandas \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    torchdata \
    "ray[default]>=2.41.0" \
    codetiming \
    hydra-core \
    pylatexenc \
    qwen-vl-utils \
    wandb \
    dill \
    pybind11 \
    liger-kernel \
    mathruler \
    pytest \
    py-spy \
    pre-commit \
    ruff \
    tensorboard \
    "nvidia-ml-py>=12.560.30" \
    "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" \
    "pydantic>=2.9" \
    "grpcio>=1.62.1" \
    packaging \
    uvicorn \
    latex2sympy2_extended \
    math-verify

# -----------------------------------------------
# 6. Install SearchAgent-Zero (no-deps, editable)
# -----------------------------------------------
echo ""
echo "[6/6] Installing SearchAgent-Zero in editable mode"
cd "${REPO_ROOT}"
pip install --no-deps -e .
if git rev-parse --git-dir &>/dev/null; then
    pre-commit install || true
fi

# -----------------------------------------------
# Verification
# -----------------------------------------------
echo ""
echo "=============================================="
echo " Verification"
echo "=============================================="
python -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count:     {torch.cuda.device_count()}')
    print(f'  GPU 0:         {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'  Compute cap:   {cap[0]}.{cap[1]}')
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  Memory:        {mem_gb:.1f} GB')
"
python -c "import vllm; print(f'  vLLM:          {vllm.__version__}')"
python -c "import xformers; print(f'  xformers:      {xformers.__version__}')"
python -c "import ray; print(f'  Ray:           {ray.__version__}')"
python -c "import verl; print(f'  verl:          installed')"

echo ""
echo "=============================================="
echo " Done! Activate with: conda activate ${ENV_NAME}"
echo "=============================================="
