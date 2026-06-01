#!/usr/bin/env bash
# =============================================================================
# SearchAgent-Zero: Retrieval service environment for V100
# =============================================================================
# Separate conda env for the local dense retriever (e5 + faiss-gpu).
# This runs on 1 GPU and serves HTTP requests to the training workers.
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-retriever-v100}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "=============================================="
echo " SearchAgent-Zero V100 Retrieval Environment"
echo "=============================================="
echo "  Env name:    ${ENV_NAME}"
echo "  Python:      ${PYTHON_VERSION}"
echo "=============================================="

# -----------------------------------------------
# 1. Create conda environment
# -----------------------------------------------
echo ""
echo "[1/4] Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# -----------------------------------------------
# 2. Install PyTorch (conda, for faiss-gpu compat)
# -----------------------------------------------
echo ""
echo "[2/4] Installing PyTorch 2.4.0 via conda (for faiss-gpu compatibility)"
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# -----------------------------------------------
# 3. Install faiss-gpu + retrieval deps
# -----------------------------------------------
echo ""
echo "[3/4] Installing faiss-gpu and retrieval dependencies"
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
conda install -y -c conda-forge 'libstdcxx-ng>=12'

pip install --no-cache-dir \
    transformers \
    datasets \
    pyserini \
    uvicorn \
    fastapi \
    huggingface_hub \
    numpy

# -----------------------------------------------
# 4. Verify
# -----------------------------------------------
echo ""
echo "[4/4] Verification"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -c "
import torch
import faiss
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.cuda.is_available()}')
print(f'  faiss-gpu:     {faiss.get_num_gpus()} GPU(s) available')
"

echo ""
echo "=============================================="
echo " Done! Activate with: conda activate ${ENV_NAME}"
echo "=============================================="
