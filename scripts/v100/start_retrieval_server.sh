#!/usr/bin/env bash
# =============================================================================
# SearchAgent-Zero: Start local retrieval server on V100
# =============================================================================
# Uses 1 GPU for faiss-gpu search. Run in the retriever-v100 conda env.
# The training scripts expect this server at http://127.0.0.1:8000/retrieve
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Use GPU 0 for retrieval (training uses GPUs 0-7, retrieval is lightweight)
RETRIEVAL_GPU="${RETRIEVAL_GPU:-0}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/examples/search_agent_rl/local_dense_retriever/search_data}"
INDEX_FILE="${INDEX_FILE:-${DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${DATA_DIR}/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
PORT="${PORT:-8000}"
PYTHON="${PYTHON:-python3}"

echo "=============================================="
echo " SearchAgent-Zero: Local Retrieval Server"
echo "=============================================="
echo "  GPU:           ${RETRIEVAL_GPU}"
echo "  Index:         ${INDEX_FILE}"
echo "  Corpus:        ${CORPUS_FILE}"
echo "  Model:         ${RETRIEVER_MODEL}"
echo "  Port:          ${PORT}"
echo "=============================================="

# Validate data files exist
if [[ ! -f "${INDEX_FILE}" ]]; then
    echo "ERROR: Index file not found: ${INDEX_FILE}" >&2
    echo "Run download_data.sh first." >&2
    exit 1
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
    echo "ERROR: Corpus file not found: ${CORPUS_FILE}" >&2
    echo "Run download_data.sh first." >&2
    exit 1
fi

echo ""
echo "Starting retrieval server on port ${PORT}..."
echo "  (Use Ctrl+C to stop, or run in tmux/screen for background)"
echo ""

CUDA_VISIBLE_DEVICES="${RETRIEVAL_GPU}" "${PYTHON}" \
    "${REPO_ROOT}/examples/search_agent_rl/local_dense_retriever/retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk 3 \
    --retriever_name "${RETRIEVER_NAME}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    --faiss_gpu
