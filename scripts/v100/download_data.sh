#!/usr/bin/env bash
# =============================================================================
# SearchAgent-Zero: Download retrieval index + preprocess training datasets
# =============================================================================
# Run this AFTER install_train_env.sh. Uses the training env for preprocessing.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

RETRIEVER_DATA_DIR="${RETRIEVER_DATA_DIR:-${REPO_ROOT}/examples/search_agent_rl/local_dense_retriever/search_data}"
SEARCH_R1_OUTPUT="${SEARCH_R1_OUTPUT:-${REPO_ROOT}/examples/search_agent_rl/search_r1_processed}"
ASEARCH_OUTPUT="${ASEARCH_OUTPUT:-${REPO_ROOT}/examples/search_agent_rl/ASearcher}"

echo "=============================================="
echo " SearchAgent-Zero: Data Download & Preprocess"
echo "=============================================="
echo "  Retriever data: ${RETRIEVER_DATA_DIR}"
echo "  Search-R1 data: ${SEARCH_R1_OUTPUT}"
echo "  ASearch data:   ${ASEARCH_OUTPUT}"
echo "=============================================="

# -----------------------------------------------
# 1. Download retrieval index and corpus
# -----------------------------------------------
echo ""
echo "[1/4] Downloading wiki-18 dense index and corpus..."
mkdir -p "${RETRIEVER_DATA_DIR}"

"${PYTHON}" "${REPO_ROOT}/examples/search_agent_rl/local_dense_retriever/download.py" \
    --save_path "${RETRIEVER_DATA_DIR}"

echo "  Merging index shards..."
cat "${RETRIEVER_DATA_DIR}/part_aa" "${RETRIEVER_DATA_DIR}/part_ab" > "${RETRIEVER_DATA_DIR}/e5_Flat.index"

echo "  Decompressing corpus..."
if [[ -f "${RETRIEVER_DATA_DIR}/wiki-18.jsonl.gz" && ! -f "${RETRIEVER_DATA_DIR}/wiki-18.jsonl" ]]; then
    gzip -dk "${RETRIEVER_DATA_DIR}/wiki-18.jsonl.gz"
fi

echo "  ✓ Retrieval data ready"

# -----------------------------------------------
# 2. Preprocess Search-R1 dataset
# -----------------------------------------------
echo ""
echo "[2/4] Preprocessing Search-R1 dataset..."
mkdir -p "${SEARCH_R1_OUTPUT}"

"${PYTHON}" "${REPO_ROOT}/examples/search_agent_rl/preprocess_search_r1_dataset_new.py" \
    --local_dir "${SEARCH_R1_OUTPUT}"

echo "  ✓ Search-R1 data ready: ${SEARCH_R1_OUTPUT}/"
ls -lh "${SEARCH_R1_OUTPUT}"/*.parquet 2>/dev/null || true

# -----------------------------------------------
# 3. Preprocess ASearcher dataset
# -----------------------------------------------
echo ""
echo "[3/4] Preprocessing ASearcher dataset..."
mkdir -p "${ASEARCH_OUTPUT}"

ASEARCH_RAW_DIR="${REPO_ROOT}/examples/search_agent_rl/ASearcher_en_no-math_Qwen3-8B-reject-sample"

"${PYTHON}" "${REPO_ROOT}/examples/search_agent_rl/preprocess_ASearcher_dataset.py" \
    --hf_repo_id "aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample" \
    --local_dir "${ASEARCH_RAW_DIR}" \
    --output_path "${ASEARCH_OUTPUT}/ASearcher_train.parquet" \
    --test_output_path "${ASEARCH_OUTPUT}/ASearcher_test.parquet" \
    --train_ratio 0.95 \
    --seed 42

echo "  ✓ ASearcher data ready: ${ASEARCH_OUTPUT}/"
ls -lh "${ASEARCH_OUTPUT}"/*.parquet 2>/dev/null || true

# -----------------------------------------------
# 4. Summary
# -----------------------------------------------
echo ""
echo "=============================================="
echo " All data downloaded and preprocessed!"
echo ""
echo " Retrieval index: ${RETRIEVER_DATA_DIR}/e5_Flat.index"
echo " Retrieval corpus: ${RETRIEVER_DATA_DIR}/wiki-18.jsonl"
echo " Search-R1 train: ${SEARCH_R1_OUTPUT}/train_search_r1.parquet"
echo " Search-R1 test:  ${SEARCH_R1_OUTPUT}/test_search_r1.parquet"
echo " ASearch train:   ${ASEARCH_OUTPUT}/ASearcher_train.parquet"
echo " ASearch test:    ${ASEARCH_OUTPUT}/ASearcher_test.parquet"
echo "=============================================="
