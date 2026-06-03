#!/usr/bin/env bash
# Search-R1 V100 smoke run through SGLang instead of vLLM.
#
# This avoids the vLLM V0/V1 compatibility path. It still uses the Search-R1
# async multi-turn agent loop, but delegates rollout serving and weight sync to
# the SGLang adapter.

set -euo pipefail
set -x

ray stop --force || true

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples/search_agent_rl/config}"
TOOL_CONFIG="${TOOL_CONFIG:-${CONFIG_PATH}/tool_config/search_tool_config.yaml}"

CACHE_BASE="${CACHE_BASE:-/mnt/models/temp_cache}"
mkdir -p "$CACHE_BASE/huggingface" "$CACHE_BASE/hf_datasets" "$CACHE_BASE/tmp" "$CACHE_BASE/ray_tmp"
mkdir -p ./logs ./output ./rollout_data

python - <<'PY'
import importlib.util

missing = [pkg for pkg in ("sglang", "sgl_kernel") if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(
        "Missing required SGLang package(s): "
        + ", ".join(missing)
        + ". Install SGLang for this env before running the smoke script."
    )
PY

export HF_HOME="$CACHE_BASE/huggingface"
export HF_DATASETS_CACHE="$CACHE_BASE/hf_datasets"
export TMPDIR="$CACHE_BASE/tmp"
export RAY_TMPDIR="$CACHE_BASE/ray_tmp"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export VERL_DISABLE_RAY_RUNTIME_ENV=1
export VERL_FORCE_RAY_LOCALHOST=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export RAY_DISABLE_DASHBOARD="${RAY_DISABLE_DASHBOARD:-1}"

# SGLang defaults to FA3 in this repo's server wrapper; V100 needs a non-FA3
# backend. Keep it overridable because the valid backend list depends on the
# installed SGLang build.
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"

SEARCH_R1_DATA_DIR="${SEARCH_R1_DATA_DIR:-/mnt/models/search_r1_processed}"
TRAIN_DATA="${TRAIN_DATA:-${SEARCH_R1_DATA_DIR}/train_search_r1.parquet}"
VAL_DATA="${VAL_DATA:-${SEARCH_R1_DATA_DIR}/test_search_r1.parquet}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/Qwen2.5-1.5B-Instruct}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2.5-1.5b-instruct_searchr1_v100_sglang_smoke}"
PROJECT_NAME="${PROJECT_NAME:-search_r1_v100}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-./output/$EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-./rollout_data/$EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-./logs/$EXPERIMENT_NAME.log}"

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="search_multiturn_grpo" \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    critic.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1500 \
    actor_rollout_ref.rollout.multi_turn.enable_tool_response_summary=False \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
    actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="$SGLANG_ATTENTION_BACKEND" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    +ray_kwargs.ray_init.include_dashboard=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=250 \
    trainer.test_freq=50 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    "$@" 2>&1 | tee "$LOG_FILE"
