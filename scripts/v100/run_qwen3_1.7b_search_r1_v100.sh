#!/usr/bin/env bash
# =============================================================================
# SearchAgent-Zero: Qwen3-1.7B Search-R1 Training on 8×V100-32GB
# =============================================================================
# Memory budget: ~10-12 GB/GPU
# Key V100 adaptations:
#   - vLLM v0 engine (VLLM_USE_V1=0)
#   - XFORMERS attention backend (no FlashAttention-2)
#   - FP16 dtype everywhere (no BF16 on Volta)
#   - SDPA attention in HuggingFace model
#   - Conservative gpu_memory_utilization
# =============================================================================
set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples/search_agent_rl/config}"
TOOL_CONFIG="${TOOL_CONFIG:-${CONFIG_PATH}/tool_config/search_tool_config.yaml}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-${CONFIG_PATH}/agent_loop/tool_agent_credit_assignment.yaml}"

CACHE_BASE="${CACHE_BASE:-/tmp/temp_cache}"

mkdir -p "$CACHE_BASE/huggingface"
mkdir -p "$CACHE_BASE/hf_datasets"
mkdir -p "$CACHE_BASE/tmp"
mkdir -p "$CACHE_BASE/ray_tmp"
mkdir -p ./logs
mkdir -p ./output
mkdir -p ./rollout_data

# -----------------------------------------------
# Environment variables
# -----------------------------------------------
export HF_HOME="$CACHE_BASE/huggingface"
export HF_DATASETS_CACHE="$CACHE_BASE/hf_datasets"
export TMPDIR="$CACHE_BASE/tmp"
export RAY_TMPDIR="$CACHE_BASE/ray_tmp"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export VERL_DISABLE_RAY_RUNTIME_ENV=1
export VERL_FORCE_RAY_LOCALHOST=1
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- V100-specific: disable V1 engine, use xformers ---
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

# Proxy settings (disable if not needed)
export no_proxy="*"
export NO_PROXY="*"
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""

# -----------------------------------------------
# Thinking mode (Qwen3 feature)
# Set ENABLE_THINKING=true to enable thinking tokens
# -----------------------------------------------
ENABLE_THINKING="${ENABLE_THINKING:-false}"
MAX_THINKING_TOKENS="${MAX_THINKING_TOKENS:-256}"

# Adjust max_response_length when thinking is enabled
BASE_MAX_RESPONSE=3000
if [ "$ENABLE_THINKING" = "true" ]; then
    MAX_RESPONSE_LENGTH=$((BASE_MAX_RESPONSE + MAX_THINKING_TOKENS))
else
    MAX_RESPONSE_LENGTH=$BASE_MAX_RESPONSE
fi

ulimit -n 65535

# -----------------------------------------------
# Data paths
# -----------------------------------------------
SEARCH_R1_DATA_DIR="${SEARCH_R1_DATA_DIR:-${REPO_ROOT}/examples/search_agent_rl/search_r1_processed}"
TRAIN_DATA="${TRAIN_DATA:-${SEARCH_R1_DATA_DIR}/train_search_r1.parquet}"
VAL_DATA="${VAL_DATA:-${SEARCH_R1_DATA_DIR}/test_search_r1.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"

# -----------------------------------------------
# Training hyperparameters (V100-tuned)
# -----------------------------------------------
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

# Importance Sampling (IS) weights
rollout_is="token"
rollout_is_threshold=2.0
rollout_is_batch_normalize="false"

# Rejection Sampling (RS) configuration
rollout_rs="token_k1,seq_max_k2"
rollout_rs_threshold="0.6_1.6,2.5"

# Bypass PPO mode
bypass_mode="False"
loss_type="ppo_clip"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-1.7b_searchr1_v100}"
PROJECT_NAME="${PROJECT_NAME:-search_r1_v100}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-./output/$EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-./rollout_data/$EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-./logs/$EXPERIMENT_NAME.log}"

# -----------------------------------------------
# Launch training
# -----------------------------------------------
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="search_multiturn_grpo" \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode=async \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
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
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    "$@" 2>&1 | tee "$LOG_FILE"
