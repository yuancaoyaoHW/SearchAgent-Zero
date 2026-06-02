#!/usr/bin/env bash
# =============================================================================
# SearchAgent-Zero: ASearch Training on 8×V100-32GB (Qwen3-8B)
# =============================================================================
# Adapted from run_qwen3_8b_instruct_search_multiturn_ASearch.sh
# Key V100 adaptations:
#   - vLLM v0 engine (VLLM_USE_V1=0), XFORMERS backend
#   - FP16 dtype (no BF16 on Volta)
#   - SDPA attention in HuggingFace model
#   - Reduced batch/n/turns to fit 32GB VRAM
#   - TP=2 for vLLM inference (8B model needs 2 GPUs for KV cache)
#   - Synchronous mode only (fully-async needs V1 engine)
#
# ⚠️  This config is tight on memory. If OOM occurs:
#   - Reduce rollout.n from 4 to 2
#   - Reduce MAX_RESPONSE_LENGTH from 16384 to 12288
#   - Reduce gpu_memory_utilization from 0.55 to 0.45
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
export VLLM_DISABLE_COMPILE_CACHE=1

# Proxy settings
export no_proxy="*"
export NO_PROXY="*"
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""

# -----------------------------------------------
# Thinking mode (Qwen3 feature) - disabled for 8B ASearch (OOM risk)
# -----------------------------------------------
ENABLE_THINKING="${ENABLE_THINKING:-false}"

ulimit -n 65535

# -----------------------------------------------
# Data paths
# -----------------------------------------------
ASEARCH_DATA_DIR="${ASEARCH_DATA_DIR:-${REPO_ROOT}/examples/search_agent_rl/ASearcher}"
TRAIN_DATA="${TRAIN_DATA:-${ASEARCH_DATA_DIR}/ASearcher_train.parquet}"
VAL_DATA="${VAL_DATA:-${ASEARCH_DATA_DIR}/ASearcher_test.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"

# -----------------------------------------------
# V100-tuned hyperparameters (reduced from H20 defaults)
# -----------------------------------------------
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"       # H20: 36864
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12000}"                   # H20: 20000
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"           # H20: 64
TURN_LIMIT_SCHEDULE="${TURN_LIMIT_SCHEDULE:-0:30,50:40,100:50,200:50,300:50}"  # H20: 100 turns

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-8b_ASearch_v100}"
PROJECT_NAME="${PROJECT_NAME:-search_agent_v100}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-./output/$EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-./rollout_data/$EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-./logs/$EXPERIMENT_NAME.log}"

# Importance Sampling (IS) weights
rollout_is="token"
rollout_is_threshold=2.0
rollout_is_batch_normalize="false"

# Rejection Sampling (RS) configuration
rollout_rs="token_k1"
rollout_rs_threshold="0.6_1.6"

# Bypass PPO mode
bypass_mode="False"
loss_type="ppo_clip"

# -----------------------------------------------
# Launch training
# -----------------------------------------------
python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="search_multiturn_grpo" \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=50 \
    actor_rollout_ref.rollout.multi_turn.turn_limit_schedule="'$TURN_LIMIT_SCHEDULE'" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=15000 \
    actor_rollout_ref.rollout.multi_turn.enable_tool_response_summary=True \
    actor_rollout_ref.rollout.multi_turn.summary_result_separator='\\n-*-*-\\n' \
    actor_rollout_ref.rollout.multi_turn.summary_temperature=0.6 \
    actor_rollout_ref.rollout.multi_turn.summary_top_p=0.95 \
    actor_rollout_ref.rollout.multi_turn.summary_top_k=20 \
    actor_rollout_ref.rollout.multi_turn.summary_max_tokens=1024 \
    actor_rollout_ref.rollout.multi_turn.summary_use_external_model=False \
    actor_rollout_ref.rollout.multi_turn.summary_external_base_urls="" \
    actor_rollout_ref.rollout.multi_turn.summary_external_model="" \
    actor_rollout_ref.rollout.multi_turn.max_queries_per_tool_call=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
    +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
    "$@" 2>&1 | tee "$LOG_FILE"
