# V100 Deployment Scripts

Complete deployment scripts for running SearchAgent-Zero on **8×V100-32GB** servers with CUDA 12.x.

## Overview

```
scripts/v100/
├── install_train_env.sh          # Step 1: Training conda env
├── install_retrieval_env.sh      # Step 2: Retrieval service conda env
├── download_data.sh              # Step 3: Download index + preprocess datasets
├── start_retrieval_server.sh     # Step 4: Launch retrieval HTTP server
├── run_search_r1_v100.sh         # Step 5a: Train Search-R1 (Qwen2.5-3B)
└── run_asearch_v100.sh           # Step 5b: Train ASearch (Qwen3-8B)
```

## Quick Start

```bash
cd /path/to/SearchAgent-Zero

# 1. Install training environment
bash scripts/v100/install_train_env.sh

# 2. Install retrieval environment (separate conda env)
bash scripts/v100/install_retrieval_env.sh

# 3. Download data (use training env)
conda activate verl-v100
bash scripts/v100/download_data.sh

# 4. Start retrieval server (in a tmux/screen session)
conda activate retriever-v100
bash scripts/v100/start_retrieval_server.sh
# Wait until "Uvicorn running on http://0.0.0.0:8000" appears

# 5. Train (in another terminal)
conda activate verl-v100
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=your_key_here

# Option A: Search-R1 (Qwen2.5-3B, recommended starting point)
bash scripts/v100/run_search_r1_v100.sh

# Option B: ASearch (Qwen3-8B, longer horizon, slower)
bash scripts/v100/run_asearch_v100.sh
```

## V100 vs H20/A100 Key Differences

| Setting | H20/A100 Default | V100 Adapted | Reason |
|---------|-----------------|--------------|--------|
| dtype | bfloat16 | float16 | V100 has no BF16 tensor cores |
| Attention | FlashAttention-2 | SDPA + xformers | FA2 requires sm_80+ |
| vLLM version | 0.11.0 | 0.6.6.post1 | Last sm_70 stable release |
| vLLM engine | V1 (`VLLM_USE_V1=1`) | V0 (`VLLM_USE_V1=0`) | V1 needs FlashInfer |
| FlashInfer | 0.3.1 | Not installed | Requires sm_80+ |
| SGLang | 0.5.2 | Not installed | Depends on FlashInfer |

## Memory Budget (V100-32GB)

### Search-R1 (Qwen2.5-3B) — Comfortable
- Model params (FP16): ~6 GB total → ~0.75 GB/GPU (FSDP sharded)
- Optimizer (Adam, offloaded to CPU): minimal GPU cost
- Activations (gradient checkpointing): ~3-5 GB/GPU
- vLLM KV cache (n=5, 15K tokens): ~8 GB/GPU
- **Total: ~15-18 GB/GPU** ✓

### ASearch (Qwen3-8B) — Tight
- Model params (FP16): ~16 GB total → ~2 GB/GPU (FSDP sharded)
- Optimizer (offloaded): minimal GPU cost
- Activations (gc + offload): ~6-10 GB/GPU
- vLLM KV cache (n=4, TP=2, 12K tokens): ~10-14 GB/GPU
- **Total: ~25-30 GB/GPU** ⚠️ (fits but tight)

## Troubleshooting

### OOM on ASearch (8B)
Reduce these in order:
```bash
# In run_asearch_v100.sh or via env vars:
MAX_RESPONSE_LENGTH=12288        # default: 16384
MAX_MODEL_LEN=10000              # default: 12000
# Or pass overrides:
bash scripts/v100/run_asearch_v100.sh actor_rollout_ref.rollout.n=2
```

### vLLM crashes on startup
```bash
# Verify V100 is detected
python -c "import torch; print(torch.cuda.get_device_capability(0))"
# Should print (7, 0)

# Ensure xformers backend
export VLLM_ATTENTION_BACKEND=XFORMERS
python -c "import vllm; print(vllm.__version__)"
# Should print 0.6.6.post1
```

### Retrieval server connection refused
The training scripts expect the retrieval server at `http://127.0.0.1:8000/retrieve`.
Check it's running:
```bash
curl http://127.0.0.1:8000/retrieve -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "topk": 3}'
```

### Slow training speed
Expected throughput on V100 is **3-5× slower** than H20/A100 due to:
- FP16 vs BF16 (similar TFLOPS but less stable numerically)
- SDPA vs FlashAttention-2 (~1.5-2× slower attention)
- Smaller batch sizes (less GPU utilization)
- CPU offloading overhead

Estimated wall time:
- Search-R1 (2 epochs): ~2-3 days
- ASearch (300 steps): ~5-8 days

## Customization

All scripts accept extra Hydra overrides as trailing arguments:
```bash
# Change model path to a local checkpoint
bash scripts/v100/run_search_r1_v100.sh \
    actor_rollout_ref.model.path=/path/to/local/model

# Disable wandb logging
bash scripts/v100/run_search_r1_v100.sh \
    trainer.logger='["console"]'

# Use a different retrieval server
TOOL_CONFIG=/path/to/custom_tool_config.yaml \
    bash scripts/v100/run_search_r1_v100.sh
```

## Documentation

| 文档 | 内容 |
|------|------|
| [TRAINING_MANUAL.md](TRAINING_MANUAL.md) | 完整训练手册（从零开始） |
| [TRAINING_MILESTONES.md](TRAINING_MILESTONES.md) | 训练里程碑 & 日志样例 |
| [TRAINING_RESOURCES.md](TRAINING_RESOURCES.md) | 论文、数据集、模型链接汇总 |
| [EVALUATION_PLAN.md](EVALUATION_PLAN.md) | 评测计划、数据集链接、目标分数 |
| [TUNING_GUIDE.md](TUNING_GUIDE.md) | 超参调优决策树 |
| [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) | 性能分析与问题定位 |
