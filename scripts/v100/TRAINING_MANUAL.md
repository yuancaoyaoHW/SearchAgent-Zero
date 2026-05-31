# SearchAgent-Zero 完整训练手册

> 面向训练新手的端到端指南。从拿到一台 8×V100-32GB 服务器开始，到训练出一个能多轮搜索的 Agent 模型。

---

## 目录

1. [前置知识](#1-前置知识)
2. [硬件与环境要求](#2-硬件与环境要求)
3. [第一阶段：环境安装](#3-第一阶段环境安装)
4. [第二阶段：数据准备](#4-第二阶段数据准备)
5. [第三阶段：检索服务部署](#5-第三阶段检索服务部署)
6. [第四阶段：训练启动](#6-第四阶段训练启动)
7. [第五阶段：训练监控](#7-第五阶段训练监控)
8. [第六阶段：模型评估](#8-第六阶段模型评估)
9. [第七阶段：模型导出与部署](#9-第七阶段模型导出与部署)
10. [性能分析与问题定位](#10-性能分析与问题定位)
11. [常见问题 FAQ](#11-常见问题-faq)

---

## 1. 前置知识

### 1.1 你需要了解的概念

| 概念 | 一句话解释 | 为什么重要 |
|------|-----------|-----------|
| GRPO | 无 Critic 的强化学习算法，对同一 prompt 采样多个回答，用组内相对奖励更新策略 | 本项目的核心训练算法 |
| FSDP | PyTorch 的分布式数据并行，把模型参数分片到多卡 | 让 3B/8B 模型装进 V100 |
| vLLM | 高性能 LLM 推理引擎，用 PagedAttention 管理 KV cache | 负责 rollout（生成训练轨迹） |
| AgentLoop | verl 的多轮工具调用框架，模型生成 → 调工具 → 拼结果 → 继续生成 | 搜索 Agent 的核心循环 |
| Rollout | 用当前策略模型生成一批轨迹（prompt → 多轮搜索 → 最终回答） | RL 训练的数据来源 |
| Reward | 对轨迹打分（答案正确性），用于计算优势函数 | 驱动模型学习的信号 |
| KV Cache | 推理时缓存的 Key/Value 张量，避免重复计算 | 决定能处理多长的上下文 |
| Offload | 把参数/优化器状态放到 CPU 内存，节省 GPU 显存 | V100 显存有限，必须开启 |

### 1.2 训练流程总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    SearchAgent-Zero 训练流程                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 环境安装  │ →  │ 数据准备  │ →  │ 检索服务  │ →  │ 开始训练  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                       │          │
│                                                       ▼          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 模型部署  │ ←  │ 模型评估  │ ←  │ 训练监控  │ ←  │  Rollout  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 训练路线

| 路线 | 模型 | 架构 | 搜索轮数 | 难度 | 预计时间 |
|------|------|------|---------|------|---------|
| **Search-R1**（推荐新手） | Qwen2.5-3B-Instruct | Qwen2.5 | 4 轮 | ⭐⭐ | 2-3 天 |
| **Search-R1** | Qwen3-1.7B | Qwen3 | 4 轮 | ⭐⭐ | 1-2 天 |
| **Search-R1** | Qwen3-4B | Qwen3 | 4 轮 | ⭐⭐⭐ | 2-3 天 |
| **ASearch** | Qwen3-4B | Qwen3 | 30-50 轮 | ⭐⭐⭐ | 3-5 天 |
| **ASearch**（进阶） | Qwen3-8B | Qwen3 | 30-50 轮 | ⭐⭐⭐⭐ | 5-8 天 |
| **Search-R1**（实验性） | Qwen3-30B-A3B | Qwen3 MoE | 4 轮 | ⭐⭐⭐⭐⭐ | 3-5 天 |

**新手建议**：先跑通 Qwen2.5-3B Search-R1 或 Qwen3-1.7B Search-R1，理解整个流程后再尝试更大模型。

### 1.4 Qwen3 架构特性

Qwen3 相比 Qwen2.5 有以下关键架构差异，影响训练配置：

| 特性 | Qwen2.5 | Qwen3 | 对训练的影响 |
|------|---------|-------|------------|
| QK LayerNorm | ❌ | ✅ per-head RMSNorm | 训练更稳定，LR 可稍大 |
| Sliding Window | 全层固定 | 交替层（全局+局部） | 长序列效率更好 |
| Thinking Mode | ❌ | ✅ `<think>...</think>` | 需配置开关，影响序列长度 |
| Tie Embeddings | ✅ (小模型) | ❌ (全部 untied) | 显存多占一份 embedding |
| MoE 变体 | 无 | 30B-A3B (128 experts, top-2) | 需要 TP=4 |

#### Thinking Mode 说明

Qwen3 支持在回答前生成 `<think>...</think>` 推理块。在搜索 Agent 训练中：

- **小模型 (1.7B/4B) + Search-R1**：可开启，序列短，显存充裕
- **8B + ASearch**：建议关闭，50 轮对话已经很长
- **MoE**：建议关闭，KV cache 开销大

通过环境变量控制：`ENABLE_THINKING=true bash run_qwen3_1.7b_search_r1_v100.sh`

#### V100 资源需求速查

| 模型 | TP | 显存/GPU | 备注 |
|------|-----|---------|------|
| Qwen3-1.7B | 1 | ~10-12 GB | 非常宽裕 |
| Qwen3-4B | 1 | ~16-18 GB | 舒适 |
| Qwen3-4B (ASearch) | 1 | ~20-24 GB | 较紧 |
| Qwen3-8B (ASearch) | 2 | ~28-30 GB | 接近极限 |
| Qwen3-30B-A3B | 4 | ~26-30 GB | 实验性 |

---

## 2. 硬件与环境要求

### 2.1 最低硬件配置

```
GPU:     8× NVIDIA V100-32GB (sm_70)
CPU:     32 核以上（Ray worker 需要）
内存:    256 GB（CPU offload 需要大内存）
磁盘:    500 GB 可用空间（模型 + 索引 + checkpoint）
网络:    能访问 HuggingFace（下载模型和数据）
```

### 2.2 软件要求

```
OS:      Ubuntu 20.04 / 22.04
CUDA:    12.1 / 12.2 / 12.4
Driver:  >= 535.x
conda:   Miniconda 或 Anaconda
git:     >= 2.x
```

### 2.3 确认硬件

```bash
# 确认 GPU 型号和显存
nvidia-smi

# 确认 CUDA 版本
nvcc --version

# 确认内存（需要 256GB+）
free -h

# 确认磁盘空间
df -h /home
```

---

## 3. 第一阶段：环境安装

### 3.1 克隆项目

```bash
cd ~
git clone https://github.com/NLPJCL/SearchAgent-Zero.git
cd SearchAgent-Zero
```

### 3.2 安装训练环境

```bash
bash scripts/v100/install_train_env.sh
```

这个脚本会：
- 创建 conda 环境 `verl-v100`（Python 3.11）
- 安装 PyTorch 2.4.0 + CUDA 12.1
- 安装 vLLM 0.6.6.post1（V100 兼容版）
- 安装 xformers 0.0.28.post3（替代 FlashAttention-2）
- 安装所有 Python 依赖
- 以 editable 模式安装 SearchAgent-Zero

**预计耗时**：10-15 分钟

**验证安装**：
```bash
conda activate verl-v100
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 应输出: Tesla V100-SXM2-32GB 或类似

python -c "import vllm; print(vllm.__version__)"
# 应输出: 0.6.6.post1
```

### 3.3 安装检索服务环境

```bash
bash scripts/v100/install_retrieval_env.sh
```

这是一个独立的 conda 环境 `retriever-v100`，用于运行本地搜索引擎。

**预计耗时**：5-10 分钟

### 3.4 常见安装问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `pip install vllm` 编译失败 | CUDA 版本不匹配 | 确认 `nvcc --version` 是 12.x |
| `xformers` 安装报错 | PyTorch 版本不匹配 | 确保先装 torch==2.4.0 |
| `conda create` 很慢 | 默认源慢 | 换清华源：`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main` |
| `ray` 启动失败 | 端口冲突 | `ray stop --force` 后重试 |

---

## 4. 第二阶段：数据准备

### 4.1 一键下载和预处理

```bash
conda activate verl-v100
bash scripts/v100/download_data.sh
```

这个脚本会：
1. 下载 wiki-18 向量索引（~12GB）
2. 下载 wiki-18 语料库（~3GB）
3. 预处理 Search-R1 训练数据
4. 预处理 ASearcher 训练数据

**预计耗时**：30-60 分钟（取决于网速）

### 4.2 验证数据

```bash
# 检查检索索引
ls -lh examples/search_agent_rl/local_dense_retriever/search_data/
# 应该看到:
#   e5_Flat.index  (~12GB)
#   wiki-18.jsonl  (~8GB 解压后)

# 检查训练数据
python -c "
import pandas as pd
df = pd.read_parquet('examples/search_agent_rl/search_r1_processed/train_search_r1.parquet')
print(f'Search-R1 训练集: {len(df)} 条')
print(f'字段: {list(df.columns)}')
print(df.iloc[0]['prompt'][:200])
"
```

### 4.3 数据格式说明

训练数据是 parquet 格式，每条包含：

```python
{
    "prompt": [{"role": "user", "content": "问题..."}],  # 对话格式
    "data_source": "nq",                                   # 数据来源
    "reward_model": {"ground_truth": "答案"},              # 奖励计算用
    "extra_info": {"split": "train", ...}                  # 元信息
}
```

---

## 5. 第三阶段：检索服务部署

### 5.1 启动检索服务

在一个 **单独的 tmux/screen 会话** 中运行：

```bash
# 新建 tmux 会话
tmux new -s retriever

# 激活检索环境
conda activate retriever-v100

# 启动（默认用 GPU 0，端口 8000）
bash scripts/v100/start_retrieval_server.sh
```

等待看到类似输出：
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 5.2 验证检索服务

在另一个终端：

```bash
curl -s http://127.0.0.1:8000/retrieve \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "topk": 3}' | python -m json.tool
```

应该返回包含 Wikipedia 文档的 JSON 结果。

### 5.3 检索服务架构

```
训练进程 (8 GPU)
    │
    │  HTTP POST /retrieve
    ▼
检索服务 (1 GPU)
    │
    ├── e5-base-v2 编码查询
    ├── faiss-gpu 向量检索
    └── 返回 top-3 文档
```

**注意**：检索服务和训练共享 GPU 0。训练脚本的 `gpu_memory_utilization` 已经调低，给检索服务留了空间。如果 OOM，可以把检索服务移到 CPU：

```bash
# CPU 模式启动（去掉 --faiss_gpu）
CUDA_VISIBLE_DEVICES="" python examples/search_agent_rl/local_dense_retriever/retrieval_server.py \
    --index_path ... --corpus_path ... --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2
```

---

## 6. 第四阶段：训练启动

### 6.1 Search-R1 训练（推荐新手首次运行）

```bash
# 新终端
conda activate verl-v100
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=your_wandb_api_key_here  # 从 wandb.ai 获取

# 启动训练
bash scripts/v100/run_search_r1_v100.sh
```

**首次运行会**：
1. 下载 Qwen2.5-3B-Instruct 模型（~6GB）
2. 启动 Ray cluster
3. 初始化 vLLM 推理引擎
4. 开始第一个 epoch 的 rollout

**正常启动标志**：
```
[INFO] Starting PPO training...
[INFO] Rollout batch 1/X ...
```

### 6.2 ASearch 训练（进阶）

```bash
conda activate verl-v100
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=your_wandb_api_key_here

bash scripts/v100/run_asearch_v100.sh
```

**注意**：ASearch 首次 rollout 会很慢（每条轨迹最多 30-50 轮搜索），这是正常的。

### 6.3 训练参数速查

#### Search-R1 关键参数

| 参数 | 值 | 含义 |
|------|---|------|
| `data.train_batch_size` | 256 | 每步采样 256 个 prompt |
| `rollout.n` | 5 | 每个 prompt 生成 5 条轨迹 |
| `max_response_length` | 3000 | 最大回答长度（token） |
| `max_assistant_turns` | 4 | 最多 4 轮搜索 |
| `ppo_mini_batch_size` | 128 | PPO 更新的 mini-batch |
| `lr` | 1e-6 | 学习率 |
| `total_epochs` | 2 | 训练 2 个 epoch |

#### ASearch 关键参数

| 参数 | 值 | 含义 |
|------|---|------|
| `data.train_batch_size` | 128 | 每步采样 128 个 prompt |
| `rollout.n` | 4 | 每个 prompt 生成 4 条轨迹 |
| `max_response_length` | 16384 | 最大回答长度 |
| `max_assistant_turns` | 50 | 最多 50 轮搜索 |
| `turn_limit_schedule` | 0:30,50:40,... | 逐步放开搜索轮数 |
| `ppo_mini_batch_size` | 32 | PPO mini-batch |
| `lr` | 5e-7 | 更小的学习率（8B 模型） |

### 6.4 恢复训练（从 checkpoint）

```bash
# 找到最新 checkpoint
ls -lt output/qwen2.5-3b-instruct_searchr1_v100/

# 从 checkpoint 恢复
bash scripts/v100/run_search_r1_v100.sh \
    trainer.resume_from_checkpoint=output/qwen2.5-3b-instruct_searchr1_v100/step_250
```

---

## 7. 第五阶段：训练监控

### 7.1 Weights & Biases 监控

训练启动后，打开 https://wandb.ai 查看实时曲线。

**关键指标解读**：

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| `reward/mean` | 持续上升 | 长期不涨或下降 → 学习率/batch 问题 |
| `reward/std` | 逐渐减小 | 持续很大 → 模型不稳定 |
| `actor/loss` | 小幅波动 | 突然飙升 → loss spike |
| `actor/entropy` | 缓慢下降 | 骤降到 0 → 模式坍塌 |
| `turn/tool_call_success_rate/mean` | > 0.8 | < 0.5 → 工具调用格式学坏了 |
| `turn/tool_call_turn/mean` | 逐步增加 | 一直是 1 → 没学会多轮搜索 |
| `abnormal_trajectory/*_percentage` | < 0.2 | > 0.5 → 需要调整过滤策略 |

### 7.2 终端日志监控

```bash
# 实时查看日志
tail -f logs/qwen2.5-3b-instruct_searchr1_v100.log

# 过滤关键信息
grep -E "reward|loss|step" logs/qwen2.5-3b-instruct_searchr1_v100.log | tail -20
```

### 7.3 GPU 监控

```bash
# 实时 GPU 使用率
watch -n 1 nvidia-smi

# 更详细的监控（推荐）
nvidia-smi dmon -s pucvmet -d 5
# p=power, u=utilization, c=clock, v=violation, m=memory, e=ecc, t=temperature
```

### 7.4 Ray Dashboard

```bash
# Ray 会自动启动 dashboard
# 默认地址: http://localhost:8265
# 可以看到 worker 状态、任务队列、资源使用
```

---

## 8. 第六阶段：模型评估

### 8.1 评测数据集

所有评测数据集来自 [RUC-NLPIR/FlashRAG_datasets](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)，通过 `data_source` 参数选择不同子集。

| 数据集 | HuggingFace 子集名 | 类型 | 说明 |
|--------|-------------------|------|------|
| NQ (Natural Questions)† | `nq` | 单跳问答 | In-domain（训练集包含 NQ） |
| TriviaQA* | `triviaqa` | 单跳问答 | Out-of-domain |
| PopQA* | `popqa` | 长尾知识 | Out-of-domain |
| HotpotQA† | `hotpotqa` | 多跳推理 | In-domain（训练集包含 HotpotQA） |
| 2WikiMultihopQA* | `2wikimultihopqa` | 多跳推理 | Out-of-domain |
| MuSiQue* | `musique` | 多跳推理 | Out-of-domain |
| Bamboogle* | `bamboogle` | 组合推理 | Out-of-domain |

> † = In-domain 数据集（训练数据中包含同源训练集）
> \* = Out-of-domain 数据集（仅用于评测，训练中未见过）

**数据集下载方式**：

```python
# 方式一：通过 HuggingFace datasets 库加载
import datasets
ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")  # 替换为对应子集名

# 方式二：评测脚本会自动加载，无需手动下载
```

**数据集参考链接**：
- FlashRAG 数据集合集：https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
- Search-R1 训练数据（NQ + HotpotQA）：https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
- Search-R1 原始项目：https://github.com/PeterGriffinJin/Search-R1

### 8.2 运行评估

```bash
conda activate verl-v100

# 评估 Search-R1 训练的模型（使用 val_only 模式）
bash run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh
```

评估脚本会加载训练好的 checkpoint（默认路径 `./output/qwen2.5-3b-instruct_searchr1/global_step_500/merged_hf_model`），在测试集上进行 rollout 并计算 reward（EM 匹配）。

### 8.3 评测目标与基线对比

| 数据集 | Search-R1 (Qwen2.5-3B-Instruct) | verl AgentLoop 复现 (Qwen2.5-3B-Instruct) | Abs. Gain | Rel. Gain |
|--------|:-------------------------------:|:-----------------------------------------:|:---------:|:---------:|
| NQ | 0.341 | 0.4640 | +0.1230 | +36.1% |
| TriviaQA* | 0.545 | 0.6164 | +0.0714 | +13.1% |
| PopQA* | 0.378 | 0.4239 | +0.0459 | +12.1% |
| HotpotQA† | 0.324 | 0.4225 | +0.0985 | +30.4% |
| 2Wiki* | 0.319 | 0.3979 | +0.0789 | +24.7% |
| Musique* | 0.103 | 0.1808 | +0.0778 | +75.5% |
| Bamboogle* | 0.264 | 0.3440 | +0.0800 | +30.3% |
| **Avg** | **0.325** | **0.4071** | **+0.0821** | **+25.3%** |

> 在完整 Search-R1 评测集合上，SearchAgent-Zero 将平均分从 0.325 提升到约 0.407，且所有数据集均取得正向提升，说明稳定的 RL infra、AgentLoop rollout 与异常轨迹处理会显著影响 Search Agent 训练效果。

### 8.4 评测计划

推荐按以下阶段进行评测：

**阶段一：训练中验证（自动）**
- 训练脚本每 `test_freq` 步（默认 50 步）自动在验证集上评估
- 关注 `reward/mean` 是否持续上升
- 验证集包含 NQ + HotpotQA 的 test split

**阶段二：中间 checkpoint 评测（可选）**
- 在 step 250 / step 500 等关键节点导出模型
- 运行完整评测脚本观察各数据集分数变化趋势

```bash
# 修改 eval 脚本中的模型路径指向中间 checkpoint
export EVAL_MODEL_PATH="./output/qwen2.5-3b-instruct_searchr1/global_step_250/merged_hf_model"
bash run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh
```

**阶段三：最终模型全量评测**
- 训练完成后，在全部 7 个数据集上运行评测
- 对比上表中的基线和目标分数
- 重点关注 Out-of-domain 数据集（标 * 的）的泛化能力

**评测通过标准**：
| 指标 | 最低要求 | 优秀 |
|------|---------|------|
| 平均分 | ≥ 0.38 | ≥ 0.40 |
| 所有数据集正向提升 | ✅ | ✅ |
| Musique（最难） | ≥ 0.14 | ≥ 0.18 |
| NQ（in-domain） | ≥ 0.42 | ≥ 0.46 |

### 8.5 手动测试模型

```python
# 用 vLLM 加载训练好的模型进行交互测试
from vllm import LLM, SamplingParams

model = LLM(
    model="output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint",
    dtype="float16",
    gpu_memory_utilization=0.8,
)

prompt = "What year was the Eiffel Tower completed?"
output = model.generate([prompt], SamplingParams(temperature=0.7, max_tokens=2048))
print(output[0].outputs[0].text)
```

---

## 9. 第七阶段：模型导出与部署

### 9.1 导出 HuggingFace 格式

训练产出的 checkpoint 已经是 HuggingFace 格式，可以直接加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint"
)
```

### 9.2 上传到 HuggingFace Hub

```bash
pip install huggingface_hub
huggingface-cli login

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint',
    repo_id='your-username/searchagent-zero-3b',
    repo_type='model',
)
"
```

---

## 10. 性能分析与问题定位

> 详见 [PERFORMANCE_GUIDE.md](./PERFORMANCE_GUIDE.md)

---

## 11. 常见问题 FAQ

### Q: 训练多久能看到效果？
**A**: Search-R1 通常在 50-100 步后 reward 开始明显上升。如果 200 步后还没动静，检查检索服务是否正常。

### Q: 可以中途停止再恢复吗？
**A**: 可以。Ctrl+C 停止后，用 `trainer.resume_from_checkpoint=<path>` 恢复。checkpoint 默认每 250 步保存一次。

### Q: 显存不够怎么办？
**A**: 按优先级尝试：
1. 降低 `gpu_memory_utilization`（0.65 → 0.55）
2. 降低 `rollout.n`（5 → 3）
3. 降低 `ppo_micro_batch_size_per_gpu`（16 → 8）
4. 降低 `max_model_len`（15000 → 12000）

### Q: 训练速度太慢？
**A**: V100 上 Search-R1 每步约 3-5 分钟是正常的。如果超过 10 分钟/步，检查：
1. 检索服务是否响应慢（`curl` 测试）
2. CPU offload 是否导致瓶颈（`htop` 看 CPU 使用率）
3. 是否有 GPU 空闲（`nvidia-smi` 看利用率）

### Q: 模型生成乱码/不调用工具？
**A**: 训练初期正常。如果持续 100+ 步还是这样：
1. 检查 `tool_call_success_rate` 指标
2. 确认 `attn_implementation=sdpa` 设置正确
3. 尝试降低学习率

### Q: Ray 报错 "No available node"？
**A**: 
```bash
ray stop --force
# 等 10 秒
ray start --head --num-gpus=8
# 然后重新启动训练
```

---

## 附录 A：完整文件清单

```
scripts/v100/
├── README.md                    # 部署脚本说明
├── TRAINING_MANUAL.md           # 本文档
├── TRAINING_RESOURCES.md        # 资料/数据集/论文汇总
├── PERFORMANCE_GUIDE.md         # 性能分析指南
├── install_train_env.sh         # 训练环境安装
├── install_retrieval_env.sh     # 检索环境安装
├── download_data.sh             # 数据下载
├── start_retrieval_server.sh    # 检索服务启动
├── run_search_r1_v100.sh        # Search-R1 训练
├── run_asearch_v100.sh          # ASearch 训练
└── training-tracker/            # 前端训练进度追踪工具
```
