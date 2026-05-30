# SearchAgent-Zero 训练资料汇总

> 本文档整理了在 8×V100-32GB 上训练 Search Agent 所需的全部资料：论文、文档、数据集、模型、工具。

---

## 一、核心论文 & 技术报告

### 1.1 本项目直接相关

| 论文/报告 | 链接 | 说明 |
|-----------|------|------|
| SearchAgent-Zero 技术报告 | https://zhuanlan.zhihu.com/p/2042036895199278392 | 本项目官方技术报告（中文） |
| Search-R1 | https://github.com/PeterGriffinJin/Search-R1 | 原始 Search-R1 框架，本项目的基线 |
| Cut the Bill, Keep the Turns: Affordable Multi-Turn Search RL | https://agate-slipper-ef0.notion.site/Cut-the-Bill-Keep-the-Turns-Affordable-Multi-Turn-Search-RL-003f78214a4d451fb06f453d084e666c | ASearch 多轮搜索 RL 方法论 |
| verl: Volcano Engine RL for LLM | https://arxiv.org/abs/2409.19256v2 | 底层 RL 训练框架论文 |

### 1.2 RL 算法基础

| 论文 | 链接 | 说明 |
|------|------|------|
| GRPO (DeepSeekMath) | https://arxiv.org/pdf/2402.03300 | 本项目使用的核心 RL 算法，无 critic |
| PPO (Proximal Policy Optimization) | https://arxiv.org/abs/1707.06347 | PPO 原始论文，GRPO 的基础 |
| GAE (Generalized Advantage Estimation) | https://arxiv.org/abs/1506.02438 | 优势估计方法 |
| DAPO | https://arxiv.org/abs/2503.14476 | 改进的 GRPO 变体，verl 已支持 |

### 1.3 异步训练 & 系统优化

| 论文 | 链接 | 说明 |
|------|------|------|
| AReaL: Asynchronous RL for Language Reasoning | https://arxiv.org/abs/2505.24298 | 异步 RL 系统设计 |
| StreamRL: Scalable Heterogeneous Elastic RL | https://arxiv.org/abs/2504.15930 | 流式 RL 训练 |
| Magistral | https://arxiv.org/abs/2506.10910 | 异步策略训练 |
| AsyncFlow | https://arxiv.org/abs/2507.01663 | 异步流式 RL 框架 |

### 1.4 Agentic / Tool-use RL

| 论文 | 链接 | 说明 |
|------|------|------|
| Retool | https://github.com/verl-project/verl-recipe/tree/main/retool | 工具调用 RL 复现 |
| PRIME | https://arxiv.org/abs/2405.00675 | Process Reward Model |
| Entropy Mechanism of RL | https://github.com/PRIME-RL/Entropy-Mechanism-of-RL | RL 熵机制分析 |

---

## 二、项目文档（本仓库内）

### 2.1 必读文档

| 文档 | 路径 | 内容 |
|------|------|------|
| 项目 README | `README.md` / `README_zh.md` | 整体介绍、快速开始、结果 |
| Agent Loop 设计 | `docs/advance/agent_loop.rst` | AgentLoop 架构、API 设计 |
| Agentic RL 训练 | `docs/start/agentic_rl.rst` | 多轮工具调用训练入门 |
| Fully Async 训练 | `docs/advance/fully_async.md` | 全异步策略训练（V100 不建议用） |
| GRPO 算法说明 | `docs/algo/grpo.md` | GRPO 配置详解 |
| PPO 算法说明 | `docs/algo/ppo.md` | PPO 配置详解 |
| 数据准备 | `docs/preparation/prepare_data.rst` | parquet 数据格式要求 |
| 安装指南 | `docs/start/install.rst` | 官方安装流程 |
| Attention 实现 | `docs/advance/attention_implementation.rst` | SDPA/FA2 切换说明 |
| Checkpoint | `docs/advance/checkpoint.rst` | 模型保存与恢复 |
| PPO LoRA | `docs/advance/ppo_lora.rst` | LoRA 微调配置 |

### 2.2 配置文件（训练核心）

| 文件 | 路径 | 说明 |
|------|------|------|
| 主训练配置 | `examples/search_agent_rl/config/search_multiturn_grpo.yaml` | Hydra 主配置 |
| 搜索工具配置 | `examples/search_agent_rl/config/tool_config/search_tool_config.yaml` | SearchTool 定义 |
| Agent Loop 配置 | `examples/search_agent_rl/config/agent_loop/tool_agent_credit_assignment.yaml` | Credit Assignment AgentLoop |
| PPO Trainer 默认配置 | `verl/trainer/config/ppo_trainer.yaml` | 全量默认参数 |
| 模型配置 | `verl/trainer/config/model/hf_model.yaml` | LoRA、dtype 等 |
| Rollout 配置 | `verl/trainer/config/rollout/rollout.yaml` | vLLM/SGLang rollout 参数 |

### 2.3 训练脚本

| 脚本 | 路径 | 说明 |
|------|------|------|
| Search-R1 训练 (H20) | `run_qwen2.5_3b_instruct_search_multiturn_SearchR1.sh` | 原始 3B 训练脚本 |
| Search-R1 评估 | `run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh` | 评估脚本 |
| ASearch 同步训练 (H20) | `run_qwen3_8b_instruct_search_multiturn_ASearch.sh` | 原始 8B 训练脚本 |
| ASearch 全异步 (H20) | `run_qwen3_8b_instruct_search_multiturn_ASearch_fully_async.sh` | 全异步版本 |
| **V100 Search-R1** | `scripts/v100/run_search_r1_v100.sh` | V100 适配版 |
| **V100 ASearch** | `scripts/v100/run_asearch_v100.sh` | V100 适配版 |

---

## 三、数据集

### 3.1 训练数据集

| 数据集 | HuggingFace 链接 | 用途 | 大小 |
|--------|-----------------|------|------|
| Search-R1 训练数据 (NQ + HotpotQA) | https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train | Search-R1 配方训练集 | ~50K 样本 |
| ASearcher (英文无数学) | https://huggingface.co/datasets/aidenjhwu/ASearcher_en_no-math_Qwen3-8B-reject-sample | ASearch 配方训练集（reject sampling） | ~10K+ 样本 |

### 3.2 检索索引 & 语料库

| 数据 | HuggingFace 链接 | 用途 | 大小 |
|------|-----------------|------|------|
| Wiki-18 E5 Dense Index | https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index | faiss 向量索引（e5-base-v2） | ~12 GB |
| Wiki-18 Corpus | https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus | 检索语料（wiki-18.jsonl.gz） | ~3 GB |

### 3.3 评估数据集

| 数据集 | HuggingFace 链接 | 子集名 | 类型 |
|--------|-----------------|--------|------|
| FlashRAG 数据集合集 | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | — | 评测数据统一来源 |
| NQ (Natural Questions)† | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | `nq` | 单跳问答（in-domain） |
| TriviaQA* | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | `triviaqa` | 单跳问答（out-of-domain） |
| PopQA* | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | `popqa` | 长尾知识（out-of-domain） |
| HotpotQA† | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | `hotpotqa` | 多跳推理（in-domain） |
| 2WikiMultihopQA* | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | `2wikimultihopqa` | 多跳推理（out-of-domain） |
| MuSiQue* | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | `musique` | 多跳推理（out-of-domain） |
| Bamboogle* | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets | `bamboogle` | 组合推理（out-of-domain） |

> † = In-domain，* = Out-of-domain。评估数据集通过 Search-R1 的评估脚本自动加载（`datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "<子集名>")`），无需单独下载。
>
> 参考：Search-R1 评测数据处理脚本 https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/qa_search_test_merge.py

---

## 四、预训练模型

| 模型 | HuggingFace 链接 | 用途 |
|------|-----------------|------|
| Qwen2.5-3B-Instruct | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct | Search-R1 配方基座模型 |
| Qwen3-8B | https://huggingface.co/Qwen/Qwen3-8B | ASearch 配方基座模型 |
| intfloat/e5-base-v2 | https://huggingface.co/intfloat/e5-base-v2 | 检索服务 embedding 模型 |

---

## 五、外部工具 & 依赖

### 5.1 核心框架

| 工具 | 链接 | 版本要求（V100） |
|------|------|-----------------|
| verl (本项目) | https://github.com/verl-project/verl | 当前 main |
| vLLM | https://github.com/vllm-project/vllm | 0.6.6.post1（V100）/ 0.11.0（A100+） |
| PyTorch | https://pytorch.org | 2.4.0 + CUDA 12.1（V100） |
| xformers | https://github.com/facebookresearch/xformers | 0.0.28.post3（V100 attention 后端） |
| Ray | https://github.com/ray-project/ray | >= 2.41.0 |
| transformers | https://github.com/huggingface/transformers | >= 4.45.0 |

### 5.2 检索服务

| 工具 | 链接 | 说明 |
|------|------|------|
| faiss-gpu | https://github.com/facebookresearch/faiss | GPU 向量检索 |
| pyserini | https://github.com/castorini/pyserini | 稀疏检索（可选） |
| FastAPI + Uvicorn | https://fastapi.tiangolo.com | 检索 HTTP 服务 |

### 5.3 监控 & 日志

| 工具 | 链接 | 说明 |
|------|------|------|
| Weights & Biases | https://wandb.ai | 训练曲线、指标监控 |
| MLflow | https://mlflow.org | Agent rollout trace 可视化 |
| TensorBoard | — | 备选日志后端 |

---

## 六、关键源码路径

```
verl/experimental/agent_loop/
├── agent_loop.py                          # AgentLoopBase 基类
├── tool_agent_loop.py                     # ToolAgentLoop（Search-R1 用）
└── tool_agent_loop_credit_assignment.py   # ToolAgentLoop + Credit Assignment（ASearch 用）

verl/tools/
├── base_tool.py                           # BaseTool 抽象
├── search_tool.py                         # SearchTool（HTTP 调检索服务）
└── mcp_search_tool.py                     # MCP 协议搜索工具

verl/trainer/
├── main_ppo.py                            # 训练入口
└── config/                                # Hydra 配置

verl/workers/
├── config/                                # actor/rollout/engine 配置 dataclass
├── engine/fsdp/                           # FSDP 训练引擎
└── rollout/vllm_rollout/                  # vLLM rollout 实现

examples/search_agent_rl/
├── config/                                # 搜索 agent 专用配置
├── local_dense_retriever/                 # 本地检索服务
├── preprocess_search_r1_dataset_new.py    # Search-R1 数据预处理
└── preprocess_ASearcher_dataset.py        # ASearcher 数据预处理
```

---

## 七、参考社区资源

| 资源 | 链接 | 说明 |
|------|------|------|
| verl 官方文档 | https://verl.readthedocs.io | 完整 API 文档 |
| verl-recipe | https://github.com/verl-project/verl-recipe | 社区训练配方集合 |
| verl Docker 镜像 | https://hub.docker.com/r/verlai/verl | 预构建镜像 |
| Search-R1 检索器文档 | https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md | 检索服务详细说明 |
| LangGraph Agent 示例 | https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent/example | LangGraph 集成 |

---

## 八、快速检查清单

在 V100 上开始训练前，确认以下资源已就绪：

- [ ] **模型下载完成**：`Qwen/Qwen2.5-3B-Instruct` 和/或 `Qwen/Qwen3-8B`
- [ ] **检索模型下载完成**：`intfloat/e5-base-v2`
- [ ] **检索索引下载完成**：`PeterJinGo/wiki-18-e5-index`（~12GB）
- [ ] **检索语料下载完成**：`PeterJinGo/wiki-18-corpus`（~3GB）
- [ ] **训练数据预处理完成**：`train_search_r1.parquet` / `ASearcher_train.parquet`
- [ ] **训练环境安装完成**：`conda activate verl-v100`
- [ ] **检索环境安装完成**：`conda activate retriever-v100`
- [ ] **检索服务运行中**：`curl http://127.0.0.1:8000/retrieve` 返回正常
- [ ] **WANDB_API_KEY 已设置**
- [ ] **8 张 V100-32GB 可用**：`nvidia-smi` 确认
