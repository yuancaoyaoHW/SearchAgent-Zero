# RL/SFT Search Agent 研究方向综合报告

> 生成时间：2026-05-31 | 基于 SearchAgent-Zero fork 代码分析 + 论文调研 + 实验设计

---

## 目录

1. [仓库技术分析](#一仓库技术分析)
2. [论文与开源项目综述](#二论文与开源项目综述)
3. [关键技术判断](#三关键技术判断)
4. [研究方向与实验设计](#四研究方向与实验设计)
5. [优先级总览](#五优先级总览)

---

## 一、仓库技术分析

### 1.1 项目定位

SearchAgent-Zero 是基于 verl 框架的 Search Agent 强化学习训练系统，覆盖：
- **短程搜索**：Search-R1 风格 QA（2-4 轮搜索）
- **长程搜索**：ASearch 风格多轮搜索（最多 128 轮工具调用）

核心创新：异常轨迹过滤、credit assignment、搜索结果 summary 压缩、同步 GRPO + 全异步训练双模式。

### 1.2 核心目录结构

```
SearchAgent-Zero/
├── examples/search_agent_rl/          # 配置、数据预处理、retriever
│   ├── config/
│   │   ├── search_multiturn_grpo.yaml       # 主训练配置
│   │   ├── tool_config/search_tool_config.yaml
│   │   └── agent_loop/tool_agent_credit_assignment.yaml
│   ├── local_dense_retriever/               # 本地检索服务 (FastAPI + BM25/Dense)
│   ├── preprocess_search_r1_dataset_new.py
│   └── preprocess_ASearcher_dataset.py
├── verl/
│   ├── experimental/agent_loop/             # AgentLoop 核心
│   │   ├── tool_agent_loop.py               # 基础 ToolAgentLoop
│   │   ├── tool_agent_loop_credit_assignment.py  # Credit assignment 版本
│   │   └── tool_parser.py                   # Hermes format tool call 解析
│   ├── experimental/fully_async_policy/     # 全异步训练
│   │   ├── fully_async_main.py
│   │   ├── fully_async_trainer.py
│   │   └── fully_async_rollouter.py
│   ├── tools/search_tool.py                 # SearchTool (Ray remote)
│   ├── trainer/ppo/
│   │   ├── core_algos.py                    # GRPO/PPO/GAE 核心算法
│   │   └── rollout_corr_helper.py           # Off-policy 修正 (IS/RS)
│   └── utils/reward_score/
│       └── search_r1_like_qa_em.py          # EM + format reward
├── run_qwen2.5_3b_instruct_search_multiturn_SearchR1.sh
├── run_qwen3_8b_instruct_search_multiturn_ASearch.sh
└── run_qwen3_8b_instruct_search_multiturn_ASearch_fully_async.sh
```

### 1.3 训练入口

| 模式 | 入口 | 脚本 |
|------|------|------|
| 同步 GRPO | `verl.trainer.main_ppo` | `run_qwen2.5_3b_*_SearchR1.sh`, `run_qwen3_8b_*_ASearch.sh` |
| 全异步 | `verl.experimental.fully_async_policy.fully_async_main` | `run_*_fully_async.sh` |

### 1.4 数据格式

Parquet 文件，每行包含：
- `prompt`: `[{"role":"system",...}, {"role":"user","content":question}]`
- `reward_model`: `{"ground_truth": {"target": [answer_list]}}`
- `extra_info`: 含 `tools_kwargs.search.create_kwargs`（检索服务配置）

### 1.5 Rollout 与 Tool Call

AgentLoop 状态机：`PENDING → GENERATING → PROCESSING_TOOLS → GENERATING → ... → TERMINATED`

Tool call 格式（Hermes）：
```xml
<thought>分析问题...</thought>
<tool_call>{"name":"search","arguments":{"query_list":["query1"]}}</tool_call>
<tool_response>检索结果...</tool_response>
<thought>总结信息...</thought>
<answer>最终答案</answer>
```

Response mask：assistant token=1（参与梯度），tool response token=0（不参与）。

### 1.6 Reward 设计

`compute_score()` in `search_r1_like_qa_em.py`：
- `origin_score`: EM 匹配 → 1.0 / 0
- `format_score`: 格式状态机验证通过 → 0.1
- `efficiency_score`: 单个 `<answer>` 标签 → 0.5
- 惩罚：`<answer>` 超 10 个 → score/4

### 1.7 异常轨迹过滤

| 异常类型 | 触发条件 | 处理 |
|----------|----------|------|
| 重复 query | normalize 后已搜索过 | TERMINATED |
| tool 解析错误 | JSON 解析失败 | TERMINATED |
| 单轮 query 过多 | 超 `max_queries_per_tool_call` | TERMINATED |
| 轮数超限 | 超 `max_assistant_turns` | TERMINATED, mask=0 |
| 序列过长 | 超 `max_model_len` | TERMINATED, mask=0 |
| 重复搜索结果 | 文档签名重叠 ≥ 2/3 | 记录不终止 |

### 1.8 Credit Assignment

`tool_agent_loop_credit_assignment.py` 的核心改进：异常发生时只将**当前轮之前**的 response_mask 置 0，保留当前轮 mask=1 用于惩罚。效果：只惩罚产生异常的 turn，不波及之前合法搜索。

### 1.9 Summary Compression

两种模式：
- **Self-summary**：用训练中的模型自身生成摘要
- **External-summary**：调用外部 API（支持多 base_url 负载均衡）

### 1.10 全异步训练

架构：`FullyAsyncRollouter` + `FullyAsyncTrainer` 通过 `MessageQueue`（Ray actor）通信。
- 4 GPU rollout + 4 GPU training
- `staleness_threshold=0.5`（数据新鲜度）
- Off-policy 修正：IS weights + Rejection Sampling

### 1.11 可改造入口

| 改造方向 | 入口文件 | 关键函数 |
|----------|----------|----------|
| 自定义 reward | `verl/utils/reward_score/search_r1_like_qa_em.py` | `compute_score()` |
| 自定义 tool | `verl/tools/search_tool.py` | `SearchTool.execute()` |
| Agent loop 逻辑 | `verl/experimental/agent_loop/tool_agent_loop_credit_assignment.py` | `_handle_generating_state()` |
| Advantage 计算 | `verl/trainer/ppo/core_algos.py` | `register_adv_est` |
| Off-policy 修正 | `verl/trainer/ppo/rollout_corr_helper.py` | `compute_rollout_correction_and_rejection_mask()` |
| Turn limit | 训练脚本 | `TURN_LIMIT_SCHEDULE` 环境变量 |

---

## 二、论文与开源项目综述

### 2.1 方法谱系

#### A. 纯 RL 搜索 Agent（无 SFT cold-start）

| 项目 | 年份 | 机构 | RL 算法 | Reward | 关键结果 | 开源 |
|------|------|------|---------|--------|----------|------|
| **[Search-R1](https://github.com/PeterGriffinJin/Search-R1)** | 2025 | UIUC | PPO/GRPO | EM | Qwen2.5-7B 比 RAG +41% | ✅ |
| **[R1-Searcher](https://github.com/RUCAIBox/R1-Searcher)** | 2025 | RUC | GRPO/REINFORCE++ | 两阶段(format→EM) | 无需 SFT 的 cold-start 替代 | ✅ |
| **[R1-Searcher++](https://arxiv.org/abs/2505.00223)** | 2025 | RUC | GRPO/REINFORCE++ | EM+format | 动态知识获取 | ✅ |
| **[ReSearch](https://github.com/Agent-RL/ReSearch)** | 2025 | ZJU+Ant | GRPO | EM | 单数据集训练跨 benchmark 泛化 | ✅ |

#### B. RL + 模拟搜索引擎

| 项目 | 年份 | 机构 | 关键创新 | 开源 |
|------|------|------|----------|------|
| **[ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch)** | 2025 | Alibaba | 用 SFT 训练 LLM 模拟搜索引擎，零 API 成本 | ✅ |

#### C. 长 Horizon 搜索 Agent

| 项目 | 年份 | 机构 | 关键创新 | 开源 |
|------|------|------|----------|------|
| **[ASearcher](https://github.com/ASearcher-project/ASearcher)** | 2025 | ByteDance | 全异步 RL，128 轮 tool call，400k+ output tokens | ✅ |
| **[WebExplorer](https://github.com/THUDM/WebExplorer)** | 2025 | Zhipu AI | SFT cold-start + RL，search+browse 组合 | ✅ |

#### D. 轻量级/数据高效

| 项目 | 年份 | 机构 | 关键创新 | 开源 |
|------|------|------|----------|------|
| **[s3](https://arxiv.org/abs/2505.07834)** | 2025 | UIUC | 仅 2.4k 样本 + 20 PPO steps 超越 Search-R1 | ✅ |
| **[RAG-R1](https://arxiv.org/abs/2507.02962)** | 2025 | — | 多查询并行搜索 | 部分 |

#### E. 训练框架

| 项目 | 年份 | 机构 | 定位 |
|------|------|------|------|
| **[verl](https://github.com/volcengine/verl)** | 2024-25 | ByteDance | LLM RL 框架，支持 PPO/GRPO/DAPO |
| **[AgentRL](https://github.com/THUDM/AgentRL)** | 2025 | Tsinghua | 多轮多任务 Agent RL，全异步 |
| **[Forge](https://huggingface.co/blog/MiniMaxAI/forge)** | 2025 | MiniMax | 通用 Agent RL，混合域训练 |

#### F. 评测基准

| Benchmark | 特点 | 难度 |
|-----------|------|------|
| HotpotQA / 2WikiMQA / Musique | 多跳 QA，社区标准 | 中 |
| FRAMES | 824 题，需 2-15 篇文档，多类型推理 | 高 |
| BrowseComp | 1266 题，需持续多跳推理 | 极高 |
| HLE | 2500 题，顶尖模型 <10% | 极高 |

### 2.2 关键论文详细分析

#### Search-R1 (UIUC, COLM 2025)
- **方法**：直接从 base/instruct 模型开始 RL，无 SFT
- **核心技术**：retrieved token masking（tool response 不参与梯度）
- **数据**：NQ + HotpotQA ~170k 样本，2100 PPO steps
- **与本仓库关系**：SearchAgent-Zero 的 Search-R1 recipe 直接复现并改进此工作（avg 0.325→0.407）
- **可借鉴**：retrieved token masking；PPO vs GRPO 对比
- **问题**：搜索轮数有限；outcome reward 稀疏

#### ASearcher (ByteDance, NeurIPS 2025)
- **方法**：纯 RL，全异步训练，基于 AReaL 框架
- **核心技术**：自动 QA 数据合成 + 128 轮 turn limit
- **结果**：GAIA 58.1, FRAMES 74.5
- **与本仓库关系**：SearchAgent-Zero 的 ASearch recipe 和 fully_async 训练直接对标
- **可借鉴**：全异步解决长 trajectory 阻塞；自动数据合成
- **问题**：资源需求极大；off-policy 程度难控制

#### ZeroSearch (Alibaba, NeurIPS 2025)
- **方法**：用 SFT 训练 simulation LLM 模拟搜索引擎
- **核心技术**：Curriculum rollout（逐步增加检索难度）
- **结果**：超越使用真实搜索引擎的 Search-R1
- **可借鉴**：消除外部搜索 API 依赖；控制文档质量
- **问题**：simulation LLM 知识截止限制；分布差异

#### s3 (UIUC, EMNLP 2025)
- **方法**：解耦 searcher/generator，仅训练 searcher
- **核心技术**：GBR reward（Gain Beyond RAG）；只训练 RAG 失败 case
- **结果**：2.4k 样本 + 20 PPO steps 超越 Search-R1
- **可借鉴**：GBR reward 比纯 EM 更 informative；数据筛选策略
- **问题**：单轮检索，不支持多轮


---

## 三、关键技术判断

### 3.1 SFT Cold-Start 是否必要？

| 证据来源 | 结论 |
|----------|------|
| Search-R1 | 不必要。直接从 base model RL 即可学会搜索 |
| R1-Searcher | 不必要。两阶段 reward（先 format 后 EM）替代 SFT |
| ReSearch | 不必要。纯 RL 无 supervised reasoning data 即可涌现 reflection |
| ASearcher | 不必要。纯 RL 从 QwQ-32B 开始 |
| WebExplorer | 必要。SFT cold-start 加速复杂工具调用学习 |
| SearchAgent-Zero | 不必要。纯 RL 在 BrowseComp-Plus 达到 SOTA |

**综合判断**：对于已有 instruction-following 能力的 instruct 模型，SFT cold-start 非必要。对于 base model 或复杂工具组合（browse+search），SFT 有加速收敛价值。R1-Searcher 的两阶段 reward 是 SFT 的有效替代。

### 3.2 纯 RL 是否可行？

**结论：可行，但有条件。**
- 小模型（1.7B/4B）：格式学习慢，建议两阶段 reward 或少量 format SFT
- 中模型（7B/8B）：纯 RL 可行，Search-R1/ReSearch 已验证
- 大模型（32B+）：纯 RL 效果最好，ASearcher 已验证

### 3.3 Outcome Reward 是否过稀疏？

**问题确实存在**，尤其在：
- 长 horizon（5+ 轮搜索）：大部分轨迹 reward=0
- 小模型：初期格式错误率高，有效 reward signal 极少
- 复杂问题：EM=1 的概率本身很低

**缓解方案**（按有效性排序）：
1. Format reward（+0.1）提供密集信号 — 当前已实现
2. 两阶段训练（R1-Searcher）— 先学格式再学质量
3. Turn-level credit assignment — 本仓库已有框架
4. GBR reward（s3）— 相对于 naive RAG 的增益
5. Curriculum learning — turn_limit_schedule 渐进增加难度

### 3.4 多轮搜索 Credit Assignment 如何处理？

**现有方案对比**：

| 方案 | 实现 | 优点 | 缺点 |
|------|------|------|------|
| Outcome-level（当前默认） | final reward 分配到所有 token | 简单 | 好/坏搜索步骤无法区分 |
| Credit assignment（本仓库） | 异常轮 mask 前序 token | 不惩罚好的搜索 | 只处理异常情况 |
| Turn-level reward | 每轮计算 info gain | 细粒度 | 依赖 ground truth token 形式 |
| GAE | 用 value function 估计 | 理论完备 | 多轮 agent 场景 value 估计难 |

**建议**：在当前 credit assignment 基础上，增加 turn-level info gain reward 作为辅助信号。

---

## 四、研究方向与实验设计

### 方向 1：SFT Cold-Start for Search Agent ⭐ P0

**题目**：Search-SFT: 基于合成搜索轨迹的 Cold-Start 监督微调

**研究问题**：如何为 Search Agent 构造高质量 SFT 数据，使模型在 RL 前具备基本搜索调用能力？

**动机**：当前直接从 base model 开始 GRPO，模型初期大量格式错误（`tool_parser_error_count`），reward signal 极度稀疏。

**方法设计**：
1. 用强模型对 NQ/HotpotQA 生成搜索轨迹（含 `<thought>`, `<tool_call>`, `<answer>` 标签）
2. 通过 retrieval server 实际执行搜索获取真实结果
3. 过滤：只保留答案正确且搜索次数 ≤ 3 的轨迹
4. SFT 训练 1-2 epoch 作为 RL 初始化

**代码改动点**：
- 新增 `examples/search_agent_rl/generate_sft_trajectories.py`
- 复用 `verl/trainer/sft_trainer.py` + `verl/utils/dataset/multiturn_sft_dataset.py`

**资源**：8×V100-32GB，SFT 2-4h + API $50-100 | **闭环**：3-5 天

---

### 方向 2：GRPO for Tool-Use Search ⭐ P0

**题目**：SearchGRPO: 面向多轮搜索工具调用的 GRPO 变体

**研究问题**：标准 GRPO 将 reward 分配到整个 response，但搜索 agent 包含多轮交互，如何设计适合 tool-use 的 GRPO？

**方法设计**：
1. Turn-level GRPO：按 assistant turn 分组计算 advantage
2. Length-normalized reward：`reward / sqrt(num_turns)` 避免短轨迹偏好
3. Adaptive group size：简单问题 n=3，难问题 n=8

**代码改动点**：
- `verl/trainer/ppo/core_algos.py`：注册 `grpo_multiturn` advantage estimator
- `verl/utils/reward_score/search_r1_like_qa_em.py`：添加 length penalty

**资源**：8×V100-32GB，12-24h/实验 | **闭环**：2-3 天

---

### 方向 3：Abnormal Trajectory Filtering ⭐ P1

**题目**：TrajectoryFilter: 分级异常轨迹过滤策略

**研究问题**：如何利用"部分有效"的异常轨迹，而非简单丢弃？

**方法设计**：
- Level 0（丢弃）：`tool_parser_error`
- Level 1（保留前 N 轮）：`too_many_turn_count`
- Level 2（降权）：`duplicate_search_result` → reward=-0.5
- 异常轨迹作为负例而非 mask

**代码改动点**：
- `verl/experimental/agent_loop/tool_agent_loop.py`：修改终止逻辑
- `verl/utils/reward_score/search_r1_like_qa_em.py`：添加 penalty reward

**资源**：8×V100-32GB | **闭环**：1-2 天 | **改动 < 50 行**

---

### 方向 4：Fine-Grained Credit Assignment ⭐ P0

**题目**：TurnCredit: 多轮搜索 Agent 的细粒度 Credit Assignment

**研究问题**：最终 reward 应如何分配到每一轮搜索决策？

**方法设计**：
1. 每轮搜索后用 EM 检查当前信息是否足以回答
2. Information Gain reward：`turn_reward[t] = overlap(docs[t], answer) - overlap(docs[t-1], answer)`
3. 利用已有 `agent_data.turn_scores` 字段

**代码改动点**：
- `tool_agent_loop_credit_assignment.py`：计算 per-turn info gain
- `core_algos.py`：新增 `grpo_credit_assignment` estimator

**资源**：8×V100-32GB | **闭环**：3-5 天

---

### 方向 5：Multi-Query Parallel Search ⭐ P1

**题目**：ParallelSearch: 单轮多 Query 并行搜索策略学习

**研究问题**：如何学习最优 query 数量和多样性？

**方法设计**：
- 放开 `max_queries_per_tool_call` 为 3-5
- Diversity reward：惩罚语义重复 query
- Efficiency reward：更少轮数达到正确答案给 bonus

**代码改动点**：主要是配置修改 + reward 函数添加 diversity/efficiency 项

**资源**：8×V100-32GB | **闭环**：2-3 天

---

### 方向 6：Internal vs External Knowledge Routing ⭐ P1

**题目**：KnowRoute: 学习何时搜索、何时直接回答

**研究问题**：模型应如何判断问题是否需要外部搜索？

**方法设计**：
- 直接回答正确：reward=1.2（efficiency bonus）
- 搜索后正确：reward=1.0
- 直接回答错误：reward=-0.1（应该搜索但没搜索）
- 训练数据混合 50% 需搜索 + 50% 不需搜索

**代码改动点**：
- `search_r1_like_qa_em.py`：routing-aware reward
- `preprocess_*.py`：混合不同难度数据

**资源**：8×V100-32GB | **闭环**：3-4 天

---

### 方向 7：Search Result Compression / Memory ⭐ P1

**题目**：CompressSearch: 学习型搜索结果压缩

**研究问题**：如何在固定 context budget 下有效压缩历史搜索信息？

**方法设计**：
1. 自适应压缩：训练模型自己生成摘要（改进 `generate_single_summary_self`）
2. Key-fact extraction：只保留与问题相关的关键事实
3. 将压缩能力纳入 RL reward：`-0.001 * total_context_tokens`

**代码改动点**：
- `tool_agent_loop.py`：修改 `_handle_processing_tools_state` 摘要逻辑
- 新增 `verl/tools/utils/compression.py`

**资源**：8×V100-32GB | **闭环**：4-5 天

---

### 方向 8：Long-Horizon Multi-Turn Search ⭐ P1

**题目**：LongSearch: 长程多轮搜索的课程学习与奖励塑形

**研究问题**：如何训练 agent 进行 5+ 轮搜索解决复杂问题？

**方法设计**：
1. 课程学习：`turn_limit_schedule="0:2,500:3,1000:4,2000:6,3000:8"`
2. 中间 reward shaping：每轮给 partial reward
3. Hindsight relabeling：失败长轨迹截取前 N 轮作为"部分成功"

**代码改动点**：已有 `turn_limit_schedule` 支持，主要添加 intermediate reward

**资源**：8×V100-32GB（需 `max_model_len=20000+`）| **闭环**：5-7 天

---

### 方向 9：Synthetic Trajectory Generation ⭐ P1

**题目**：SynthTraj: 基于 Rejection Sampling 和 Tree Search 的合成轨迹生成

**研究问题**：如何高效生成大量高质量搜索轨迹用于 offline RL 或 SFT？

**方法设计**：
1. Best-of-N rejection sampling：生成 N 条轨迹取 reward 最高的
2. Tree search：每轮搜索后分支，选择 info gain 最大的路径
3. 用合成数据做 offline RL 或 SFT warm-start

**代码改动点**：
- 新增 `examples/search_agent_rl/generate_synthetic_trajectories.py`
- 复用 rollout infrastructure

**资源**：8×V100-32GB | **闭环**：4-5 天

---

### 方向 10：Agentic RAG Evaluation ⭐ P2

**题目**：SearchBench: 面向搜索 Agent 的多维评测框架

**研究问题**：如何全面评估搜索 Agent 的能力（不仅是最终答案正确率）？

**方法设计**：
- 搜索效率：达到正确答案的平均轮数
- Query 质量：query 与 ground truth evidence 的相关性
- 信息利用率：检索到的相关文档是否被有效利用
- 鲁棒性：面对噪声检索结果的表现

**代码改动点**：新增 `examples/search_agent_rl/evaluation/` 目录

**资源**：低（评测不需要训练）| **闭环**：2-3 天

---

### 方向 11：Offline Search Environment Simulation ⭐ P2

**题目**：SimSearch: 基于 LLM 的离线搜索环境模拟

**研究问题**：如何消除对外部搜索 API 的依赖，实现完全离线训练？

**方法设计**（参考 ZeroSearch）：
1. 用 SFT 训练 simulation LLM 模拟搜索引擎
2. Curriculum：从高质量文档逐步引入噪声
3. 对比 simulation vs real retrieval 的训练效果

**代码改动点**：
- 新增 `verl/tools/simulated_search_tool.py`
- 修改 `search_tool_config.yaml` 支持 simulation mode

**资源**：额外 4 GPU 部署 simulation LLM | **闭环**：5-7 天

---

### 方向 12：Latency-Aware / Cost-Aware Reward ⭐ P2

**题目**：CostSearch: 延迟感知的搜索 Agent 训练

**研究问题**：如何在答案质量和搜索成本之间取得平衡？

**方法设计**：
- `reward = em_score - λ * cost`
- `cost = α * num_turns + β * total_tokens + γ * retrieval_latency`
- 多目标优化：Pareto front 上的不同 trade-off

**代码改动点**：`search_r1_like_qa_em.py` 添加 cost penalty

**资源**：8×V100-32GB | **闭环**：2-3 天

---

### 方向 13：Retrieval Budget Control ⭐ P2

**题目**：BudgetSearch: 检索预算约束下的搜索策略学习

**研究问题**：给定固定检索次数预算，如何最大化答案质量？

**方法设计**：
- 在 system prompt 中告知剩余预算
- Budget-conditioned policy：不同预算下学习不同策略
- 超预算惩罚：`reward -= 1.0 if num_searches > budget`

**代码改动点**：修改 system prompt + reward 函数

**资源**：8×V100-32GB | **闭环**：2-3 天

---

### 方向 14：Hard Negative Trajectory Construction ⭐ P1

**题目**：HardNeg: 困难负例搜索轨迹构造

**研究问题**：如何构造"看起来合理但实际错误"的搜索轨迹作为训练信号？

**方法设计**：
1. 用模型生成答案错误但格式正确的轨迹
2. 构造 misleading 检索结果（包含干扰信息）
3. 将 hard negative 轨迹的 reward 设为负值

**代码改动点**：数据预处理 + reward 函数

**资源**：8×V100-32GB | **闭环**：3-4 天

---

### 方向 15：Self-Correction After Failed Search ⭐ P1

**题目**：SearchRetry: 搜索失败后的自我纠正策略

**研究问题**：当搜索结果无关或矛盾时，agent 应如何调整策略？

**方法设计**：
1. 检测搜索失败：结果与 query 相关性低于阈值
2. 允许 query reformulation：重新表述 query 再搜索
3. Self-correction reward：纠正后正确给 bonus

**代码改动点**：
- `tool_agent_loop.py`：添加 relevance check
- reward 函数：添加 correction bonus

**资源**：8×V100-32GB | **闭环**：3-4 天

---

### 方向 16：Evidence Attribution Reward ⭐ P2

**题目**：AttrSearch: 基于证据归因的搜索 Agent 训练

**研究问题**：如何训练 agent 在回答时明确引用支持证据？

**方法设计**：
- 要求 `<answer>` 中包含 evidence span
- Attribution reward：答案中引用的 span 确实出现在检索文档中
- Faithfulness reward：答案与引用 span 一致

**代码改动点**：reward 函数添加 attribution check

**资源**：8×V100-32GB | **闭环**：3-4 天

---

### 方向 17：Answerability Estimation ⭐ P2

**题目**：CanAnswer: 搜索 Agent 的可回答性判断

**研究问题**：agent 应如何判断当前信息是否足以回答问题？

**方法设计**：
- 允许输出 "unanswerable" 标签
- Reward：正确判断不可回答 = 0.5，错误判断 = -0.5
- 混合可回答/不可回答问题训练

**代码改动点**：reward 函数 + 数据预处理

**资源**：8×V100-32GB | **闭环**：2-3 天

---

### 方向 18：Query Decomposition ⭐ P1

**题目**：DecompSearch: 复杂问题的查询分解策略学习

**研究问题**：如何训练 agent 将复杂多跳问题分解为子查询序列？

**方法设计**：
1. 鼓励 `<thought>` 中显式分解问题
2. 每个子查询对应一轮搜索
3. Decomposition reward：子查询覆盖所有 reasoning hop

**代码改动点**：reward 函数添加 decomposition quality 评估

**资源**：8×V100-32GB | **闭环**：3-4 天

---

### 方向 19：Search Stopping Policy ⭐ P1

**题目**：StopSearch: 最优搜索停止策略学习

**研究问题**：agent 应在何时停止搜索并给出答案？

**方法设计**：
1. 过早停止惩罚：信息不足时停止 → reward penalty
2. 过晚停止惩罚：已有足够信息仍搜索 → efficiency penalty
3. Confidence-based stopping：模型输出 confidence score

**代码改动点**：reward 函数 + system prompt 修改

**资源**：8×V100-32GB | **闭环**：2-3 天

---

### 方向 20：Retrieval-Noise Robustness ⭐ P2

**题目**：RobustSearch: 检索噪声鲁棒的搜索 Agent

**研究问题**：如何训练 agent 在检索结果包含错误/无关信息时仍能正确推理？

**方法设计**：
1. 训练时注入噪声：随机替换部分检索结果为无关文档
2. Adversarial retrieval：故意返回包含错误答案的文档
3. Noise-robust reward：在噪声环境下正确回答给 bonus

**代码改动点**：
- `search_tool.py`：添加 noise injection 模式
- reward 函数：noise-conditioned bonus

**资源**：8×V100-32GB | **闭环**：3-4 天

---

## 五、优先级总览

### P0（最高优先级，应最先完成）

| # | 方向 | 核心价值 | 闭环时间 |
|---|------|----------|----------|
| 1 | SFT Cold-Start | 所有后续 RL 实验的基础 | 3-5 天 |
| 2 | GRPO for Tool-Use | 直接改进当前训练效果 | 2-3 天 |
| 4 | Fine-Grained Credit Assignment | 核心研究贡献点 | 3-5 天 |

### P1（高优先级，P0 完成后推进）

| # | 方向 | 核心价值 | 闭环时间 |
|---|------|----------|----------|
| 3 | Abnormal Trajectory Filtering | 低成本高收益工程优化 | 1-2 天 |
| 5 | Multi-Query Parallel Search | 配置修改即可验证 | 2-3 天 |
| 6 | Knowledge Routing | RAG 热门话题 | 3-4 天 |
| 7 | Search Result Compression | 实用问题 | 4-5 天 |
| 8 | Long-Horizon Search | 前沿方向 | 5-7 天 |
| 9 | Synthetic Trajectory | 数据扩增 | 4-5 天 |
| 14 | Hard Negative Trajectory | 训练信号增强 | 3-4 天 |
| 15 | Self-Correction | 实用能力 | 3-4 天 |
| 18 | Query Decomposition | 多跳推理核心 | 3-4 天 |
| 19 | Search Stopping Policy | 效率优化 | 2-3 天 |

### P2（中优先级，探索性方向）

| # | 方向 | 核心价值 | 闭环时间 |
|---|------|----------|----------|
| 10 | Agentic RAG Evaluation | 评测基础设施 | 2-3 天 |
| 11 | Offline Simulation | 降低训练成本 | 5-7 天 |
| 12 | Cost-Aware Reward | 部署优化 | 2-3 天 |
| 13 | Budget Control | 实用约束 | 2-3 天 |
| 16 | Evidence Attribution | 可解释性 | 3-4 天 |
| 17 | Answerability Estimation | 鲁棒性 | 2-3 天 |
| 20 | Retrieval-Noise Robustness | 鲁棒性 | 3-4 天 |

### 推荐执行路径

```
Week 1: 方向 2 (GRPO) + 方向 3 (Filtering) → 快速验证基础改进
Week 2: 方向 1 (SFT) + 方向 4 (Credit Assignment) → 核心研究贡献
Week 3: 方向 5 (Multi-Query) + 方向 19 (Stopping) → 效率优化
Week 4: 方向 8 (Long-Horizon) + 方向 6 (Routing) → 能力扩展
```

### 论文可写性排序

1. **方向 4** (Credit Assignment) — 核心问题，novelty 高
2. **方向 2** (GRPO for Tool-Use) — 热门方向，明确 novelty
3. **方向 6** (Knowledge Routing) — RAG 热门话题
4. **方向 8** (Long-Horizon) — 前沿方向
5. **方向 1** (SFT Cold-Start) — 作为完整 pipeline 一部分

---

## 附录：资源需求汇总

| 硬件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 8×V100-32GB | 8×A100-80GB |
| 检索服务 | 4 GPU (dense retriever) | 独立节点 |
| 存储 | 500GB (Wikipedia corpus + index) | 1TB |
| API 费用 | $50-100 (SFT 轨迹生成) | $200 |

---


## 附录：项目与论文链接

| 名称 | 类型 | 链接 |
|------|------|------|
| [Search-R1](https://github.com/PeterGriffinJin/Search-R1) | 论文+代码 | https://github.com/PeterGriffinJin/Search-R1 |
| [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) | 论文+代码 | https://github.com/RUCAIBox/R1-Searcher |
| [R1-Searcher++](https://arxiv.org/abs/2505.00223) | 论文 | https://arxiv.org/abs/2505.00223 |
| [ReSearch](https://github.com/Agent-RL/ReSearch) | 论文+代码 | https://github.com/Agent-RL/ReSearch |
| [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch) | 论文+代码 | https://github.com/Alibaba-NLP/ZeroSearch |
| [ASearcher](https://github.com/ASearcher-project/ASearcher) | 论文+代码 | https://github.com/ASearcher-project/ASearcher |
| [WebExplorer](https://github.com/THUDM/WebExplorer) | 论文+代码 | https://github.com/THUDM/WebExplorer |
| [s3](https://arxiv.org/abs/2505.07834) | 论文 | https://arxiv.org/abs/2505.07834 |
| [RAG-R1](https://arxiv.org/abs/2507.02962) | 论文 | https://arxiv.org/abs/2507.02962 |
| [verl](https://github.com/volcengine/verl) | 框架 | https://github.com/volcengine/verl |
| [AgentRL](https://github.com/THUDM/AgentRL) | 框架 | https://github.com/THUDM/AgentRL |
| [Forge](https://huggingface.co/blog/MiniMaxAI/forge) | 博客 | https://huggingface.co/blog/MiniMaxAI/forge |
| [SearchAgent-Zero](https://github.com/SearchAgent-Zero/SearchAgent-Zero) | 代码 | https://github.com/SearchAgent-Zero/SearchAgent-Zero |
| [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) | 论文 | https://arxiv.org/abs/2402.03300 |
| [DAPO](https://arxiv.org/abs/2503.14476) | 论文 | https://arxiv.org/abs/2503.14476 |
| [BrowseComp](https://github.com/openai/browsecomp) | Benchmark | https://github.com/openai/browsecomp |
| [FRAMES](https://arxiv.org/abs/2409.12941) | Benchmark | https://arxiv.org/abs/2409.12941 |
| [HLE](https://lastexam.ai/) | Benchmark | https://lastexam.ai/ |
| [HotpotQA](https://hotpotqa.github.io/) | Benchmark | https://hotpotqa.github.io/ |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | Benchmark | https://github.com/Alab-NII/2wikimultihop |
| [Musique](https://github.com/StonyBrookNLP/musique) | Benchmark | https://github.com/StonyBrookNLP/musique |

---

*本报告由 repo-method-analyst、rl-sft-paper-scout、direction-experiment-designer 三个 agent 并行生成，主会话综合整理。*
