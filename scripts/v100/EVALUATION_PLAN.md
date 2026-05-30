# SearchAgent-Zero 评测计划

> 本文档描述 Search-R1 配方训练完成后的完整评测流程、数据集来源、目标分数与评测脚本使用方法。

---

## 一、评测数据集

### 1.1 数据来源

所有评测数据集统一来自 **FlashRAG** 数据集合集：

```
https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
```

加载方式：
```python
import datasets
ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "<子集名>")
```

> 评测脚本会自动加载，通常无需手动下载。

### 1.2 数据集详情

| # | 数据集 | HuggingFace 子集名 | 类型 | Domain | 难度 |
|---|--------|-------------------|------|--------|------|
| 1 | NQ (Natural Questions) | `nq` | 单跳问答 | In-domain† | ⭐⭐ |
| 2 | TriviaQA | `triviaqa` | 单跳问答 | Out-of-domain* | ⭐⭐ |
| 3 | PopQA | `popqa` | 长尾知识 | Out-of-domain* | ⭐⭐⭐ |
| 4 | HotpotQA | `hotpotqa` | 多跳推理 | In-domain† | ⭐⭐⭐ |
| 5 | 2WikiMultihopQA | `2wikimultihopqa` | 多跳推理 | Out-of-domain* | ⭐⭐⭐ |
| 6 | MuSiQue | `musique` | 多跳推理 | Out-of-domain* | ⭐⭐⭐⭐ |
| 7 | Bamboogle | `bamboogle` | 组合推理 | Out-of-domain* | ⭐⭐⭐ |

> † In-domain：训练数据 (`PeterJinGo/nq_hotpotqa_train`) 包含同源训练集
> \* Out-of-domain：仅用于评测，训练中未见过

### 1.3 相关链接

| 资源 | 链接 |
|------|------|
| 评测数据集（FlashRAG） | https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets |
| 训练数据集（NQ + HotpotQA） | https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train |
| Search-R1 原始项目 | https://github.com/PeterGriffinJin/Search-R1 |
| Search-R1 数据处理脚本 | https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/qa_search_test_merge.py |
| SearchAgent-Zero 技术报告 | https://zhuanlan.zhihu.com/p/2042036895199278392 |
| SearchAgent-Zero README（中文） | https://github.com/NLPJCL/SearchAgent-Zero/blob/main/README_zh.md |

---

## 二、评测目标

### 2.1 基线与目标对比

| 数据集 | Search-R1 基线 (Qwen2.5-3B-Instruct) | verl AgentLoop 目标 (Qwen2.5-3B-Instruct) | Abs. Gain | Rel. Gain |
|--------|:-------------------------------------:|:-----------------------------------------:|:---------:|:---------:|
| NQ | 0.341 | 0.4640 | +0.1230 | +36.1% |
| TriviaQA* | 0.545 | 0.6164 | +0.0714 | +13.1% |
| PopQA* | 0.378 | 0.4239 | +0.0459 | +12.1% |
| HotpotQA† | 0.324 | 0.4225 | +0.0985 | +30.4% |
| 2Wiki* | 0.319 | 0.3979 | +0.0789 | +24.7% |
| Musique* | 0.103 | 0.1808 | +0.0778 | +75.5% |
| Bamboogle* | 0.264 | 0.3440 | +0.0800 | +30.3% |
| **Avg** | **0.325** | **0.4071** | **+0.0821** | **+25.3%** |

### 2.2 评测通过标准

| 等级 | 平均分 | 附加条件 |
|------|--------|---------|
| ❌ 不合格 | < 0.35 | — |
| ⚠️ 基本合格 | ≥ 0.35 | 至少 5/7 数据集正向提升 |
| ✅ 合格 | ≥ 0.38 | 所有数据集正向提升 |
| 🌟 优秀 | ≥ 0.40 | 所有数据集正向提升，Musique ≥ 0.14 |
| 🏆 超预期 | ≥ 0.42 | 接近或超过目标分数 |

---

## 三、评测流程

### 3.1 阶段一：训练中自动验证

训练脚本每 `test_freq` 步（默认 50 步）自动在验证集上评估。

**关注指标**：
- `reward/mean`：应持续上升
- 验证集包含 NQ + HotpotQA 的 test split（每个 data_source 最多 500 条）

**无需额外操作**，训练日志和 WandB 会自动记录。

### 3.2 阶段二：中间 checkpoint 评测（可选）

在关键训练节点（step 250、step 500）导出模型并评测：

```bash
conda activate verl-v100

# 修改模型路径指向中间 checkpoint
export EVAL_MODEL_PATH="./output/qwen2.5-3b-instruct_searchr1/global_step_250/merged_hf_model"

# 运行评测（需要修改 eval 脚本中的 model.path）
bash run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh
```

**目的**：观察各数据集分数随训练步数的变化趋势，判断是否需要提前停止或调整超参。

### 3.3 阶段三：最终模型全量评测

训练完成后（默认 500 步 / 2 epoch），运行完整评测：

```bash
conda activate verl-v100

# 确保检索服务运行中
curl -s http://127.0.0.1:8000/retrieve | head -c 100

# 运行评测脚本
bash run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh
```

评测结果会输出到：
- 终端日志：`./logs/qwen2.5-3b-instruct_searchr1_eval.log`
- WandB：项目 `search_r1_like_async_rl`，实验名 `qwen2.5-3b-instruct_searchr1_eval`
- Rollout 数据：`./rollout_data/qwen2.5-3b-instruct_searchr1_eval/`

### 3.4 结果分析

评测完成后，检查各 data_source 的 reward/mean：

```bash
# 查看评测日志中的分数
grep "data_source\|reward" logs/qwen2.5-3b-instruct_searchr1_eval.log | tail -30
```

对照 §2.1 的目标表格，逐数据集确认：
1. 所有数据集是否均取得正向提升（相比 Search-R1 基线）
2. 平均分是否达到 ≥ 0.38（合格线）
3. Out-of-domain 数据集（标 * 的）泛化能力如何

---

## 四、评测脚本说明

### 4.1 主评测脚本

```
run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh
```

关键参数：
| 参数 | 值 | 说明 |
|------|---|------|
| `trainer.val_only` | True | 仅评测，不训练 |
| `trainer.val_before_train` | True | 启动后立即评测 |
| `actor_rollout_ref.model.path` | `./output/.../merged_hf_model` | 待评测模型路径 |
| `data.val_files` | `test_search_r1.parquet` | 评测数据（包含 7 个数据集） |
| `rollout.n` | 5 | 每个 prompt 采样 5 条轨迹 |
| `rollout.temperature` | 0.7 | 采样温度 |

### 4.2 评测数据预处理

评测数据已包含在 `preprocess_search_r1_dataset_new.py` 的 test split 中：

```bash
# 预处理会自动下载并处理 test split
python examples/search_agent_rl/preprocess_search_r1_dataset_new.py \
    --hf_repo_id PeterJinGo/nq_hotpotqa_train \
    --local_dir examples/search_agent_rl/search_r1_processed
```

生成的 `test_search_r1.parquet` 包含所有 7 个评测数据集的测试样本（每个 data_source 最多 500 条）。

---

## 五、常见问题

### Q: 评测分数比预期低怎么办？

1. 确认检索服务正常运行（`curl http://127.0.0.1:8000/retrieve`）
2. 确认使用的是训练完成的 checkpoint（不是初始模型）
3. 检查 `tool_call_success_rate` 是否 > 0.8
4. 如果 Musique 分数特别低，这是正常的（最难的多跳推理数据集）

### Q: 如何只评测部分数据集？

修改 `test_search_r1.parquet` 中的 `data_source` 过滤条件，或在预处理时指定子集。

### Q: 评测需要多长时间？

在 8×V100 上，完整评测（7 个数据集，每个最多 500 条）约需 1-2 小时。

---

## 六、参考

- SearchAgent-Zero README（中文）：https://github.com/NLPJCL/SearchAgent-Zero/blob/main/README_zh.md
- Search-R1 论文：https://arxiv.org/abs/2503.09516
- FlashRAG 数据集：https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
- verl 官方文档：https://verl.readthedocs.io
