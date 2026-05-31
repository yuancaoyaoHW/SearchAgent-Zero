# Qwen3 架构训练计划设计文档

**日期：** 2026-05-31
**状态：** 待实现
**范围：** FSDP attention patch、thinking mode 配置、V100 训练脚本矩阵、文档更新

---

## 1. 背景

SearchAgent-Zero 当前训练计划支持两条路线：

- Qwen2.5-3B Search-R1（入门）
- Qwen3-8B ASearch（进阶）

但 Qwen3 架构在代码层面缺乏完整支持：

- 无 FSDP 序列并行 attention patch（`verl/models/transformers/` 下无 `qwen3.py`）
- Legacy model registry 未注册 Qwen3
- Thinking mode 无配置机制
- 缺少多尺寸训练脚本和架构文档

本设计补全以上所有缺口，建立完整的 Qwen3 训练路线矩阵。

---

## 2. Qwen3 vs Qwen2.5 架构差异

| 特性 | Qwen2.5 | Qwen3 | 训练影响 |
|------|---------|-------|---------|
| QK LayerNorm | ❌ | ✅ per-head RMSNorm | attention patch 需额外处理 |
| Sliding Window | 全层固定窗口 | 交替层（全局+局部） | patch 逻辑更复杂 |
| Thinking Mode | ❌ | ✅ `<think>...</think>` | 需配置机制、reward strip |
| Tool Call 格式 | Hermes | Hermes | 无差异 |
| Tie Word Embeddings | ✅ (小模型) | ❌ (全部 untied) | 显存多占一份 embed |
| MoE 变体 | 无官方 MoE | 30B-A3B / 235B-A22B | 新增 MoE 路线 |

### 同尺寸对比

| 尺寸 | Qwen2.5 | Qwen3 | 差异 |
|------|---------|-------|------|
| ~1.5-1.7B | Qwen2.5-1.5B (28层, 1536d) | Qwen3-1.7B (28层, 2048d) | Qwen3 hidden dim 更大 |
| ~3-4B | Qwen2.5-3B (36层, 2048d) | Qwen3-4B (36层, 2560d) | Qwen3 参数多 ~30% |
| ~7-8B | Qwen2.5-7B (28层, 3584d) | Qwen3-8B (36层, 4096d) | Qwen3 更深更宽 |
| MoE | — | Qwen3-30B-A3B (128 experts, top-2) | 全新路线 |

---

## 3. Phase 1：FSDP Attention Patch

### 3.1 新增文件

**`verl/models/transformers/qwen3.py`**

基于 `qwen2.py` 的 `qwen2_flash_attn_forward` 改写：

- 在 Q/K projection 后、RoPE 前插入 per-head RMSNorm（对齐 HuggingFace `Qwen3Attention` 的 `q_norm` / `k_norm`）
- 保留 Ulysses all-to-all 通信逻辑（`ulysses_attn_forward`）
- Sliding window 参数从 `layer_idx` 和 config 中读取：偶数层全局注意力、奇数层局部窗口
- 支持 SDPA 和 FlashAttention 两种 backend（V100 用 SDPA，A100+ 用 FA2）

### 3.2 注册

**`verl/models/transformers/__init__.py`：**

在 patch 映射表中添加 `Qwen3ForCausalLM` → `qwen3_flash_attn_forward`

**`verl/models/registry.py`：**

添加 `Qwen3ForCausalLM` 到 legacy registry

### 3.3 验证

- 用 Qwen3-1.7B 跑 1 step forward + backward
- 对比 SP=1 和 SP>1 的 logits 一致性（误差 < 1e-5）

---

## 4. Phase 2：Thinking Mode 配置机制

### 4.1 配置结构

```yaml
# search_multiturn_grpo.yaml
thinking:
  enabled: false                # 默认关闭
  max_thinking_tokens: 512     # 限制 thinking 长度
  strip_at_inference: true     # 推理时去掉 <think> 块

data:
  apply_chat_template_kwargs:
    enable_thinking: ${thinking.enabled}
```

### 4.2 脚本级覆盖

```bash
# 小模型 (1.7B/4B) Search-R1 — 可开启
+thinking.enabled=true +thinking.max_thinking_tokens=256

# 8B ASearch — 关闭
+thinking.enabled=false

# MoE — 关闭
+thinking.enabled=false
```

### 4.3 序列长度联动

开启 thinking 时自动调整：

```
effective_max_response = base_max_response + thinking.max_thinking_tokens
```

在 rollout config 中体现，确保 vLLM `max_tokens` 和 trainer sequence padding 一致。

### 4.4 Reward 处理

- Thinking tokens 不参与 reward 计算
- Reward function 中 strip `<think>...</think>` 后再计算 F1/EM
- Credit assignment 只分配到 tool call 和 final answer tokens

### 4.5 按路线配置策略

| 模型 | 路线 | Thinking | 理由 |
|------|------|----------|------|
| Qwen3-1.7B | Search-R1 | ✅ 可选 | 序列短，显存充裕，可做对比实验 |
| Qwen3-4B | Search-R1 | ✅ 可选 | 同上 |
| Qwen3-4B | ASearch | ❌ | 多轮对话序列长，开启会 OOM |
| Qwen3-8B | ASearch | ❌ | TP=2 已接近 32GB 上限 |
| Qwen3-30B-A3B | Search-R1 | ❌ | MoE KV cache 开销大 |

---

## 5. Phase 3：训练脚本矩阵

### 5.1 完整路线图

```
Qwen2.5 路线（保留）：
  └─ Qwen2.5-3B Search-R1（入门，2-3 天）

Qwen3 路线（新增）：
  ├─ Qwen3-1.7B Search-R1（快速验证，1-2 天）
  ├─ Qwen3-4B Search-R1（中等，2-3 天）
  ├─ Qwen3-4B ASearch（进阶，3-5 天）
  ├─ Qwen3-8B ASearch（高级，5-8 天）
  └─ Qwen3-30B-A3B Search-R1（实验性，3-5 天）
```

### 5.2 脚本详情

| 脚本 | 模型 | 路线 | TP | Thinking | 显存/GPU | 时间 |
|------|------|------|-----|----------|---------|------|
| `run_qwen3_1.7b_search_r1_v100.sh` | Qwen3-1.7B | Search-R1 | 1 | ✅ 可选 | ~10-12 GB | 1-2 天 |
| `run_qwen3_4b_search_r1_v100.sh` | Qwen3-4B | Search-R1 | 1 | ✅ 可选 | ~16-18 GB | 2-3 天 |
| `run_qwen3_4b_asearch_v100.sh` | Qwen3-4B | ASearch | 1 | ❌ | ~20-24 GB | 3-5 天 |
| `run_qwen3_8b_asearch_v100.sh` | Qwen3-8B | ASearch | 2 | ❌ | ~28-30 GB | 5-8 天 |
| `run_qwen3_30b_a3b_search_r1_v100.sh` | Qwen3-30B-A3B | Search-R1 | 4 | ❌ | ~26-30 GB | 3-5 天 |

### 5.3 V100 适配要点

**通用：**
- FP16 dtype（V100 无 BF16）
- SDPA + xformers attention（无 FlashAttention-2）
- vLLM v0 engine
- FSDP + CPU offloading

**1.7B / 4B：**
- TP=1，单卡放得下
- 可用更大 batch size（rollout_n=8）
- Thinking 开启时 `max_response_length` += `max_thinking_tokens`

**8B：**
- TP=2 必须
- 现有 `run_asearch_v100.sh` 补上 attention patch 注册
- Thinking 关闭

**MoE 30B-A3B：**
- TP=4（总参数 30B 需要跨 4 卡加载）
- FSDP 切分 expert 权重（不用 expert parallel，V100 通信瓶颈）
- Search-R1 短对话路线（4 轮），控制 KV cache
- 标记为实验性，准备 CPU offload fallback

### 5.4 渐进训练路径

```
验证环境 → 1.7B Search-R1 (1-2天，验证 patch + thinking)
         → 4B Search-R1 (2-3天，验证中等规模)
         → 4B ASearch (3-5天，验证多轮)
         → 8B ASearch (5-8天，主力训练)
         → MoE 30B-A3B Search-R1 (实验性)
```

---

## 6. Phase 4：文档更新

### 6.1 `scripts/v100/TRAINING_MANUAL.md`

新增章节：
- "Qwen3 架构特性"（QK-LayerNorm、sliding window、thinking mode、untied embeddings、MoE）
- 更新 "训练路线" 为完整矩阵（Qwen2.5 + Qwen3 并列）

### 6.2 `scripts/v100/training-tracker/src/data/guide.ts`

Guide tab 新增：
- Qwen3 vs Qwen2.5 架构对比表
- 各模型尺寸 V100 资源需求速查
- Thinking mode 开关决策树
- MoE 特殊注意事项

### 6.3 `scripts/v100/TUNING_GUIDE.md`

新增 Qwen3 调参建议：
- QK Norm → learning rate 可稍大（训练更稳定）
- Thinking mode 开启时 `max_response_length` 调整公式
- MoE load balancing loss 权重

### 6.4 `scripts/v100/TRAINING_MILESTONES.md`

为每个新路线添加预期 milestone：
- 各 step 的 reward 均值、response length、grad norm 参考值
- 与 Qwen2.5-3B 路线的收敛速度对比

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Sliding window 交替逻辑与 Ulysses SP 冲突 | 序列并行结果不正确 | 对比 SP=1 和 SP>1 的 logits 一致性 |
| MoE 30B 在 V100 上 OOM | 无法训练 | 标记为实验性，CPU offload fallback |
| Thinking tokens 导致 RL 不稳定 | reward hacking | 限制 max_thinking_tokens + KL penalty |
| HuggingFace Qwen3 实现版本变动 | patch 失效 | 锁定 transformers 版本，注释标注依赖 |
| Qwen3-1.7B/4B 模型能力不足 | Search-R1 收敛困难 | 用 Qwen2.5-3B 作为 baseline 对比 |

---

## 8. 实现顺序

```
Phase 1 → Phase 2 → Phase 3 → Phase 4
(patch)   (config)   (scripts)  (docs)
```

每个 Phase 完成后独立验证，不依赖后续 Phase。Phase 3 内部按模型尺寸递增，每个脚本 dry-run 通过后再进入下一个。
