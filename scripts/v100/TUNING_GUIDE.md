# 超参调优策略

> 当训练不如预期时，该调什么参数、调多少、按什么顺序。

---

## 一、调优决策树

遇到问题时，按这个流程走：

```
训练出了问题
    │
    ├── Reward 不涨？ ──────────────────────────────────────┐
    │   ├── tool_call_success_rate < 30%?                    │
    │   │   └── → 检查 format/template/检索服务              │
    │   ├── tool_call_success_rate > 60% 但 reward 不涨?     │
    │   │   └── → 检查 reward 函数 / ground_truth 数据       │
    │   ├── reward/std ≈ 0?                                  │
    │   │   └── → 增大 rollout.n (3→5→8)                    │
    │   └── reward 有波动但不上升?                            │
    │       └── → 调学习率 (见下方)                          │
    │                                                        │
    ├── Loss NaN / 梯度爆炸？ ──────────────────────────────┐
    │   ├── 发生在前 10 步?                                  │
    │   │   └── → lr 太高，降到 5e-7 或 1e-7                │
    │   ├── 发生在中后期?                                    │
    │   │   └── → IS weight 爆炸，降低 rollout_is_threshold │
    │   └── 偶发性?                                          │
    │       └── → 加 grad clipping (max_grad_norm=1.0)      │
    │                                                        │
    ├── 训练太慢？ ─────────────────────────────────────────┐
    │   ├── GPU 利用率 < 30%?                                │
    │   │   └── → CPU offload 瓶颈，加内存或减少 offload     │
    │   ├── GPU 利用率 > 80% 但步速慢?                       │
    │   │   └── → 序列太长，降低 max_response_length         │
    │   └── 检索延迟 > 500ms?                                │
    │       └── → 检索服务瓶颈，增加实例或用 CPU 模式        │
    │                                                        │
    ├── OOM？ ──────────────────────────────────────────────┐
    │   ├── 启动即 OOM?                                      │
    │   │   └── → 降低 gpu_memory_utilization (0.65→0.55)   │
    │   ├── Rollout 阶段 OOM?                                │
    │   │   └── → 降低 max_model_len 或 rollout.n           │
    │   └── Actor update 阶段 OOM?                           │
    │       └── → 降低 ppo_micro_batch_size_per_gpu          │
    │                                                        │
    └── 模式坍塌 / Reward Hacking？ ────────────────────────┐
        ├── entropy 骤降?                                    │
        │   └── → 增大 kl_loss_coef (0.001→0.01)           │
        ├── reward 突然跳到 1.0?                             │
        │   └── → 检查是否有 reward 计算 bug                │
        └── KL > 5?                                          │
            └── → 降低 lr + 增大 kl_loss_coef               │
```

---

## 二、核心超参详解

### 2.1 学习率 (lr)

| 场景 | 推荐值 | 调整方向 |
|------|--------|---------|
| 3B 模型正常训练 | 1e-6 | — |
| 3B 模型 reward 不涨 | 2e-6 | ↑ 加倍 |
| 3B 模型 loss spike | 5e-7 | ↓ 减半 |
| 8B 模型正常训练 | 5e-7 | — |
| 8B 模型 reward 不涨 | 1e-6 | ↑ 加倍 |
| 8B 模型 loss spike | 2e-7 | ↓ 减半 |

**原则**：
- 模型越大，lr 越小
- FP16 比 BF16 更敏感，V100 上 lr 偏保守
- 如果 KL 增长太快（> 0.5/100步），说明 lr 太大

### 2.2 rollout.n (GRPO 组大小)

| 值 | 效果 | 代价 |
|----|------|------|
| 2 | 最快，但 GRPO 信号弱 | 可能学不动 |
| 3 | 平衡速度和质量 | — |
| 5 | 推荐值，信号充足 | 显存 ×1.7 |
| 8 | 信号最强 | 显存 ×2.7，V100 可能 OOM |

**调优规则**：
- 如果 `reward/std ≈ 0`（所有轨迹得分一样），增大 n
- 如果 OOM，减小 n
- n 越大，每步越慢，但每步学到的越多

### 2.3 train_batch_size

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 64 | 更新频繁，但方差大 | 快速实验 |
| 128 | ASearch 默认 | 8B 模型 |
| 256 | Search-R1 默认 | 3B 模型 |
| 512 | 更稳定，但每步更慢 | 大规模训练 |

**注意**：实际每步的轨迹数 = `train_batch_size × rollout.n`

### 2.4 gpu_memory_utilization

控制 vLLM 占用多少 GPU 显存用于 KV cache。

| 值 | KV cache 大小 | 留给 actor 的空间 | 适用 |
|----|--------------|-----------------|------|
| 0.45 | 小 | 大 | 8B + 长序列 |
| 0.55 | 中 | 中 | 8B 默认 |
| 0.65 | 大 | 小 | 3B 默认 |
| 0.75 | 很大 | 很小 | 只有推理没有训练时 |

**调优规则**：
- Actor update OOM → 降低此值（给训练留更多空间）
- Rollout 报 "Not enough KV cache" → 增大此值
- 两者冲突时 → 降低 `max_model_len` 或 `rollout.n`

### 2.5 max_response_length & max_model_len

```
max_model_len >= max_prompt_length + max_response_length
```

| 参数 | Search-R1 | ASearch | 作用 |
|------|-----------|---------|------|
| max_prompt_length | 4096 | 2048 | prompt 最大 token 数 |
| max_response_length | 3000 | 16384 | 回复最大 token 数 |
| max_model_len | 15000 | 12000 | vLLM 总上下文窗口 |

**调优规则**：
- OOM → 先降 max_model_len（影响最大）
- 轨迹被截断太多 → 增大 max_response_length
- 两者矛盾时 → 开启 `enable_tool_response_summary=True` 压缩搜索结果

### 2.6 kl_loss_coef

控制策略偏离参考模型的惩罚力度。

| 值 | 效果 |
|----|------|
| 0.0001 | 几乎不约束，策略自由探索 |
| 0.001 | 默认值，轻度约束 |
| 0.01 | 强约束，防止模式坍塌 |
| 0.1 | 过强，模型几乎不更新 |

**调优规则**：
- KL 增长太快（> 1.0/100步）→ 增大 kl_loss_coef
- Reward 完全不涨 → 减小 kl_loss_coef（让模型更自由探索）
- 出现模式坍塌 → 增大到 0.01-0.05

### 2.7 ppo_micro_batch_size_per_gpu

控制每次 forward/backward 处理多少样本。**不影响算法效果，只影响显存和速度**。

| 值 | 显存 | 速度 | 适用 |
|----|------|------|------|
| 2 | 最小 | 最慢 | 8B + V100 |
| 4 | 小 | 慢 | 8B 默认 |
| 8 | 中 | 中 | 3B + 长序列 |
| 16 | 大 | 快 | 3B 默认 |
| 32 | 很大 | 最快 | 短序列 |

**调优规则**：
- Actor update OOM → 减半
- 训练太慢 → 加倍（如果显存允许）

---

## 三、调优实战案例

### 案例 1：Reward 在 0.1 附近停滞不前

**现象**：Step 100 后 reward 在 0.08-0.12 之间波动，不再上升。

**排查**：
```bash
# 检查 reward 分布
grep "reward/std" logs/*.log | tail -10
# 发现 std ≈ 0.05，太小了

# 检查 tool_call 情况
grep "tool_call_success_rate" logs/*.log | tail -10
# 发现 success_rate = 0.72，正常

# 检查 KL
grep "kl_divergence" logs/*.log | tail -10
# 发现 KL = 0.008，太小了 → 模型几乎没在学习
```

**诊断**：lr 太小 + rollout.n 太小，导致 GRPO 信号不足。

**修复**：
```bash
# 从 step 100 的 checkpoint 恢复，调整参数
bash scripts/v100/run_search_r1_v100.sh \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.rollout.n=8 \
    trainer.resume_from_checkpoint=output/.../step_100
```

### 案例 2：Step 200 后突然 OOM

**现象**：前 200 步正常，突然 CUDA OOM。

**排查**：
```bash
# 检查是否是序列变长了
grep "response_length\|seq_len" logs/*.log | tail -20
# 发现平均序列长度从 1500 涨到 2800

# 原因：模型学会了多轮搜索，轨迹变长了
```

**修复**（不需要重新训练）：
```bash
# 方案 A：降低 KV cache
bash scripts/v100/run_search_r1_v100.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    trainer.resume_from_checkpoint=output/.../step_200

# 方案 B：降低 micro batch
bash scripts/v100/run_search_r1_v100.sh \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    trainer.resume_from_checkpoint=output/.../step_200
```

### 案例 3：ASearch 训练极慢（15min/step）

**现象**：每步 15 分钟，GPU 利用率只有 40%。

**排查**：
```bash
# 检查 CPU 使用率
htop
# 发现 CPU 100%，内存 240/256 GB

# 检查 swap
free -h
# 发现 swap 使用了 30GB → CPU offload 导致内存不足
```

**修复**：
```bash
# 方案 A：减少需要 offload 的量
bash scripts/v100/run_asearch_v100.sh \
    actor_rollout_ref.rollout.n=2 \
    data.train_batch_size=64

# 方案 B：只 offload optimizer，不 offload param
bash scripts/v100/run_asearch_v100.sh \
    actor_rollout_ref.actor.fsdp_config.param_offload=False
```

---

## 四、参数敏感度排序

从"影响最大"到"影响最小"：

| 排名 | 参数 | 影响维度 | 调错的后果 |
|------|------|---------|-----------|
| 1 | lr | 精度 | 太高→NaN，太低→不学习 |
| 2 | rollout.n | 精度+速度 | 太小→信号弱，太大→OOM |
| 3 | gpu_memory_utilization | 稳定性 | 太高→OOM，太低→rollout 慢 |
| 4 | max_model_len | 稳定性+精度 | 太小→截断，太大→OOM |
| 5 | kl_loss_coef | 精度 | 太小→坍塌，太大→不学习 |
| 6 | train_batch_size | 精度+速度 | 太小→不稳定，太大→慢 |
| 7 | ppo_micro_batch_size | 速度 | 只影响速度不影响精度 |
| 8 | max_response_length | 精度 | 太小→截断好轨迹 |

---

## 五、V100 特有调优建议

### 5.1 FP16 稳定性

V100 只能用 FP16（不支持 BF16），FP16 的动态范围更小，更容易溢出：

```bash
# 如果频繁出现 loss spike，加这些保护：
bash scripts/v100/run_search_r1_v100.sh \
    actor_rollout_ref.actor.max_grad_norm=1.0 \
    actor_rollout_ref.actor.optim.lr=5e-7
```

### 5.2 SDPA vs FlashAttention-2 的精度差异

SDPA 和 FA2 在数值上有微小差异（< 1e-5），但在长序列 + FP16 下可能累积：

- 如果发现 V100 上的 reward 比 A100 上低 5-10%，这是正常的
- 不需要额外调参，这是硬件精度限制

### 5.3 CPU Offload 调优

V100 上几乎必须开 offload，但 offload 的粒度可以调：

| 配置 | 显存节省 | 速度代价 | 推荐场景 |
|------|---------|---------|---------|
| optimizer_offload=True, param_offload=False | 中 | 小 | 3B 模型首选 |
| optimizer_offload=True, param_offload=True | 大 | 大 | 8B 模型必须 |
| + activation_offload=True | 最大 | 最大 | 8B + 长序列 |

```bash
# 3B 模型推荐配置（速度优先）
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.actor.fsdp_config.param_offload=False

# 8B 模型推荐配置（能跑就行）
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.model.enable_activation_offload=True
```
