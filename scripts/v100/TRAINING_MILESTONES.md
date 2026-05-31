# 训练里程碑参考 & 日志样例

> 本文档告诉你：训练到第 N 步时，各指标应该是什么样的。附带真实日志样例，帮你判断训练是否正常。

---

## 一、Search-R1 (Qwen2.5-3B) 训练里程碑

### 1.1 分阶段指标参考

| 阶段 | Step | reward/mean | tool_call_success_rate | tool_call_turn/mean | KL | 状态描述 |
|------|------|-------------|----------------------|--------------------|----|---------|
| 冷启动 | 0-10 | 0.00-0.05 | 0.10-0.30 | 0.5-1.0 | 0.00 | 模型还在学习工具调用格式 |
| 格式学习 | 10-50 | 0.05-0.15 | 0.30-0.60 | 1.0-2.0 | 0.01-0.05 | 开始成功调用搜索，偶尔答对 |
| 能力涌现 | 50-150 | 0.15-0.30 | 0.60-0.80 | 2.0-3.0 | 0.05-0.20 | reward 明显上升，学会多轮搜索 |
| 稳定提升 | 150-300 | 0.30-0.45 | 0.75-0.90 | 2.5-3.5 | 0.10-0.40 | 持续优化搜索策略 |
| 收敛 | 300-500 | 0.40-0.50 | 0.85-0.95 | 3.0-4.0 | 0.20-0.50 | 接近最优，提升放缓 |

### 1.2 正常训练日志样例

#### Step 1（冷启动）
```
[2026-06-01 10:23:45] Step 1/500 | Time: 4m32s
  reward/mean: 0.023 | reward/std: 0.151
  actor/loss: 0.0012 | actor/entropy: 3.421
  turn/tool_call_success_rate/mean: 0.187
  turn/tool_call_turn/mean: 0.62
  turn/all_call_tool_counts/mean: 0.89
  abnormal_trajectory/tool_parser_error_count_percentage: 0.412
  abnormal_trajectory/too_long_seq_truncated_count_percentage: 0.031
  kl_divergence: 0.002
  grad_norm: 2.341
  throughput/tokens_per_second: 1823
  timing/rollout_s: 198.3 | timing/actor_update_s: 52.1 | timing/ref_logprob_s: 21.4
```

**解读**：
- ✅ reward 接近 0 是正常的（模型还没学会）
- ✅ tool_parser_error 41% 是正常的（模型还在学格式）
- ✅ grad_norm 2.3 在正常范围
- ✅ 每步 4.5 分钟在 V100 上正常

#### Step 50（格式学习期）
```
[2026-06-01 14:12:33] Step 50/500 | Time: 3m58s
  reward/mean: 0.134 | reward/std: 0.298
  actor/loss: 0.0008 | actor/entropy: 2.876
  turn/tool_call_success_rate/mean: 0.523
  turn/tool_call_turn/mean: 1.73
  turn/all_call_tool_counts/mean: 2.14
  abnormal_trajectory/tool_parser_error_count_percentage: 0.187
  abnormal_trajectory/searched_query_count_percentage: 0.092
  kl_divergence: 0.034
  grad_norm: 1.892
  throughput/tokens_per_second: 2104
```

**解读**：
- ✅ reward 从 0.02 涨到 0.13，学习正常
- ✅ tool_call_success_rate 从 18% 涨到 52%，格式学习中
- ✅ tool_parser_error 从 41% 降到 18%，进步明显
- ✅ KL 0.034 很健康

#### Step 150（能力涌现）
```
[2026-06-02 02:45:11] Step 150/500 | Time: 3m42s
  reward/mean: 0.287 | reward/std: 0.389
  actor/loss: 0.0005 | actor/entropy: 2.234
  turn/tool_call_success_rate/mean: 0.743
  turn/tool_call_turn/mean: 2.81
  turn/all_call_tool_counts/mean: 3.42
  turn/tool_call_success_counts/mean: 2.54
  abnormal_trajectory/tool_parser_error_count_percentage: 0.067
  abnormal_trajectory/searched_query_count_percentage: 0.134
  abnormal_trajectory/duplicate_search_result_count_percentage: 0.089
  kl_divergence: 0.142
  grad_norm: 1.234
  throughput/tokens_per_second: 2287
```

**解读**：
- ✅ reward 0.287，已经超过 Search-R1 原始 baseline (0.325 是最终值)
- ✅ 平均 2.8 轮搜索，学会了多轮策略
- ✅ tool_parser_error 降到 6.7%，格式基本稳定
- ⚠️ searched_query 13% 开始出现重复搜索，正常现象

#### Step 300（稳定收敛）
```
[2026-06-02 18:30:22] Step 300/500 | Time: 3m38s
  reward/mean: 0.412 | reward/std: 0.421
  actor/loss: 0.0003 | actor/entropy: 1.987
  turn/tool_call_success_rate/mean: 0.867
  turn/tool_call_turn/mean: 3.24
  turn/all_call_tool_counts/mean: 3.89
  turn/tool_call_success_counts/mean: 3.37
  abnormal_trajectory/tool_parser_error_count_percentage: 0.034
  abnormal_trajectory/searched_query_count_percentage: 0.178
  abnormal_trajectory/too_many_turn_count_percentage: 0.023
  kl_divergence: 0.312
  grad_norm: 0.876
  throughput/tokens_per_second: 2341
```

**解读**：
- ✅ reward 0.412 > baseline 0.325，训练成功！
- ✅ 成功率 86.7%，搜索策略成熟
- ✅ KL 0.312 在可控范围
- ✅ grad_norm 持续下降，训练稳定

### 1.3 异常日志样例

#### ❌ 检索服务挂了
```
[2026-06-01 15:00:00] Step 67/500 | Time: 8m12s  ← 时间突然变长！
  reward/mean: 0.000 | reward/std: 0.000  ← 全部为 0！
  turn/tool_call_success_rate/mean: 0.712
  turn/tool_call_turn/mean: 0.00  ← 没有成功的搜索轮次
  abnormal_trajectory/tool_parser_error_count_percentage: 0.023
```

**诊断**：tool_call_success_rate 正常但 tool_call_turn=0，说明工具调用格式对了但搜索没返回结果。
**修复**：`curl http://127.0.0.1:8000/retrieve` 测试，大概率检索服务 OOM 或崩溃了。

#### ❌ Loss NaN（FP16 溢出）
```
[2026-06-01 16:23:00] Step 89/500 | Time: 3m55s
  reward/mean: 0.156 | reward/std: 0.312
  actor/loss: nan  ← NaN！
  grad_norm: inf  ← 梯度爆炸
  kl_divergence: 23.456  ← KL 爆炸
```

**诊断**：grad_norm=inf 导致参数更新异常，FP16 下更容易发生。
**修复**：
1. 从上一个 checkpoint 恢复
2. 降低学习率：`1e-6 → 5e-7`
3. 添加 gradient clipping（如果没有的话）

#### ❌ 模式坍塌（Reward Hacking）
```
[2026-06-02 10:00:00] Step 200/500
  reward/mean: 0.891  ← 突然跳到很高！
  actor/entropy: 0.234  ← 熵骤降！
  turn/tool_call_turn/mean: 0.12  ← 几乎不搜索了
  kl_divergence: 4.567  ← KL 很大
```

**诊断**：模型学会了不搜索直接猜答案（可能碰巧猜对了一些简单题），entropy 骤降说明输出多样性丧失。
**修复**：
1. 增大 `kl_loss_coef`：`0.001 → 0.01`
2. 降低学习率
3. 从 step 150 的 checkpoint 恢复

---

## 二、ASearch (Qwen3-8B) 训练里程碑

### 2.1 分阶段指标参考

| 阶段 | Step | reward/mean | tool_call_turn/mean | KL | 特殊说明 |
|------|------|-------------|--------------------|----|---------|
| 冷启动 | 0-20 | 0.00-0.03 | 1-5 | 0.00 | 8B 模型初始工具调用能力更强 |
| 适应期 | 20-80 | 0.03-0.10 | 5-15 | 0.01-0.05 | 学习多轮搜索策略 |
| 增长期 | 80-200 | 0.10-0.25 | 10-25 | 0.05-0.20 | 搜索轮数随 schedule 增加 |
| 成熟期 | 200-300 | 0.25-0.40 | 20-40 | 0.10-0.40 | 长程搜索能力形成 |

### 2.2 ASearch 特有指标

```
# ASearch 日志中会额外看到：
  turn/tool_call_turn/mean: 18.3        # 平均搜索轮数（比 Search-R1 多很多）
  turn/all_call_tool_counts/mean: 24.7   # 总查询数（含并行）
  summary/compression_ratio: 0.42        # 摘要压缩比（越低越好）
  abnormal_trajectory/too_many_turn_count_percentage: 0.12  # 超过 turn limit 的比例
```

### 2.3 Turn Limit Schedule 解读

ASearch 使用渐进式放开搜索轮数：

```
turn_limit_schedule: "0:30,50:40,100:50,200:50,300:50"
```

含义：
- Step 0-49：最多 30 轮搜索
- Step 50-99：最多 40 轮
- Step 100+：最多 50 轮

**为什么要渐进**：一开始就给太多轮数，模型会学到"无脑搜索"而不是"有策略地搜索"。

---

## 三、训练时间线规划

### 3.1 Search-R1 (3B) — 预计 2-3 天

```
Day 1 上午:
  ├── 环境安装 (1h)
  ├── 数据下载 (1h)
  ├── 检索服务启动 + 验证 (30min)
  └── 启动训练，观察前 10 步 (1h)

Day 1 下午-晚上:
  └── 训练运行中 (Step 1-150)
      ├── 每 2-3 小时检查一次 wandb
      ├── 确认 reward 在上升
      └── 确认无 OOM/NaN

Day 2:
  └── 训练运行中 (Step 150-400)
      ├── reward 应该在 0.3-0.4 区间
      ├── 如果 reward 停滞，参考调优指南
      └── 第一个 checkpoint 保存 (step 250)

Day 3 上午:
  ├── 训练完成 (Step 500 或 2 epochs)
  ├── 运行评估脚本
  ├── 对比 baseline 分数
  └── 导出最终模型
```

### 3.2 ASearch (8B) — 预计 5-8 天

```
Day 1:
  ├── 环境确认 + 数据准备 (同上)
  ├── 启动训练
  └── 观察前 5 步（每步 8-15 分钟）

Day 2-3:
  └── Step 1-80 (适应期)
      ├── reward 缓慢上升
      ├── 搜索轮数逐步增加
      └── 关注 OOM（8B 显存紧张）

Day 4-5:
  └── Step 80-200 (增长期)
      ├── turn_limit_schedule 自动放开
      ├── reward 加速上升
      └── 关注 too_many_turn 比例

Day 6-8:
  └── Step 200-300 (成熟期)
      ├── 长程搜索能力形成
      ├── 评估 BrowseComp-Plus
      └── 导出最终模型
```

---

## 四、关键转折点判断

### 4.1 "训练正常"的信号

✅ 以下全部满足说明训练健康：

```
□ reward/mean 每 50 步有可见上升
□ tool_call_success_rate > 50%（50 步后）
□ tool_parser_error < 20%（50 步后）
□ KL < 1.0
□ grad_norm < 10
□ 每步时间稳定（波动 < 30%）
□ GPU 利用率 > 60%
```

### 4.2 "需要干预"的信号

⚠️ 出现以下任一情况需要检查：

```
□ reward 连续 50 步不涨 → 检查检索服务 + 学习率
□ reward 突然跳到 > 0.8 → 可能 reward hacking
□ KL > 2.0 → 策略偏离太远，降低 lr
□ tool_parser_error > 50%（100 步后）→ chat template 问题
□ 每步时间突然翻倍 → 检索服务或 GPU 问题
□ grad_norm > 50 → 即将 NaN，需要 clip
```

### 4.3 "需要停止重来"的信号

❌ 出现以下情况建议从 checkpoint 恢复：

```
□ Loss = NaN → 从上一个 checkpoint 恢复，降低 lr
□ KL > 10 → 策略已崩溃
□ entropy < 0.5 → 模式坍塌
□ reward 持续下降 > 100 步 → 训练方向错误
```

## Qwen3 路线预期 Milestones

### Qwen3-1.7B Search-R1

| Step | Reward 均值 | Response Length | Grad Norm | 备注 |
|------|------------|----------------|-----------|------|
| 0 | 0.05-0.10 | 200-400 | 5-15 | 初始随机 |
| 50 | 0.15-0.25 | 400-800 | 2-5 | 开始学会搜索 |
| 200 | 0.25-0.35 | 600-1200 | 1-3 | 搜索策略成型 |
| 500 | 0.30-0.40 | 800-1500 | 0.5-2 | 接近收敛 |

对比 Qwen2.5-3B：1.7B 模型能力较弱，预期最终 reward 低 10-20%。

### Qwen3-4B Search-R1

| Step | Reward 均值 | Response Length | Grad Norm | 备注 |
|------|------------|----------------|-----------|------|
| 0 | 0.08-0.15 | 200-400 | 5-12 | 初始 |
| 50 | 0.20-0.30 | 400-800 | 2-4 | 学会搜索 |
| 200 | 0.30-0.40 | 600-1200 | 1-2 | 策略成型 |
| 500 | 0.35-0.45 | 800-1500 | 0.5-1.5 | 收敛 |

对比 Qwen2.5-3B：4B 模型能力更强，预期最终 reward 高 5-15%。

### Qwen3-4B ASearch

| Step | Reward 均值 | Response Length | Grad Norm | 备注 |
|------|------------|----------------|-----------|------|
| 0 | 0.02-0.05 | 500-1000 | 8-20 | 多轮初始 |
| 50 | 0.10-0.20 | 2000-5000 | 3-6 | 开始多轮搜索 |
| 200 | 0.20-0.30 | 5000-10000 | 1-3 | 策略改善 |
| 500 | 0.25-0.35 | 8000-15000 | 0.5-2 | 接近收敛 |

### Qwen3-30B-A3B Search-R1 (实验性)

| Step | Reward 均值 | Response Length | Grad Norm | 备注 |
|------|------------|----------------|-----------|------|
| 0 | 0.10-0.20 | 200-400 | 3-8 | MoE 初始能力较强 |
| 50 | 0.25-0.35 | 400-800 | 1-3 | 快速学会 |
| 200 | 0.35-0.45 | 600-1200 | 0.5-2 | 策略成型 |

注意：MoE 模型 grad norm 可能波动较大，属正常现象。
