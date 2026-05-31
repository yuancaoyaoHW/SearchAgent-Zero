# 性能分析与问题定位指南

> 当训练出现问题时（太慢、OOM、reward 不涨、loss 爆炸），如何系统性地定位和解决。

---

## 一、性能监控工具箱

### 1.1 实时 GPU 监控

```bash
# 每 2 秒刷新一次 GPU 状态
watch -n 2 nvidia-smi

# 更详细的监控（推荐）
nvidia-smi dmon -s pucvmet -d 5
# p=power, u=utilization, c=clock, v=violation, m=memory, e=ecc, t=temperature

# 持续记录到文件（后续分析用）
nvidia-smi dmon -s pucvmet -d 5 -f gpu_monitor.csv &
```

### 1.2 关键指标含义

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| GPU Util % | 70-95%（训练时） | < 30% 说明有瓶颈 |
| Memory Used | 25-30 GB（V100-32GB） | > 31 GB 即将 OOM |
| SM Clock | 1290-1530 MHz | 持续低于 1000 MHz 说明降频 |
| Power | 200-300W | 持续 < 100W 说明 GPU 空闲 |
| Temperature | 60-80°C | > 85°C 会触发降频 |

### 1.3 CPU 和内存监控

```bash
# CPU 使用率（offload 时 CPU 会很忙）
htop

# 内存使用（offload 需要大量 CPU 内存）
free -h
# 如果 swap 使用量大，说明内存不够，训练会极慢

# 磁盘 IO（checkpoint 保存时）
iostat -x 5
```

### 1.4 网络监控（检索服务）

```bash
# 检索服务延迟测试
time curl -s http://127.0.0.1:8000/retrieve \
  -X POST -H "Content-Type: application/json" \
  -d '{"query": "test", "topk": 3}' > /dev/null

# 正常: < 100ms
# 异常: > 500ms（检索服务过载或 faiss 索引未加载到 GPU）
```

---

## 二、训练阶段性能分析

### 2.1 训练各阶段耗时分解

一个完整的训练 step 包含：

```
┌─────────────────────────────────────────────────────────────┐
│                    一个训练 Step 的时间分解                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Rollout (生成轨迹)          ████████████████  60-70%        │
│    ├── vLLM 推理              ████████████     50%           │
│    ├── 工具调用 (HTTP)         ████             10%           │
│    └── 摘要压缩               ██               5%            │
│                                                               │
│  Actor Update (策略更新)      ██████           20-25%        │
│    ├── Forward pass           ███              10%           │
│    ├── Backward pass          ███              10%           │
│    └── Optimizer step         █                5%            │
│                                                               │
│  Ref Log Prob (参考模型)      ███              10-15%        │
│                                                               │
│  Data Transfer / Sync         █                < 5%          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 如何测量各阶段耗时

训练日志中会打印每步的时间：

```bash
# 从日志中提取时间信息
grep -E "step|time|rollout|actor|ref" logs/your_experiment.log | tail -50
```

如果日志不够详细，可以开启 profiling：

```bash
# 方法 1: py-spy 采样（不影响训练速度）
py-spy record -o profile.svg --pid $(pgrep -f "verl.trainer.main_ppo") --duration 60

# 方法 2: PyTorch Profiler（会略微降速）
# 在训练命令后加：
#   trainer.enable_profiling=True
```

### 2.3 各阶段瓶颈判断

| 现象 | 瓶颈位置 | 解决方案 |
|------|---------|---------|
| GPU 利用率低 + CPU 100% | CPU offload 瓶颈 | 减少 offload 或加大 CPU 内存 |
| GPU 利用率低 + 网络等待 | 检索服务慢 | 增加检索服务 worker 数或用 CPU 模式 |
| GPU 利用率高但步速慢 | 序列太长 / batch 太大 | 减小 max_response_length |
| 某些 GPU 100% 其他空闲 | 负载不均衡 | 检查 TP 设置 |
| 训练快但 rollout 慢 | vLLM KV cache 不够 | 增大 gpu_memory_utilization |

---

## 三、显存问题定位

### 3.1 显存使用分解

```
V100-32GB 显存分配（Search-R1 / 3B 模型）:
┌────────────────────────────────────────┐
│ vLLM KV Cache        ~8 GB  (25%)     │
│ Model Params (FSDP)  ~1 GB  (3%)      │
│ Activations (GC)     ~5 GB  (16%)     │
│ Gradients            ~1 GB  (3%)      │
│ Temp Buffers         ~3 GB  (9%)      │
│ 空闲 / 碎片          ~14 GB (44%)     │
└────────────────────────────────────────┘

V100-32GB 显存分配（ASearch / 8B 模型）:
┌────────────────────────────────────────┐
│ vLLM KV Cache        ~10 GB (31%)     │
│ Model Params (FSDP)  ~2 GB  (6%)      │
│ Activations (GC)     ~8 GB  (25%)     │
│ Gradients            ~2 GB  (6%)      │
│ Temp Buffers         ~5 GB  (16%)     │
│ 空闲 / 碎片          ~5 GB  (16%)     │  ← 很紧！
└────────────────────────────────────────┘
```

### 3.2 OOM 排查步骤

```bash
# Step 1: 确认哪张卡 OOM
grep -i "out of memory\|CUDA error\|OOM" logs/your_experiment.log

# Step 2: 查看 OOM 时的显存快照
# 在训练脚本中加环境变量：
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Step 3: 如果是 vLLM OOM，降低 KV cache
# gpu_memory_utilization: 0.65 → 0.55 → 0.45

# Step 4: 如果是 actor update OOM，降低 micro batch
# ppo_micro_batch_size_per_gpu: 16 → 8 → 4

# Step 5: 如果还是 OOM，开启更激进的 offload
# actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
# actor_rollout_ref.actor.fsdp_config.param_offload=True
```

### 3.3 显存优化手段（按效果排序）

| 手段 | 节省显存 | 速度代价 | 配置项 |
|------|---------|---------|--------|
| Gradient Checkpointing | ~40% activation | ~30% 慢 | `enable_gradient_checkpointing=True` |
| Optimizer Offload | ~60% optimizer | ~20% 慢 | `fsdp_config.optimizer_offload=True` |
| Param Offload | ~80% params | ~40% 慢 | `fsdp_config.param_offload=True` |
| Activation Offload | ~30% activation | ~15% 慢 | `enable_activation_offload=True` |
| 降低 rollout.n | 线性减少 KV cache | 降低 GRPO 质量 | `rollout.n=5→3` |
| 降低 max_model_len | 线性减少 KV cache | 截断长轨迹 | `max_model_len=15000→10000` |
| 增大 TP | 分摊 KV cache | 减少 DP 并行度 | `tensor_model_parallel_size=1→2` |

---

## 四、精度问题定位

### 4.1 Reward 不涨

**症状**：训练 100+ 步后 reward 仍然在 0 附近波动。

**排查清单**：

```bash
# 1. 检查 tool_call_success_rate
grep "tool_call_success_rate" logs/your_experiment.log | tail -20
# 如果 < 0.1，模型还没学会调用工具

# 2. 检查是否有有效的搜索结果
grep "tool_call_turn" logs/your_experiment.log | tail -20
# 如果 = 0，检索服务可能挂了

# 3. 检查 reward 分布
# 在 wandb 上看 reward/mean 和 reward/std
# 如果 std ≈ 0，说明所有轨迹得分一样（GRPO 无法学习）

# 4. 检查 KL divergence
grep "kl" logs/your_experiment.log | tail -20
# 如果 KL > 10，策略偏离太远，降低学习率
```

**常见原因和解决方案**：

| 原因 | 诊断方法 | 解决方案 |
|------|---------|---------|
| 检索服务不可用 | `curl` 测试返回错误 | 重启检索服务 |
| 模型不会生成工具调用格式 | `tool_call_success_rate ≈ 0` | 确认 chat template 正确 |
| 学习率太高 | KL > 5, reward 震荡 | lr: 1e-6 → 5e-7 |
| 学习率太低 | reward 完全不动 | lr: 1e-6 → 2e-6 |
| rollout.n 太小 | reward std ≈ 0 | n: 3 → 5 |
| 数据质量问题 | 检查 prompt 内容 | 重新预处理数据 |

### 4.2 Loss 爆炸 / NaN

**症状**：loss 突然变成 NaN 或极大值。

```bash
# 检查是否有 NaN
grep -i "nan\|inf" logs/your_experiment.log | head -20

# 检查 gradient norm
grep "grad_norm" logs/your_experiment.log | tail -20
# 正常: 0.1 - 10
# 异常: > 100 或 NaN
```

**FP16 特有问题**（V100 必须用 FP16）：

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Loss NaN | FP16 溢出 | 开启 grad clipping: `max_grad_norm=1.0` |
| 训练不稳定 | FP16 精度不够 | 降低学习率，增大 batch size |
| Reward 突然跳变 | IS weight 爆炸 | 调低 `rollout_is_threshold`: 2.0 → 1.5 |

### 4.3 异常轨迹过多

**症状**：`abnormal_trajectory` 相关指标持续 > 50%。

```bash
# 查看各类异常比例
grep "abnormal_trajectory" logs/your_experiment.log | tail -20
```

| 异常类型 | 含义 | 解决方案 |
|---------|------|---------|
| `tool_parser_error` > 30% | 模型生成的工具调用格式错误 | 检查 chat template，确认 format=hermes |
| `searched_query` > 50% | 模型重复搜索相同内容 | 正常现象，credit assignment 会处理 |
| `too_many_tool_call` > 30% | 单轮并行查询太多 | 降低 `max_queries_per_tool_call` |
| `too_long_seq_truncated` > 30% | 轨迹超长被截断 | 增大 `max_response_length` 或开启摘要 |

### 4.4 训练曲线解读

**健康的训练曲线**：
```
Reward:     缓慢上升，偶有波动
KL:         0.01 - 0.5，缓慢增长
Loss:       缓慢下降
Tool calls: 成功率从 ~30% 逐步提升到 ~80%
Turns:      平均搜索轮数逐步增加
```

**不健康的信号**：
```
❌ Reward 突然跳到 1.0 → 可能是 reward hacking
❌ KL > 5 → 策略崩溃，需要降低 lr 或增大 kl_loss_coef
❌ Tool calls 成功率持续 0% → 检索服务或 template 问题
❌ Loss 持续不降 → 学习率太低或数据问题
```

---

## 五、吞吐量优化

### 5.1 V100 上的预期吞吐

| 配方 | 每步时间 | 每步 tokens | 吞吐 (tokens/s/GPU) |
|------|---------|------------|-------------------|
| Search-R1 (3B, n=5) | 3-5 min | ~3.8M | ~1500-2500 |
| ASearch (8B, n=4) | 8-15 min | ~8M | ~800-1500 |

### 5.2 提升吞吐的方法

**低风险（不影响精度）**：

```bash
# 1. 增大 vLLM 的 max_num_seqs（并发请求数）
actor_rollout_ref.rollout.max_num_seqs=128  # 默认 256，V100 可能需要降低

# 2. 确保 TOKENIZERS_PARALLELISM=true
export TOKENIZERS_PARALLELISM=true

# 3. 增大 ulimit
ulimit -n 65535

# 4. 关闭不必要的日志
export VLLM_LOGGING_LEVEL=ERROR
```

**中风险（可能轻微影响精度）**：

```bash
# 1. 减少 rollout.n（降低 GRPO 组大小）
actor_rollout_ref.rollout.n=3  # 5 → 3，速度提升 ~40%

# 2. 减少 max_response_length
data.max_response_length=2048  # 3000 → 2048

# 3. 减少 ppo_epochs
actor_rollout_ref.actor.ppo_epochs=1  # 默认 1，不要超过 2
```

### 5.3 Profiling 实战

```bash
# 1. 用 py-spy 生成火焰图
pip install py-spy
py-spy record -o flamegraph.svg --pid $(pgrep -f main_ppo) --duration 120 --native

# 2. 用 torch.profiler（需要改代码或用 verl 内置）
# 在 wandb 上查看 trace

# 3. 用 nsight systems（最详细，但需要 root）
nsys profile --trace=cuda,nvtx --output=profile \
    python -m verl.trainer.main_ppo ...
```

---

## 六、检索服务性能调优

### 6.1 检索延迟分析

```bash
# 批量测试延迟
for i in $(seq 1 100); do
  time curl -s http://127.0.0.1:8000/retrieve \
    -X POST -H "Content-Type: application/json" \
    -d '{"query": "random query '$i'", "topk": 3}' > /dev/null
done 2>&1 | grep real | awk '{print $2}'
```

### 6.2 检索服务瓶颈

| 瓶颈 | 诊断 | 解决 |
|------|------|------|
| faiss 搜索慢 | 单次 > 50ms | 确认用了 `--faiss_gpu` |
| e5 编码慢 | 单次 > 100ms | 确认模型在 GPU 上 |
| 并发过载 | 延迟随训练增加 | 增大 `num_workers` 或部署多实例 |
| 内存不足 | 进程被 kill | 语料太大，用 mmap 模式 |

### 6.3 多实例部署（高级）

如果检索成为瓶颈，可以部署多个实例：

```bash
# 实例 1: 端口 8000
CUDA_VISIBLE_DEVICES=0 PORT=8000 bash start_retrieval_server.sh &

# 实例 2: 端口 8001（需要修改 tool_config）
CUDA_VISIBLE_DEVICES=0 PORT=8001 bash start_retrieval_server.sh &
```

然后修改 `search_tool_config.yaml` 中的 `retrieval_service_url` 为负载均衡地址。

---

## 七、Checkpoint 与恢复

### 7.1 Checkpoint 结构

```
output/your_experiment/
├── global_step_250/
│   ├── actor/                    # 策略模型权重
│   │   ├── model-00001-of-00002.safetensors
│   │   ├── model-00002-of-00002.safetensors
│   │   └── config.json
│   └── optimizer/                # 优化器状态
└── global_step_500/
    └── ...
```

### 7.2 从 Checkpoint 恢复训练

```bash
# 在训练命令后加：
bash scripts/v100/run_search_r1_v100.sh \
    trainer.default_local_dir=./output/your_experiment \
    trainer.resume_from_checkpoint=True
```

### 7.3 Checkpoint 空间管理

每个 checkpoint 大小：
- 3B 模型: ~6 GB
- 8B 模型: ~16 GB

```bash
# 只保留最近 3 个 checkpoint
find output/ -name "global_step_*" -type d | sort -V | head -n -3 | xargs rm -rf
```

---

## 八、问题速查表

| 症状 | 最可能原因 | 快速修复 |
|------|-----------|---------|
| 启动即 OOM | gpu_memory_utilization 太高 | 降到 0.55 |
| 训练 10 步后 OOM | activation 累积 | 确认 gradient_checkpointing=True |
| Reward 始终为 0 | 检索服务挂了 | `curl` 测试 + 重启 |
| Loss = NaN | FP16 溢出 | 降低 lr，加 grad clip |
| 训练极慢（>10min/step） | CPU offload + 内存不足 | 加物理内存或减少 offload |
| vLLM 报 "CUDA error" | attention backend 不兼容 | 确认 `VLLM_ATTENTION_BACKEND=XFORMERS` |
| Ray 连接超时 | 端口冲突 | `ray stop --force` + 重启 |
| 模型不调用工具 | chat template 错误 | 检查 `format=hermes` |
| 检索返回空结果 | 索引未加载 | 重启检索服务，观察启动日志 |
| Checkpoint 保存失败 | 磁盘满 | 清理旧 checkpoint |
