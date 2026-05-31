import { useState } from 'react'
import type { Symptom } from '../types'

const SYMPTOMS: Symptom[] = [
  {
    id: 'reward-stuck',
    label: 'Reward 不涨（连续 50+ 步）',
    diagnosis: '可能原因：检索服务异常、学习率过低、batch size 不够、reward 计算 bug。',
    commands: [
      'curl http://127.0.0.1:8000/retrieve -X POST -H "Content-Type: application/json" -d \'{"query": "test", "topk": 3}\'',
      'grep "reward" logs/*.log | tail -20',
      '# 尝试提高学习率: actor_rollout_ref.actor.optim.lr=2e-6',
    ],
  },
  {
    id: 'kl-high',
    label: 'KL 过高 (> 2.0)',
    diagnosis: '策略偏离参考模型太远。可能学习率过高或 batch 太小导致更新幅度过大。',
    commands: [
      '# 降低学习率',
      '# actor_rollout_ref.actor.optim.lr=5e-7',
      '# 增大 KL 惩罚',
      '# actor_rollout_ref.actor.kl_loss_coef=0.01',
    ],
  },
  {
    id: 'tool-error',
    label: 'Tool 解析错误率高 (> 30%)',
    diagnosis: 'Chat template 或工具调用格式问题。模型没学会正确的 tool_call 格式。',
    commands: [
      '# 检查 rollout 样本',
      'grep "tool_parser_error" logs/*.log | tail -10',
      '# 确认 chat template 配置正确',
      'python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(\'Qwen/Qwen2.5-3B-Instruct\'); print(t.chat_template[:200])"',
    ],
  },
  {
    id: 'oom',
    label: 'GPU OOM',
    diagnosis: '显存不足。V100-32GB 需要精细控制显存分配。',
    commands: [
      'nvidia-smi',
      '# 降低 gpu_memory_utilization: 0.60',
      '# 降低 rollout.n: 3',
      '# 降低 max_response_length: 2000',
      '# 确认 param_offload 和 optimizer_offload 已开启',
    ],
  },
  {
    id: 'loss-spike',
    label: 'Loss 突然飙升 / NaN',
    diagnosis: '梯度爆炸或数值不稳定。FP16 训练更容易出现。',
    commands: [
      'grep "grad_norm\\|loss" logs/*.log | tail -20',
      '# 从上一个 checkpoint 恢复',
      '# 降低学习率: lr=5e-7',
      '# 增大 gradient clipping: max_grad_norm=0.5',
    ],
  },
  {
    id: 'slow',
    label: '训练速度异常慢',
    diagnosis: '可能是检索服务瓶颈、GPU 利用率低、或 CPU offload 开销过大。',
    commands: [
      'nvidia-smi dmon -s pucvmet -d 5',
      '# 检查检索服务延迟',
      'time curl http://127.0.0.1:8000/retrieve -X POST -H "Content-Type: application/json" -d \'{"query": "test", "topk": 3}\'',
      '# 检查 Ray worker 状态',
      'ray status',
    ],
  },
  {
    id: 'mode-collapse',
    label: 'Entropy 骤降 / 模式坍塌',
    diagnosis: '模型输出多样性急剧下降，所有 prompt 生成相似回答。',
    commands: [
      'grep "entropy" logs/*.log | tail -20',
      '# 增大 entropy 系数',
      '# actor_rollout_ref.actor.entropy_coeff=0.01',
      '# 提高采样温度: rollout.temperature=1.0',
    ],
  },
]

export function Troubleshoot() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selected = SYMPTOMS.find((s) => s.id === selectedId)

  return (
    <section>
      <h2 className="text-xl font-semibold mb-2">🔧 问题诊断</h2>
      <p className="text-sm text-gray-500 mb-4">选择你观察到的症状，获取诊断建议和排查命令</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
        {SYMPTOMS.map((symptom) => (
          <button
            key={symptom.id}
            onClick={() => setSelectedId(symptom.id === selectedId ? null : symptom.id)}
            className={`text-left p-3 rounded-lg border transition-colors ${
              symptom.id === selectedId
                ? 'border-blue-600 bg-blue-50'
                : 'border-gray-200 bg-white hover:border-gray-300'
            }`}
          >
            <span className="text-sm font-medium">{symptom.label}</span>
          </button>
        ))}
      </div>

      {selected && (
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h3 className="font-medium text-gray-900 mb-2">诊断: {selected.label}</h3>
          <p className="text-sm text-gray-700 mb-4">{selected.diagnosis}</p>
          <h4 className="text-sm font-medium text-gray-700 mb-2">排查命令:</h4>
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-green-400 font-mono whitespace-pre-wrap">
              {selected.commands.join('\n')}
            </pre>
          </div>
        </div>
      )}
    </section>
  )
}
