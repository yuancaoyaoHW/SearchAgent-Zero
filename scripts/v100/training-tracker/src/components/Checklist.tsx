import { useLocalStorage } from '../hooks/useLocalStorage'
import type { Phase } from '../types'

const DEFAULT_PHASES: Phase[] = [
  {
    id: 'env',
    label: '环境安装',
    items: [
      { id: 'env-1', label: 'conda 环境创建 (verl-v100)', checked: false },
      { id: 'env-2', label: 'verl 安装完成', checked: false },
      { id: 'env-3', label: 'vLLM 0.6.6.post1 安装', checked: false },
      { id: 'env-4', label: 'xformers 安装', checked: false },
      { id: 'env-5', label: 'retriever 环境创建', checked: false },
    ],
  },
  {
    id: 'data',
    label: '数据准备',
    items: [
      { id: 'data-1', label: '模型下载 (Qwen2.5-3B-Instruct)', checked: false },
      { id: 'data-2', label: 'E5 检索模型下载', checked: false },
      { id: 'data-3', label: 'Wiki-18 索引下载 (~12GB)', checked: false },
      { id: 'data-4', label: 'Wiki-18 语料下载 (~3GB)', checked: false },
      { id: 'data-5', label: '训练数据预处理完成', checked: false },
    ],
  },
  {
    id: 'service',
    label: '服务部署',
    items: [
      { id: 'svc-1', label: '检索服务启动', checked: false },
      { id: 'svc-2', label: '检索服务健康检查通过', checked: false },
      { id: 'svc-3', label: 'WANDB_API_KEY 已设置', checked: false },
      { id: 'svc-4', label: '8 张 V100 可用', checked: false },
    ],
  },
  {
    id: 'train',
    label: '训练启动',
    items: [
      { id: 'train-1', label: 'Ray 集群初始化成功', checked: false },
      { id: 'train-2', label: 'vLLM 引擎加载完成', checked: false },
      { id: 'train-3', label: '首次 rollout 完成', checked: false },
      { id: 'train-4', label: 'WandB 日志开始记录', checked: false },
    ],
  },
]

export function Checklist() {
  const [phases, setPhases] = useLocalStorage<Phase[]>('tracker-phases', DEFAULT_PHASES)

  function toggleItem(phaseId: string, itemId: string) {
    setPhases((prev) =>
      prev.map((phase) =>
        phase.id === phaseId
          ? {
              ...phase,
              items: phase.items.map((item) =>
                item.id === itemId ? { ...item, checked: !item.checked } : item
              ),
            }
          : phase
      )
    )
  }

  function phaseProgress(phase: Phase) {
    const done = phase.items.filter((i) => i.checked).length
    return { done, total: phase.items.length, pct: Math.round((done / phase.items.length) * 100) }
  }

  const totalDone = phases.reduce((acc, p) => acc + p.items.filter((i) => i.checked).length, 0)
  const totalItems = phases.reduce((acc, p) => acc + p.items.length, 0)

  return (
    <section>
      <h2 className="text-xl font-semibold mb-2">📋 训练前检查清单</h2>
      <p className="text-sm text-gray-500 mb-4">
        总进度: {totalDone}/{totalItems} ({Math.round((totalDone / totalItems) * 100)}%)
      </p>

      <div className="w-full bg-gray-200 rounded-full h-2 mb-6">
        <div
          className="bg-blue-600 h-2 rounded-full transition-all"
          style={{ width: `${(totalDone / totalItems) * 100}%` }}
        />
      </div>

      <div className="space-y-6">
        {phases.map((phase) => {
          const { done, total, pct } = phaseProgress(phase)
          return (
            <div key={phase.id} className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium text-gray-900">{phase.label}</h3>
                <span className="text-sm text-gray-500">{done}/{total} ({pct}%)</span>
              </div>
              <div className="space-y-2">
                {phase.items.map((item) => (
                  <label key={item.id} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={item.checked}
                      onChange={() => toggleItem(phase.id, item.id)}
                      className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className={item.checked ? 'line-through text-gray-400' : 'text-gray-700'}>
                      {item.label}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
