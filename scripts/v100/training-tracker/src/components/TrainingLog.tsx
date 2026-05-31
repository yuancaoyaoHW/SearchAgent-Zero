import { useState } from 'react'
import { useLocalStorage } from '../hooks/useLocalStorage'
import type { TrainingLogEntry } from '../types'

export function TrainingLog() {
  const [logs, setLogs] = useLocalStorage<TrainingLogEntry[]>('tracker-logs', [])
  const [step, setStep] = useState('')
  const [content, setContent] = useState('')
  const [type, setType] = useState<TrainingLogEntry['type']>('observation')

  function addLog() {
    if (!content.trim()) return
    const entry: TrainingLogEntry = {
      id: Date.now().toString(),
      timestamp: new Date().toLocaleString('zh-CN'),
      step: parseInt(step) || 0,
      content: content.trim(),
      type,
    }
    setLogs((prev) => [entry, ...prev])
    setContent('')
    setStep('')
  }

  function removeLog(id: string) {
    setLogs((prev) => prev.filter((l) => l.id !== id))
  }

  const typeColors: Record<TrainingLogEntry['type'], string> = {
    observation: 'bg-blue-100 text-blue-800',
    issue: 'bg-red-100 text-red-800',
    action: 'bg-green-100 text-green-800',
    milestone: 'bg-purple-100 text-purple-800',
  }

  const typeLabels: Record<TrainingLogEntry['type'], string> = {
    observation: '观察',
    issue: '问题',
    action: '操作',
    milestone: '里程碑',
  }

  return (
    <section>
      <h2 className="text-xl font-semibold mb-2">📝 训练日志</h2>
      <p className="text-sm text-gray-500 mb-4">记录训练过程中的观察、问题和操作</p>

      {/* Add entry form */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-4">
        <div className="flex gap-3 mb-3">
          <input
            value={step}
            onChange={(e) => setStep(e.target.value)}
            placeholder="Step"
            type="number"
            className="px-3 py-1.5 border border-gray-300 rounded text-sm w-24"
          />
          <select
            value={type}
            onChange={(e) => setType(e.target.value as TrainingLogEntry['type'])}
            className="px-3 py-1.5 border border-gray-300 rounded text-sm"
          >
            <option value="observation">观察</option>
            <option value="issue">问题</option>
            <option value="action">操作</option>
            <option value="milestone">里程碑</option>
          </select>
        </div>
        <div className="flex gap-3">
          <input
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addLog()}
            placeholder="记录内容..."
            className="flex-1 px-3 py-1.5 border border-gray-300 rounded text-sm"
          />
          <button onClick={addLog} className="px-4 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
            添加
          </button>
        </div>
      </div>

      {/* Log entries */}
      <div className="space-y-2">
        {logs.length === 0 && (
          <p className="text-center text-gray-400 py-8">暂无日志记录</p>
        )}
        {logs.map((log) => (
          <div key={log.id} className="flex items-start gap-3 bg-white rounded-lg border border-gray-200 p-3">
            <span className={`px-2 py-0.5 rounded text-xs font-medium ${typeColors[log.type]}`}>
              {typeLabels[log.type]}
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-gray-800">{log.content}</p>
              <p className="text-xs text-gray-400 mt-1">
                {log.timestamp} {log.step > 0 && `· Step ${log.step}`}
              </p>
            </div>
            <button onClick={() => removeLog(log.id)} className="text-gray-400 hover:text-red-500 text-sm">✕</button>
          </div>
        ))}
      </div>
    </section>
  )
}
