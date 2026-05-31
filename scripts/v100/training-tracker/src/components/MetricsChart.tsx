import { useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { useLocalStorage } from '../hooks/useLocalStorage'
import type { Metric } from '../types'

export function MetricsChart() {
  const [metrics] = useLocalStorage<Metric[]>('tracker-metrics', [])
  const [importText, setImportText] = useState('')

  // Allow importing metrics JSON from WandB export
  const [, setStoredMetrics] = useLocalStorage<Metric[]>('tracker-metrics', [])

  function handleImport() {
    try {
      const data = JSON.parse(importText) as Metric[]
      if (Array.isArray(data)) {
        setStoredMetrics(data)
        setImportText('')
      }
    } catch {
      alert('JSON 格式错误')
    }
  }

  if (metrics.length < 2) {
    return (
      <section>
        <h2 className="text-xl font-semibold mb-2">📊 训练曲线</h2>
        <p className="text-sm text-gray-500 mb-4">连接实时日志流或导入 JSON 数据后显示曲线</p>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-sm font-medium mb-2">导入指标数据 (JSON)</h3>
          <textarea
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            placeholder='[{"step": 1, "reward": 0.02, "kl": 0.001}, ...]'
            className="w-full h-32 px-3 py-2 border border-gray-300 rounded text-sm font-mono"
          />
          <button onClick={handleImport} className="mt-2 px-4 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
            导入
          </button>
        </div>
      </section>
    )
  }

  return (
    <section>
      <h2 className="text-xl font-semibold mb-4">📊 训练曲线</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reward chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Reward</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="reward" stroke="#2563eb" strokeWidth={2} dot={false} name="reward/mean" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Tool success rate */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Tool Call 成功率</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" fontSize={12} />
              <YAxis fontSize={12} domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="tool_success" stroke="#16a34a" strokeWidth={2} dot={false} name="success_rate" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* KL divergence */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">KL Divergence</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="kl" stroke="#d97706" strokeWidth={2} dot={false} name="KL" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Grad norm */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Gradient Norm</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="grad_norm" stroke="#dc2626" strokeWidth={2} dot={false} name="grad_norm" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  )
}
