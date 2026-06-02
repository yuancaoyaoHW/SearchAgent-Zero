import { useState, useRef, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { Metric, LogLine, Alert } from '../types'

interface LiveLogProps {
  wsUrl: string
  setWsUrl: (value: string | ((prev: string) => string)) => void
  connected: boolean
  connect: () => void
  disconnect: () => void
  logLines: LogLine[]
  metrics: Metric[]
  alerts: Alert[]
  onDismissAlert: (id: string) => void
  defaultWsUrl: string
}

export function LiveLog({
  wsUrl,
  setWsUrl,
  connected,
  connect,
  disconnect,
  logLines,
  metrics,
  alerts,
  onDismissAlert,
  defaultWsUrl,
}: LiveLogProps) {
  const [autoScroll, setAutoScroll] = useState(true)
  const logContainerRef = useRef<HTMLDivElement>(null)

  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null

  // Auto-scroll
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logLines, autoScroll])

  const lineClass = (type: LogLine['type']) => {
    switch (type) {
      case 'error': return 'text-red-600'
      case 'warn': return 'text-yellow-600'
      case 'metric': return 'text-blue-600'
      default: return 'text-gray-700'
    }
  }

  const rewardColor = (reward?: number) => {
    if (reward === undefined) return ''
    if (reward > 0.3) return 'text-green-600'
    if (reward > 0.1) return 'text-yellow-600'
    return 'text-gray-600'
  }

  return (
    <section>
      <h2 className="text-xl font-semibold mb-2">📡 实时训练日志</h2>
      <p className="text-sm text-gray-500 mb-4">
        连接 training_log_server.py 自动采集训练指标。启动命令：python training_log_server.py --log-file /path/to/log
      </p>

      {/* Connection controls */}
      <div className="flex items-center gap-3 mb-4">
        <input
          value={wsUrl}
          onChange={(e) => setWsUrl(e.target.value)}
          placeholder={defaultWsUrl}
          className="px-3 py-1.5 border border-gray-300 rounded text-sm w-60"
        />
        {!connected ? (
          <button onClick={connect} className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
            🔌 连接
          </button>
        ) : (
          <button onClick={disconnect} className="px-3 py-1.5 bg-red-600 text-white rounded text-sm hover:bg-red-700">
            ⏹️ 断开
          </button>
        )}
        <label className="flex items-center gap-1 text-sm text-gray-600">
          <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} />
          自动滚动
        </label>
      </div>

      {/* Live metrics cards */}
      {latestMetric && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
          <MetricCard label="Step" value={String(latestMetric.step)} />
          <MetricCard label="Reward" value={latestMetric.reward?.toFixed(3) ?? '-'} className={rewardColor(latestMetric.reward)} />
          <MetricCard label="Tool 成功率" value={latestMetric.tool_success ? `${(latestMetric.tool_success * 100).toFixed(1)}%` : '-'} />
          <MetricCard label="KL" value={latestMetric.kl?.toFixed(4) ?? '-'} />
          <MetricCard label="Grad Norm" value={latestMetric.grad_norm?.toFixed(2) ?? '-'} />
        </div>
      )}

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="mb-4 space-y-2">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`flex items-center justify-between px-3 py-2 rounded text-sm ${
                alert.severity === 'critical' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'
              }`}
            >
              <span>{alert.timestamp} — {alert.message}</span>
              <button onClick={() => onDismissAlert(alert.id)} className="ml-2 text-gray-500 hover:text-gray-700">✕</button>
            </div>
          ))}
        </div>
      )}

      {/* Live reward chart */}
      {metrics.length > 1 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">实时 Reward 曲线</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Line type="monotone" dataKey="reward" stroke="#2563eb" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Log stream */}
      <div
        ref={logContainerRef}
        className="bg-gray-900 text-gray-100 rounded-lg p-4 h-80 overflow-y-auto font-mono text-xs"
      >
        {logLines.length === 0 ? (
          <p className="text-gray-500">等待日志连接...</p>
        ) : (
          logLines.map((line, i) => (
            <div key={i} className={lineClass(line.type)}>
              <span className="text-gray-500">[{line.timestamp}]</span> {line.text}
            </div>
          ))
        )}
      </div>
    </section>
  )
}

function MetricCard({ label, value, className = '' }: { label: string; value: string; className?: string }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-3 text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-lg font-semibold mt-0.5 ${className}`}>{value}</div>
    </div>
  )
}
