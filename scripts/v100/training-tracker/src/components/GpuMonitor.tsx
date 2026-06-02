import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { GpuInfo, GpuSnapshot } from '../types'

interface GpuMonitorProps {
  wsUrl: string
  setWsUrl: (value: string | ((prev: string) => string)) => void
  connected: boolean
  connect: () => void
  disconnect: () => void
  gpuData: GpuInfo[]
  history: GpuSnapshot[]
  defaultWsUrl: string
}

export function GpuMonitor({
  wsUrl,
  setWsUrl,
  connected,
  connect,
  disconnect,
  gpuData,
  history,
  defaultWsUrl,
}: GpuMonitorProps) {
  const utilColor = (pct: number) => {
    if (pct > 80) return 'text-green-600'
    if (pct > 50) return 'text-yellow-600'
    return 'text-red-600'
  }

  const memColor = (used: number, total: number) => {
    const pct = (used / total) * 100
    if (pct > 90) return 'text-red-600'
    if (pct > 75) return 'text-yellow-600'
    return 'text-green-600'
  }

  // Prepare chart data: average utilization over time
  const chartData = history.map((snap) => ({
    time: snap.timestamp,
    avgUtil: Math.round(snap.gpus.reduce((s, g) => s + g.utilization, 0) / snap.gpus.length),
    avgMem: Math.round(snap.gpus.reduce((s, g) => s + (g.memory_used / g.memory_total) * 100, 0) / snap.gpus.length),
  }))

  return (
    <section>
      <h2 className="text-xl font-semibold mb-2">🖥️ GPU 监控</h2>
      <p className="text-sm text-gray-500 mb-4">
        连接 gpu_monitor_server.py 实时监控 GPU 状态。启动命令：python gpu_monitor_server.py
      </p>

      {/* Connection */}
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
      </div>

      {/* GPU cards */}
      {gpuData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
          {gpuData.map((gpu) => (
            <div key={gpu.id} className="bg-white rounded-lg border border-gray-200 p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">GPU {gpu.id}</span>
                <span className="text-xs text-gray-400">{gpu.temperature}°C</span>
              </div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">利用率</span>
                  <span className={`font-mono ${utilColor(gpu.utilization)}`}>{gpu.utilization}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div className="bg-blue-600 h-1.5 rounded-full" style={{ width: `${gpu.utilization}%` }} />
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">显存</span>
                  <span className={`font-mono ${memColor(gpu.memory_used, gpu.memory_total)}`}>
                    {(gpu.memory_used / 1024).toFixed(1)}/{(gpu.memory_total / 1024).toFixed(0)} GB
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div
                    className="bg-green-500 h-1.5 rounded-full"
                    style={{ width: `${(gpu.memory_used / gpu.memory_total) * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-400">
                  <span>功耗</span>
                  <span>{gpu.power}W</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* History chart */}
      {chartData.length > 1 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-3">GPU 利用率趋势</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" fontSize={10} />
              <YAxis domain={[0, 100]} fontSize={12} />
              <Tooltip />
              <Line type="monotone" dataKey="avgUtil" stroke="#2563eb" strokeWidth={2} dot={false} name="平均利用率%" />
              <Line type="monotone" dataKey="avgMem" stroke="#16a34a" strokeWidth={2} dot={false} name="平均显存%" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {!connected && gpuData.length === 0 && (
        <div className="text-center text-gray-400 py-12">
          连接 GPU 监控服务后显示实时数据
        </div>
      )}
    </section>
  )
}
