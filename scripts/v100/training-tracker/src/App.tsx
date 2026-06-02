import { useState, useCallback } from 'react'
import { useDarkMode } from './hooks/useDarkMode'
import { useLocalStorage } from './hooks/useLocalStorage'
import { useWebSocket } from './hooks/useWebSocket'
import { Header } from './components/Header'
import { TabNav } from './components/TabNav'
import { LiveLog } from './components/LiveLog'
import { Checklist } from './components/Checklist'
import { TrainingLog } from './components/TrainingLog'
import { MetricsChart } from './components/MetricsChart'
import { GpuMonitor } from './components/GpuMonitor'
import { Guide } from './components/Guide'
import { Resources } from './components/Resources'
import { Troubleshoot } from './components/Troubleshoot'
import { Footer } from './components/Footer'
import { ResearchReport } from './components/ResearchReport'
import { useRetrievalHealth } from './hooks/useRetrievalHealth'
import type { Tab, TabId, Metric, Alert, GpuInfo, GpuSnapshot, LogLine } from './types'

const MAX_LOG_LINES = 500
const MAX_METRICS = 200
const MAX_GPU_HISTORY = 60
const DEFAULT_LOG_WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8766`
const DEFAULT_GPU_WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8767`

interface RawGpuInfo {
  index?: number
  id?: number
  name?: string
  temperature?: number | null
  gpu_util?: number | null
  utilization?: number | null
  mem_used_mb?: number | null
  memory_used?: number | null
  mem_total_mb?: number | null
  memory_total?: number | null
  power_w?: number | null
  power?: number | null
}

function normalizeGpuInfo(gpu: RawGpuInfo): GpuInfo {
  return {
    id: gpu.id ?? gpu.index ?? 0,
    name: gpu.name ?? 'GPU',
    temperature: gpu.temperature ?? 0,
    utilization: gpu.utilization ?? gpu.gpu_util ?? 0,
    memory_used: gpu.memory_used ?? gpu.mem_used_mb ?? 0,
    memory_total: gpu.memory_total ?? gpu.mem_total_mb ?? 1,
    power: gpu.power ?? gpu.power_w ?? 0,
  }
}

function normalizeMetric(msg: Record<string, unknown>): Metric | null {
  const raw = (msg.data && typeof msg.data === 'object' ? msg.data : msg) as Record<string, unknown>
  if (typeof raw.step !== 'number') return null
  return raw as unknown as Metric
}

const TABS: Tab[] = [
  { id: 'live', label: '实时日志', icon: '📡' },
  { id: 'checklist', label: '检查清单', icon: '📋' },
  { id: 'log', label: '训练日志', icon: '📝' },
  { id: 'metrics', label: '训练曲线', icon: '📊' },
  { id: 'gpu', label: 'GPU 监控', icon: '🖥️' },
  { id: 'guide', label: '训练指南', icon: '📖' },
  { id: 'resources', label: '参考资料', icon: '📚' },
  { id: 'troubleshoot', label: '问题诊断', icon: '🔧' },
  { id: 'research', label: '研究方向', icon: '🔬' },
]

export default function App() {
  const { isDark, mode, setMode } = useDarkMode()
  const [currentTab, setCurrentTab] = useState<TabId>('live')
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [logWsUrl, setLogWsUrl] = useLocalStorage('tracker-log-ws-url', DEFAULT_LOG_WS_URL)
  const [gpuWsUrl, setGpuWsUrl] = useLocalStorage('tracker-gpu-ws-url', DEFAULT_GPU_WS_URL)
  const [logLines, setLogLines] = useState<LogLine[]>([])
  const [metrics, setMetrics] = useLocalStorage<Metric[]>('tracker-metrics', [])
  const [gpuData, setGpuData] = useState<GpuInfo[]>([])
  const [gpuHistory, setGpuHistory] = useState<GpuSnapshot[]>([])

  const retrievalStatus = useRetrievalHealth('http://127.0.0.1:8000/retrieve')

  const handleLogMessage = useCallback((data: unknown) => {
    const msg = data as Record<string, unknown>

    if (msg.type === 'log') {
      const line: LogLine = {
        text: String(msg.line ?? msg.text ?? ''),
        type: (msg.level as LogLine['type']) || 'info',
        timestamp: new Date().toLocaleTimeString(),
      }
      setLogLines((prev) => [...prev.slice(-MAX_LOG_LINES), line])
    }

    if (msg.type === 'metric') {
      const metric = normalizeMetric(msg)
      if (!metric) return
      setMetrics((prev) => [...prev.slice(-MAX_METRICS), metric])

      const newAlerts: Alert[] = []
      if (metric.kl && metric.kl > 2.0) {
        newAlerts.push({
          id: `kl-${Date.now()}`,
          message: `KL 过高: ${metric.kl.toFixed(4)} (阈值 2)`,
          severity: metric.kl > 5 ? 'critical' : 'warning',
          timestamp: new Date().toLocaleTimeString(),
        })
      }
      if (metric.grad_norm && metric.grad_norm > 50) {
        newAlerts.push({
          id: `grad-${Date.now()}`,
          message: `Grad Norm 过高: ${metric.grad_norm.toFixed(2)} (阈值 50)`,
          severity: metric.grad_norm > 100 ? 'critical' : 'warning',
          timestamp: new Date().toLocaleTimeString(),
        })
      }
      if (newAlerts.length > 0) {
        setAlerts((prev) => [...prev, ...newAlerts])
      }
    }
  }, [setMetrics])

  const handleGpuMessage = useCallback((data: unknown) => {
    const msg = data as { gpus?: RawGpuInfo[] }
    if (!msg.gpus) return
    const gpus = msg.gpus.map(normalizeGpuInfo)
    setGpuData(gpus)
    setGpuHistory((prev) => [
      ...prev.slice(-MAX_GPU_HISTORY),
      { timestamp: new Date().toLocaleTimeString(), gpus },
    ])
  }, [])

  const logSocket = useWebSocket({
    url: logWsUrl,
    onMessage: handleLogMessage,
  })

  const gpuSocket = useWebSocket({
    url: gpuWsUrl,
    onMessage: handleGpuMessage,
  })

  const dismissAlert = useCallback((id: string) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id))
  }, [])

  function handleExport() {
    const data: Record<string, string | null> = {}
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith('tracker-')) {
        data[key] = localStorage.getItem(key)
      }
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training-tracker-export-${new Date().toISOString().slice(0, 10)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function handleImport() {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = (ev) => {
        try {
          const data = JSON.parse(ev.target?.result as string) as Record<string, string>
          Object.entries(data).forEach(([key, value]) => {
            if (key.startsWith('tracker-')) {
              localStorage.setItem(key, value)
            }
          })
          window.location.reload()
        } catch {
          alert('导入失败：JSON 格式错误')
        }
      }
      reader.readAsText(file)
    }
    input.click()
  }

  function handleClear() {
    if (confirm('确定清空所有训练追踪数据？此操作不可恢复。')) {
      const keys: string[] = []
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i)
        if (key?.startsWith('tracker-')) keys.push(key)
      }
      keys.forEach((k) => localStorage.removeItem(k))
      window.location.reload()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 dark:bg-[var(--aod-bg)] dark:text-[var(--aod-fg)] transition-colors">
      <div className="max-w-6xl mx-auto px-5 py-4">
        <div className="flex justify-end mb-2">
          <button
            onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')}
            className="px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-[var(--aod-border)] dark:bg-[var(--aod-bg-highlight)] dark:text-[var(--aod-fg)] hover:opacity-80 transition-opacity"
          >
            {isDark ? '☀️ Light' : '🌙 Dark'}
          </button>
        </div>
        <Header
          retrievalStatus={retrievalStatus}
          gpuConnected={gpuSocket.connected}
          logConnected={logSocket.connected}
        />

      <TabNav
        tabs={TABS}
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        gpuConnected={gpuSocket.connected}
        logConnected={logSocket.connected}
        alertCount={alerts.length}
      />

      <main>
        {currentTab === 'live' && (
          <LiveLog
            wsUrl={logWsUrl}
            setWsUrl={setLogWsUrl}
            connected={logSocket.connected}
            connect={logSocket.connect}
            disconnect={logSocket.disconnect}
            logLines={logLines}
            metrics={metrics}
            alerts={alerts}
            onDismissAlert={dismissAlert}
            defaultWsUrl={DEFAULT_LOG_WS_URL}
          />
        )}
        {currentTab === 'checklist' && <Checklist />}
        {currentTab === 'log' && <TrainingLog />}
        {currentTab === 'metrics' && <MetricsChart />}
        {currentTab === 'gpu' && (
          <GpuMonitor
            wsUrl={gpuWsUrl}
            setWsUrl={setGpuWsUrl}
            connected={gpuSocket.connected}
            connect={gpuSocket.connect}
            disconnect={gpuSocket.disconnect}
            gpuData={gpuData}
            history={gpuHistory}
            defaultWsUrl={DEFAULT_GPU_WS_URL}
          />
        )}
        {currentTab === 'guide' && <Guide />}
        {currentTab === 'resources' && <Resources />}
        {currentTab === 'troubleshoot' && <Troubleshoot />}
        {currentTab === 'research' && <ResearchReport />}
      </main>

      <Footer onExport={handleExport} onImport={handleImport} onClear={handleClear} />

      </div>
    </div>
  )
}
