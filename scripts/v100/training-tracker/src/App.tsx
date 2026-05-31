import { useState, useCallback } from 'react'
import { Header } from './components/Header'
import { TabNav } from './components/TabNav'
import { LiveLog } from './components/LiveLog'
import { Checklist } from './components/Checklist'
import { TrainingLog } from './components/TrainingLog'
import { MetricsChart } from './components/MetricsChart'
import { GpuMonitor } from './components/GpuMonitor'
import { Troubleshoot } from './components/Troubleshoot'
import { Reference } from './components/Reference'
import { Footer } from './components/Footer'
import { useRetrievalHealth } from './hooks/useRetrievalHealth'
import { useLocalStorage } from './hooks/useLocalStorage'
import type { Tab, TabId, Metric, Alert } from './types'

const TABS: Tab[] = [
  { id: 'live', label: '实时日志', icon: '📡' },
  { id: 'checklist', label: '检查清单', icon: '📋' },
  { id: 'log', label: '训练日志', icon: '📝' },
  { id: 'metrics', label: '训练曲线', icon: '📊' },
  { id: 'gpu', label: 'GPU 监控', icon: '🖥️' },
  { id: 'troubleshoot', label: '问题诊断', icon: '🔧' },
  { id: 'reference', label: '快速参考', icon: '📚' },
]

export default function App() {
  const [currentTab, setCurrentTab] = useState<TabId>('live')
  const [logConnected, setLogConnected] = useState(false)
  const [gpuConnected, setGpuConnected] = useState(false)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [, setMetrics] = useLocalStorage<Metric[]>('tracker-metrics', [])

  const retrievalStatus = useRetrievalHealth('http://127.0.0.1:8000/retrieve')

  const handleMetricsUpdate = useCallback((newMetrics: Metric[]) => {
    setMetrics(newMetrics)
  }, [setMetrics])

  const handleAlertsUpdate = useCallback((newAlerts: Alert[]) => {
    setAlerts(newAlerts)
  }, [])

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
    <div className="max-w-6xl mx-auto px-5 py-4">
      <Header
        retrievalStatus={retrievalStatus}
        gpuConnected={gpuConnected}
        logConnected={logConnected}
      />

      <TabNav
        tabs={TABS}
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        gpuConnected={gpuConnected}
        logConnected={logConnected}
        alertCount={alerts.length}
      />

      <main>
        {currentTab === 'live' && (
          <LiveLog
            onMetricsUpdate={handleMetricsUpdate}
            onAlertsUpdate={handleAlertsUpdate}
            alerts={alerts}
            onDismissAlert={dismissAlert}
          />
        )}
        {currentTab === 'checklist' && <Checklist />}
        {currentTab === 'log' && <TrainingLog />}
        {currentTab === 'metrics' && <MetricsChart />}
        {currentTab === 'gpu' && <GpuMonitor />}
        {currentTab === 'troubleshoot' && <Troubleshoot />}
        {currentTab === 'reference' && <Reference />}
      </main>

      <Footer onExport={handleExport} onImport={handleImport} onClear={handleClear} />

      {/* Hidden state sync for child components */}
      <ConnectionSync onLogChange={setLogConnected} onGpuChange={setGpuConnected} />
    </div>
  )
}

// Invisible component to sync connection state from child WebSocket hooks
// In a real app you'd lift state or use context; this is a pragmatic bridge
function ConnectionSync({ onLogChange: _onLogChange, onGpuChange: _onGpuChange }: {
  onLogChange: (v: boolean) => void
  onGpuChange: (v: boolean) => void
}) {
  // Connection state is managed within LiveLog and GpuMonitor directly
  // This is a placeholder for future context-based state sharing
  return null
}
