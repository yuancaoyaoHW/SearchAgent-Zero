import type { ConnectionStatus } from '../types'

interface HeaderProps {
  retrievalStatus: ConnectionStatus
  gpuConnected: boolean
  logConnected: boolean
}

export function Header({ retrievalStatus, gpuConnected, logConnected }: HeaderProps) {
  const statusColor = (online: boolean) =>
    online ? 'bg-green-500' : 'bg-gray-300'

  const retrievalColor =
    retrievalStatus === 'online' ? 'bg-green-500' :
    retrievalStatus === 'checking' ? 'bg-yellow-400' : 'bg-gray-300'

  const retrievalText =
    retrievalStatus === 'online' ? '正常' :
    retrievalStatus === 'checking' ? '检测中' : '离线'

  return (
    <header className="text-center py-8">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-[var(--aod-fg)]">
        🔍 SearchAgent-Zero Training Tracker
      </h1>
      <p className="text-gray-500 dark:text-[var(--aod-fg-muted)] mt-1">训练进度追踪 · 性能监控 · 问题记录</p>
      <div className="flex items-center justify-center gap-4 mt-4 text-sm">
        <span className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${retrievalColor}`} />
          🔍 检索服务: {retrievalText}
        </span>
        <span className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${statusColor(gpuConnected)}`} />
          🖥️ GPU 监控: {gpuConnected ? '已连接' : '未连接'}
        </span>
        <span className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${statusColor(logConnected)}`} />
          📝 日志流: {logConnected ? '已连接' : '未连接'}
        </span>
      </div>
    </header>
  )
}
