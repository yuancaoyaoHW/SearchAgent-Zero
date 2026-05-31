import type { Tab, TabId } from '../types'

interface TabNavProps {
  tabs: Tab[]
  currentTab: TabId
  onTabChange: (id: TabId) => void
  gpuConnected: boolean
  logConnected: boolean
  alertCount: number
}

export function TabNav({ tabs, currentTab, onTabChange, gpuConnected, logConnected, alertCount }: TabNavProps) {
  return (
    <nav className="flex gap-1 border-b-2 border-gray-200 dark:border-[var(--aod-border)] mb-6 overflow-x-auto">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`px-4 py-2.5 text-sm whitespace-nowrap border-b-2 -mb-[2px] transition-colors ${
            currentTab === tab.id
              ? 'border-[var(--aod-accent)] text-[var(--aod-accent)] font-medium'
              : 'border-transparent text-gray-500 dark:text-[var(--aod-fg-muted)] hover:text-gray-700 dark:hover:text-[var(--aod-fg)]'
          }`}
        >
          {tab.icon} {tab.label}
          {tab.id === 'gpu' && gpuConnected && (
            <span className="ml-1 text-green-500">●</span>
          )}
          {tab.id === 'live' && logConnected && (
            <span className="ml-1 text-green-500">●</span>
          )}
          {tab.id === 'live' && alertCount > 0 && (
            <span className="ml-1 inline-flex items-center justify-center w-4 h-4 text-xs bg-red-500 text-white rounded-full">
              {alertCount}
            </span>
          )}
        </button>
      ))}
    </nav>
  )
}
