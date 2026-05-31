export interface Metric {
  step: number
  reward?: number
  reward_std?: number
  loss?: number
  entropy?: number
  tool_success?: number
  tool_turn?: number
  kl?: number
  grad_norm?: number
  throughput?: number
  time_per_step?: number
  timestamp?: string
}

export interface LogLine {
  text: string
  type: 'info' | 'warn' | 'error' | 'metric'
  timestamp: string
}

export interface Alert {
  id: string
  message: string
  severity: 'warning' | 'critical'
  timestamp: string
}

export interface GpuInfo {
  id: number
  name: string
  temperature: number
  utilization: number
  memory_used: number
  memory_total: number
  power: number
}

export interface GpuSnapshot {
  timestamp: string
  gpus: GpuInfo[]
}

export interface TrainingLogEntry {
  id: string
  timestamp: string
  step: number
  content: string
  type: 'observation' | 'issue' | 'action' | 'milestone'
}

export interface Phase {
  id: string
  label: string
  items: ChecklistItem[]
}

export interface ChecklistItem {
  id: string
  label: string
  checked: boolean
}

export type TabId = 'live' | 'checklist' | 'log' | 'metrics' | 'gpu' | 'guide' | 'resources' | 'troubleshoot' | 'research'

export interface Tab {
  id: TabId
  label: string
  icon: string
}

export interface Symptom {
  id: string
  label: string
  diagnosis: string
  commands: string[]
}

export type ConnectionStatus = 'online' | 'offline' | 'checking'
