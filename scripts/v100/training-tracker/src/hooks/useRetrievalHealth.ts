import { useState, useEffect, useRef } from 'react'
import type { ConnectionStatus } from '../types'

export function useRetrievalHealth(url: string, intervalMs = 10000) {
  const [status, setStatus] = useState<ConnectionStatus>('checking')
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    async function probe() {
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: 'health check', topk: 1 }),
          signal: AbortSignal.timeout(5000),
        })
        setStatus(res.ok ? 'online' : 'offline')
      } catch {
        setStatus('offline')
      }
    }

    probe()
    timerRef.current = setInterval(probe, intervalMs)
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [url, intervalMs])

  return status
}
