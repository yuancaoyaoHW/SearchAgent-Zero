import { useState, useEffect, useRef, useCallback } from 'react'

interface UseWebSocketOptions {
  url: string
  onMessage?: (data: unknown) => void
  reconnectInterval?: number
  autoConnect?: boolean
}

export function useWebSocket({ url, onMessage, reconnectInterval = 3000, autoConnect = false }: UseWebSocketOptions) {
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const urlRef = useRef(url)
  urlRef.current = url

  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
    }
    try {
      const ws = new WebSocket(urlRef.current)
      ws.onopen = () => {
        setConnected(true)
        if (reconnectTimer.current) {
          clearTimeout(reconnectTimer.current)
          reconnectTimer.current = null
        }
      }
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data as string) as unknown
          onMessage?.(data)
        } catch {
          // non-JSON message, ignore
        }
      }
      ws.onclose = () => {
        setConnected(false)
        scheduleReconnect()
      }
      ws.onerror = () => {
        setConnected(false)
      }
      wsRef.current = ws
    } catch {
      setConnected(false)
    }
  }, [onMessage])

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setConnected(false)
  }, [])

  function scheduleReconnect() {
    if (reconnectTimer.current) return
    reconnectTimer.current = setTimeout(() => {
      reconnectTimer.current = null
      if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
        connect()
      }
    }, reconnectInterval)
  }

  useEffect(() => {
    if (autoConnect) {
      connect()
    }
    return () => {
      disconnect()
    }
  }, [autoConnect, connect, disconnect])

  return { connected, connect, disconnect }
}
