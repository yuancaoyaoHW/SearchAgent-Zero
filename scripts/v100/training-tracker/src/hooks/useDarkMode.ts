import { useState, useEffect } from 'react'
import { useLocalStorage } from './useLocalStorage'

export function useDarkMode() {
  const [stored, setStored] = useLocalStorage<'dark' | 'light' | 'system'>('tracker-theme', 'system')
  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const update = () => {
      const dark = stored === 'dark' || (stored === 'system' && mq.matches)
      setIsDark(dark)
      document.documentElement.classList.toggle('dark', dark)
    }
    update()
    mq.addEventListener('change', update)
    return () => mq.removeEventListener('change', update)
  }, [stored])

  return { isDark, mode: stored, setMode: setStored }
}
