import { useState, useEffect } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export function ResearchReport() {
  const [content, setContent] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/research-report.md')
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText)
        return r.text()
      })
      .then((text) => { setContent(text); setLoading(false) })
      .catch(() => { setContent('# 加载失败\n\n请确认 `public/research-report.md` 存在'); setLoading(false) })
  }, [])

  if (loading) return <div className="text-center py-12 text-gray-500">加载报告中...</div>

  return (
    <div className="prose prose-sm max-w-none prose-table:text-sm prose-th:bg-gray-100 dark:prose-th:bg-[var(--aod-bg-highlight)] prose-td:border prose-th:border prose-td:px-2 prose-th:px-2 prose-a:text-blue-600 dark:prose-a:text-[var(--aod-accent)] dark:prose-headings:text-[var(--aod-fg)] dark:prose-strong:text-[var(--aod-fg)] dark:text-[var(--aod-fg)] dark:prose-td:border-[var(--aod-border)] dark:prose-th:border-[var(--aod-border)] dark:prose-code:text-[var(--aod-green)] dark:prose-pre:bg-[var(--aod-bg-light)]">
      <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
    </div>
  )
}
