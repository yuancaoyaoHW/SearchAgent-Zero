import { useState, useEffect } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const REPORT_PATH = '../../../../reports/rl_sft_search_agent_research_directions.md'

export function ResearchReport() {
  const [content, setContent] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(REPORT_PATH)
      .then((r) => r.text())
      .then((text) => { setContent(text); setLoading(false) })
      .catch(() => { setContent('# 加载失败\n\n请确认报告文件存在于 `reports/rl_sft_search_agent_research_directions.md`'); setLoading(false) })
  }, [])

  if (loading) return <div className="text-center py-12 text-gray-500">加载报告中...</div>

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert prose-table:text-sm prose-th:bg-gray-100 dark:prose-th:bg-gray-800 prose-td:border prose-th:border prose-td:px-2 prose-th:px-2">
      <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
    </div>
  )
}
