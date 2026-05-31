import { useState } from 'react'
import { GUIDE_CHAPTERS, type GuideSection, type GuideChapter } from '../data/guide'

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className="absolute top-2 right-2 px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
    >
      {copied ? '✓ 已复制' : '复制'}
    </button>
  )
}

function SectionRenderer({ section }: { section: GuideSection }) {
  switch (section.type) {
    case 'text':
      return <p className="text-gray-700 leading-relaxed">{section.content}</p>

    case 'tip':
      return (
        <div className="flex items-start gap-2 bg-blue-50 border border-blue-200 rounded-md p-3">
          <span className="text-blue-500 mt-0.5">💡</span>
          <p className="text-blue-800 text-sm">{section.content}</p>
        </div>
      )

    case 'warning':
      return (
        <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-md p-3">
          <span className="text-amber-500 mt-0.5">⚠️</span>
          <p className="text-amber-800 text-sm">{section.content}</p>
        </div>
      )

    case 'code':
      return (
        <div className="relative">
          <CopyButton text={section.content || ''} />
          <pre className="bg-gray-900 text-gray-100 rounded-md p-4 overflow-x-auto text-sm leading-relaxed">
            <code>{section.content}</code>
          </pre>
        </div>
      )

    case 'list':
      return (
        <ul className="list-disc list-inside space-y-1 text-gray-700 text-sm">
          {section.items?.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      )

    case 'table':
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                {section.headers?.map((h, i) => (
                  <th key={i} className="text-left py-2 px-3 font-medium text-gray-700">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {section.rows?.map((row, i) => (
                <tr key={i} className="border-b border-gray-100">
                  {row.map((cell, j) => (
                    <td key={j} className="py-2 px-3 text-gray-600">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )

    default:
      return null
  }
}

export function Guide() {
  const [allOpen, setAllOpen] = useState(false)

  return (
    <section>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">📖 训练指南</h2>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">
            共 {GUIDE_CHAPTERS.length} 章
          </span>
          <button
            onClick={() => setAllOpen(!allOpen)}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            {allOpen ? '全部收起' : '全部展开'}
          </button>
        </div>
      </div>

      <p className="text-sm text-gray-500 mb-4">
        来自 TRAINING_MANUAL.md 的完整训练流程。点击章节标题展开详情。
      </p>

      {/* Progress indicator */}
      <div className="flex items-center gap-1 mb-6 overflow-x-auto pb-2">
        {GUIDE_CHAPTERS.map((ch, i) => (
          <div key={ch.id} className="flex items-center">
            <span className="text-xs whitespace-nowrap px-2 py-1 bg-gray-100 rounded text-gray-600">
              {ch.icon} {ch.title.replace(/第.阶段：/, '')}
            </span>
            {i < GUIDE_CHAPTERS.length - 1 && (
              <span className="text-gray-300 mx-1">→</span>
            )}
          </div>
        ))}
      </div>

      <div className="space-y-3">
        {GUIDE_CHAPTERS.map((chapter) => (
          <ChapterAccordionControlled
            key={chapter.id}
            chapter={chapter}
            forceOpen={allOpen}
          />
        ))}
      </div>
    </section>
  )
}

// Controlled version that responds to "expand all"
function ChapterAccordionControlled({
  chapter,
  forceOpen,
}: {
  chapter: GuideChapter
  forceOpen: boolean
}) {
  const [localOpen, setLocalOpen] = useState(false)
  const isOpen = forceOpen || localOpen

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setLocalOpen(!localOpen)}
        className="w-full flex items-center justify-between px-5 py-4 bg-white hover:bg-gray-50 transition-colors text-left"
      >
        <span className="flex items-center gap-3">
          <span className="text-xl">{chapter.icon}</span>
          <span className="font-medium text-gray-900">{chapter.title}</span>
          <span className="text-xs text-gray-400">
            ({chapter.sections.length} 节)
          </span>
        </span>
        <span
          className={`text-gray-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        >
          ▼
        </span>
      </button>

      {isOpen && (
        <div className="px-5 py-4 border-t border-gray-100 space-y-5 bg-white">
          {chapter.sections.map((section, i) => (
            <div key={i}>
              <h4 className="text-sm font-medium text-gray-800 mb-2">{section.title}</h4>
              <SectionRenderer section={section} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
