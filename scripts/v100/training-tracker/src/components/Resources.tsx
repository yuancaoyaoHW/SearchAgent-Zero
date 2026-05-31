import { useState } from 'react'
import { RESOURCE_CATEGORIES, type ResourceCategory, type ResourceItem } from '../data/resources'

function ResourceLink({ item }: { item: ResourceItem }) {
  return (
    <div className="flex items-start gap-3 py-2">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          {item.url ? (
            <a
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-blue-600 hover:text-blue-800 hover:underline truncate"
            >
              {item.name}
            </a>
          ) : (
            <span className="text-sm font-medium text-gray-800">{item.name}</span>
          )}
          {item.url && (
            <span className="text-xs text-gray-400">↗</span>
          )}
        </div>
        <p className="text-xs text-gray-500 mt-0.5">{item.desc}</p>
        {item.path && (
          <code className="text-xs text-gray-400 bg-gray-50 px-1 rounded mt-0.5 inline-block">
            {item.path}
          </code>
        )}
      </div>
    </div>
  )
}

function CategoryCard({ category }: { category: ResourceCategory }) {
  const [open, setOpen] = useState(true)

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 transition-colors text-left"
      >
        <span className="flex items-center gap-2">
          <span>{category.icon}</span>
          <span className="font-medium text-gray-900 text-sm">{category.title}</span>
          <span className="text-xs text-gray-400">
            ({category.subcategories.reduce((sum, sub) => sum + sub.items.length, 0)} 项)
          </span>
        </span>
        <span
          className={`text-gray-400 text-xs transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
        >
          ▼
        </span>
      </button>

      {open && (
        <div className="px-4 pb-4 border-t border-gray-100">
          {category.subcategories.map((sub, i) => (
            <div key={i} className="mt-3">
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                {sub.title}
              </h4>
              <div className="divide-y divide-gray-50">
                {sub.items.map((item, j) => (
                  <ResourceLink key={j} item={item} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function Resources() {
  const totalItems = RESOURCE_CATEGORIES.reduce(
    (sum, cat) => sum + cat.subcategories.reduce((s, sub) => s + sub.items.length, 0),
    0
  )

  return (
    <section>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">📚 参考资料</h2>
        <span className="text-sm text-gray-500">
          {RESOURCE_CATEGORIES.length} 个分类 · {totalItems} 项资源
        </span>
      </div>

      <p className="text-sm text-gray-500 mb-4">
        来自 TRAINING_RESOURCES.md 的完整资料索引。论文、数据集、工具、项目文档一站式查阅。
      </p>

      <div className="space-y-3">
        {RESOURCE_CATEGORIES.map((category) => (
          <CategoryCard key={category.id} category={category} />
        ))}
      </div>
    </section>
  )
}
