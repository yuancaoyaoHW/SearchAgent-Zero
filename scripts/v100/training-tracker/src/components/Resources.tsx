import { useState } from 'react'
import { useLocalStorage } from '../hooks/useLocalStorage'
import { RESOURCE_CATEGORIES, type ResourceCategory, type ResourceSubcategory, type ResourceItem } from '../data/resources'

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

export function Resources() {
  const [categories, setCategories] = useLocalStorage<ResourceCategory[]>('tracker-resources', RESOURCE_CATEGORIES)
  const [expandedCat, setExpandedCat] = useState<string | null>(categories[0]?.id ?? null)
  const [editingItem, setEditingItem] = useState<{ catId: string; subIdx: number; itemIdx: number } | null>(null)
  const [showAddItem, setShowAddItem] = useState<{ catId: string; subIdx: number } | null>(null)
  const [showAddSub, setShowAddSub] = useState<string | null>(null)
  const [showAddCat, setShowAddCat] = useState(false)

  const [formName, setFormName] = useState('')
  const [formUrl, setFormUrl] = useState('')
  const [formDesc, setFormDesc] = useState('')
  const [formPath, setFormPath] = useState('')
  const [formTitle, setFormTitle] = useState('')
  const [formIcon, setFormIcon] = useState('📂')

  function resetForm() {
    setFormName(''); setFormUrl(''); setFormDesc(''); setFormPath(''); setFormTitle(''); setFormIcon('📂')
  }

  function addCategory() {
    if (!formTitle.trim()) return
    const newCat: ResourceCategory = {
      id: uid(),
      title: formTitle.trim(),
      icon: formIcon || '📂',
      subcategories: [],
    }
    setCategories((prev) => [...prev, newCat])
    setShowAddCat(false)
    resetForm()
  }

  function deleteCategory(catId: string) {
    if (!confirm('确定删除该分类及其所有内容？')) return
    setCategories((prev) => prev.filter((c) => c.id !== catId))
    if (expandedCat === catId) setExpandedCat(null)
  }

  function addSubcategory(catId: string) {
    if (!formTitle.trim()) return
    const newSub: ResourceSubcategory = { title: formTitle.trim(), items: [] }
    setCategories((prev) =>
      prev.map((c) => c.id === catId ? { ...c, subcategories: [...c.subcategories, newSub] } : c)
    )
    setShowAddSub(null)
    resetForm()
  }

  function deleteSubcategory(catId: string, subIdx: number) {
    if (!confirm('确定删除该子分类？')) return
    setCategories((prev) =>
      prev.map((c) => c.id === catId
        ? { ...c, subcategories: c.subcategories.filter((_, i) => i !== subIdx) }
        : c
      )
    )
  }

  function addItem(catId: string, subIdx: number) {
    if (!formName.trim()) return
    const newItem: ResourceItem = {
      name: formName.trim(),
      url: formUrl.trim() || null,
      desc: formDesc.trim(),
      path: formPath.trim() || undefined,
    }
    setCategories((prev) =>
      prev.map((c) => c.id === catId
        ? {
            ...c,
            subcategories: c.subcategories.map((sub, i) =>
              i === subIdx ? { ...sub, items: [...sub.items, newItem] } : sub
            ),
          }
        : c
      )
    )
    setShowAddItem(null)
    resetForm()
  }

  function updateItem(catId: string, subIdx: number, itemIdx: number) {
    if (!formName.trim()) return
    const updated: ResourceItem = {
      name: formName.trim(),
      url: formUrl.trim() || null,
      desc: formDesc.trim(),
      path: formPath.trim() || undefined,
    }
    setCategories((prev) =>
      prev.map((c) => c.id === catId
        ? {
            ...c,
            subcategories: c.subcategories.map((sub, i) =>
              i === subIdx
                ? { ...sub, items: sub.items.map((item, j) => j === itemIdx ? updated : item) }
                : sub
            ),
          }
        : c
      )
    )
    setEditingItem(null)
    resetForm()
  }

  function deleteItem(catId: string, subIdx: number, itemIdx: number) {
    setCategories((prev) =>
      prev.map((c) => c.id === catId
        ? {
            ...c,
            subcategories: c.subcategories.map((sub, i) =>
              i === subIdx ? { ...sub, items: sub.items.filter((_, j) => j !== itemIdx) } : sub
            ),
          }
        : c
      )
    )
  }

  function startEdit(catId: string, subIdx: number, itemIdx: number, item: ResourceItem) {
    setEditingItem({ catId, subIdx, itemIdx })
    setShowAddItem(null)
    setFormName(item.name)
    setFormUrl(item.url || '')
    setFormDesc(item.desc)
    setFormPath(item.path || '')
  }

  function startAddItem(catId: string, subIdx: number) {
    setShowAddItem({ catId, subIdx })
    setEditingItem(null)
    resetForm()
  }

  function resetToDefaults() {
    if (!confirm('确定恢复为默认参考资料？自定义内容将丢失。')) return
    setCategories(RESOURCE_CATEGORIES)
  }

  const totalItems = categories.reduce(
    (acc, c) => acc + c.subcategories.reduce((a, s) => a + s.items.length, 0), 0
  )

  return (
    <section>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold">📚 参考资料</h2>
          <p className="text-sm text-gray-500 mt-1">{categories.length} 个分类 · {totalItems} 条资料</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => { setShowAddCat(true); resetForm() }}
            className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
          >
            + 新分类
          </button>
          <button
            onClick={resetToDefaults}
            className="px-3 py-1.5 border border-gray-300 text-gray-600 rounded text-sm hover:bg-gray-100"
          >
            恢复默认
          </button>
        </div>
      </div>

      {/* Add category form */}
      {showAddCat && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
          <h4 className="text-sm font-medium mb-2">新建分类</h4>
          <div className="flex gap-2">
            <input
              value={formIcon}
              onChange={(e) => setFormIcon(e.target.value)}
              placeholder="图标"
              className="px-2 py-1.5 border border-gray-300 rounded text-sm w-16 text-center"
            />
            <input
              value={formTitle}
              onChange={(e) => setFormTitle(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addCategory()}
              placeholder="分类名称"
              className="flex-1 px-3 py-1.5 border border-gray-300 rounded text-sm"
              autoFocus
            />
            <button onClick={addCategory} className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm">确定</button>
            <button onClick={() => setShowAddCat(false)} className="px-3 py-1.5 border border-gray-300 rounded text-sm">取消</button>
          </div>
        </div>
      )}

      {/* Category list */}
      <div className="space-y-3">
        {categories.map((cat) => (
          <div key={cat.id} className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            {/* Category header */}
            <div
              className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-gray-50"
              onClick={() => setExpandedCat(expandedCat === cat.id ? null : cat.id)}
            >
              <div className="flex items-center gap-2">
                <span className="text-lg">{cat.icon}</span>
                <h3 className="font-medium text-gray-900">{cat.title}</h3>
                <span className="text-xs text-gray-400">
                  ({cat.subcategories.reduce((a, s) => a + s.items.length, 0)} 条)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={(e) => { e.stopPropagation(); deleteCategory(cat.id) }}
                  className="text-gray-400 hover:text-red-500 text-sm"
                  title="删除分类"
                >
                  🗑️
                </button>
                <span className={`text-gray-400 transition-transform ${expandedCat === cat.id ? 'rotate-90' : ''}`}>
                  ▶
                </span>
              </div>
            </div>

            {/* Expanded content */}
            {expandedCat === cat.id && (
              <div className="border-t border-gray-200 px-4 py-3">
                {cat.subcategories.map((sub, subIdx) => (
                  <div key={subIdx} className="mb-4 last:mb-0">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-medium text-gray-700">{sub.title}</h4>
                      <div className="flex gap-1">
                        <button
                          onClick={() => startAddItem(cat.id, subIdx)}
                          className="text-xs text-blue-600 hover:text-blue-800"
                        >
                          + 添加
                        </button>
                        <button
                          onClick={() => deleteSubcategory(cat.id, subIdx)}
                          className="text-xs text-gray-400 hover:text-red-500 ml-2"
                        >
                          删除
                        </button>
                      </div>
                    </div>

                    {/* Items */}
                    <div className="space-y-1.5">
                      {sub.items.map((item, itemIdx) => {
                        const isEditing = editingItem?.catId === cat.id && editingItem.subIdx === subIdx && editingItem.itemIdx === itemIdx

                        if (isEditing) {
                          return (
                            <ItemForm
                              key={itemIdx}
                              name={formName} url={formUrl} desc={formDesc} path={formPath}
                              onNameChange={setFormName} onUrlChange={setFormUrl}
                              onDescChange={setFormDesc} onPathChange={setFormPath}
                              onSubmit={() => updateItem(cat.id, subIdx, itemIdx)}
                              onCancel={() => { setEditingItem(null); resetForm() }}
                              submitLabel="保存"
                            />
                          )
                        }

                        return (
                          <div key={itemIdx} className="flex items-start gap-2 group py-1.5 px-2 rounded hover:bg-gray-50">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                {item.url ? (
                                  <a href={item.url} target="_blank" rel="noopener noreferrer"
                                    className="text-sm text-blue-600 hover:underline font-medium truncate">
                                    {item.name}
                                  </a>
                                ) : (
                                  <span className="text-sm font-medium text-gray-800 truncate">{item.name}</span>
                                )}
                                {item.path && (
                                  <code className="text-xs bg-gray-100 text-gray-500 px-1 rounded hidden sm:inline">
                                    {item.path}
                                  </code>
                                )}
                              </div>
                              <p className="text-xs text-gray-500 truncate">{item.desc}</p>
                            </div>
                            <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                              <button
                                onClick={() => startEdit(cat.id, subIdx, itemIdx, item)}
                                className="text-xs text-gray-400 hover:text-blue-600"
                                title="编辑"
                              >
                                ✏️
                              </button>
                              <button
                                onClick={() => deleteItem(cat.id, subIdx, itemIdx)}
                                className="text-xs text-gray-400 hover:text-red-500"
                                title="删除"
                              >
                                ✕
                              </button>
                            </div>
                          </div>
                        )
                      })}
                    </div>

                    {/* Add item form */}
                    {showAddItem?.catId === cat.id && showAddItem.subIdx === subIdx && (
                      <ItemForm
                        name={formName} url={formUrl} desc={formDesc} path={formPath}
                        onNameChange={setFormName} onUrlChange={setFormUrl}
                        onDescChange={setFormDesc} onPathChange={setFormPath}
                        onSubmit={() => addItem(cat.id, subIdx)}
                        onCancel={() => { setShowAddItem(null); resetForm() }}
                        submitLabel="添加"
                      />
                    )}
                  </div>
                ))}

                {/* Add subcategory */}
                {showAddSub === cat.id ? (
                  <div className="mt-3 pt-3 border-t border-gray-100">
                    <div className="flex gap-2">
                      <input
                        value={formTitle}
                        onChange={(e) => setFormTitle(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && addSubcategory(cat.id)}
                        placeholder="子分类名称"
                        className="flex-1 px-3 py-1.5 border border-gray-300 rounded text-sm"
                        autoFocus
                      />
                      <button onClick={() => addSubcategory(cat.id)} className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm">确定</button>
                      <button onClick={() => { setShowAddSub(null); resetForm() }} className="px-3 py-1.5 border border-gray-300 rounded text-sm">取消</button>
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={() => { setShowAddSub(cat.id); resetForm() }}
                    className="mt-3 text-xs text-blue-600 hover:text-blue-800"
                  >
                    + 新建子分类
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

// --- Reusable item form ---
interface ItemFormProps {
  name: string; url: string; desc: string; path: string
  onNameChange: (v: string) => void
  onUrlChange: (v: string) => void
  onDescChange: (v: string) => void
  onPathChange: (v: string) => void
  onSubmit: () => void
  onCancel: () => void
  submitLabel: string
}

function ItemForm({ name, url, desc, path, onNameChange, onUrlChange, onDescChange, onPathChange, onSubmit, onCancel, submitLabel }: ItemFormProps) {
  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 mt-2">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-2">
        <input
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="名称 *"
          className="px-2 py-1.5 border border-gray-300 rounded text-sm"
          autoFocus
        />
        <input
          value={url}
          onChange={(e) => onUrlChange(e.target.value)}
          placeholder="URL (可选)"
          className="px-2 py-1.5 border border-gray-300 rounded text-sm"
        />
        <input
          value={desc}
          onChange={(e) => onDescChange(e.target.value)}
          placeholder="描述"
          className="px-2 py-1.5 border border-gray-300 rounded text-sm"
        />
        <input
          value={path}
          onChange={(e) => onPathChange(e.target.value)}
          placeholder="文件路径 (可选)"
          className="px-2 py-1.5 border border-gray-300 rounded text-sm"
        />
      </div>
      <div className="flex gap-2">
        <button
          onClick={onSubmit}
          onKeyDown={(e) => e.key === 'Enter' && onSubmit()}
          className="px-3 py-1 bg-blue-600 text-white rounded text-sm"
        >
          {submitLabel}
        </button>
        <button onClick={onCancel} className="px-3 py-1 border border-gray-300 rounded text-sm">取消</button>
      </div>
    </div>
  )
}
