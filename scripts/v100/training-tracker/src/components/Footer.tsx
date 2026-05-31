interface FooterProps {
  onExport: () => void
  onImport: () => void
  onClear: () => void
}

export function Footer({ onExport, onImport, onClear }: FooterProps) {
  return (
    <footer className="mt-8 py-4 border-t border-gray-200 flex items-center justify-between text-sm text-gray-500">
      <p>SearchAgent-Zero Training Tracker v3.0 · 数据保存在浏览器 localStorage</p>
      <div className="flex gap-2">
        <button onClick={onExport} className="px-3 py-1.5 border border-gray-300 rounded hover:bg-gray-100">
          📥 导出数据
        </button>
        <button onClick={onImport} className="px-3 py-1.5 border border-gray-300 rounded hover:bg-gray-100">
          📤 导入数据
        </button>
        <button onClick={onClear} className="px-3 py-1.5 border border-red-300 text-red-600 rounded hover:bg-red-50">
          🗑️ 清空所有数据
        </button>
      </div>
    </footer>
  )
}
