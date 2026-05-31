export function Reference() {
  return (
    <section>
      <h2 className="text-xl font-semibold mb-4">📚 快速参考</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Metrics reference */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-medium text-gray-900 mb-3">指标健康范围</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 pr-4 font-medium text-gray-700">指标</th>
                  <th className="text-left py-2 pr-4 font-medium text-gray-700">健康范围</th>
                  <th className="text-left py-2 font-medium text-gray-700">异常信号</th>
                </tr>
              </thead>
              <tbody className="text-gray-600">
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">reward/mean</td>
                  <td className="py-2 pr-4">逐步上升</td>
                  <td className="py-2">持续为 0 或突然跳到 {'>'} 0.8</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">kl_divergence</td>
                  <td className="py-2 pr-4">0.01 - 0.5</td>
                  <td className="py-2">{'>'} 5 策略崩溃</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">tool_call_success_rate</td>
                  <td className="py-2 pr-4">30% → 80%</td>
                  <td className="py-2">持续 {'<'} 10%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">grad_norm</td>
                  <td className="py-2 pr-4">0.1 - 10</td>
                  <td className="py-2">{'>'} 100 或 NaN</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">GPU Util</td>
                  <td className="py-2 pr-4">70-95%</td>
                  <td className="py-2">{'<'} 30%</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 font-mono">GPU Memory</td>
                  <td className="py-2 pr-4">{'<'} 30 GB</td>
                  <td className="py-2">{'>'} 31 GB (即将 OOM)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Config reference */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-medium text-gray-900 mb-3">配置速查</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 pr-4 font-medium text-gray-700">参数</th>
                  <th className="text-left py-2 pr-4 font-medium text-gray-700">Search-R1</th>
                  <th className="text-left py-2 pr-4 font-medium text-gray-700">ASearch</th>
                  <th className="text-left py-2 font-medium text-gray-700">作用</th>
                </tr>
              </thead>
              <tbody className="text-gray-600">
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">rollout.n</td>
                  <td className="py-2 pr-4">5</td>
                  <td className="py-2 pr-4">4</td>
                  <td className="py-2">GRPO 组大小</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">train_batch_size</td>
                  <td className="py-2 pr-4">256</td>
                  <td className="py-2 pr-4">128</td>
                  <td className="py-2">每步 prompt 数</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">max_response_length</td>
                  <td className="py-2 pr-4">3000</td>
                  <td className="py-2 pr-4">16384</td>
                  <td className="py-2">最大回复长度</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">max_model_len</td>
                  <td className="py-2 pr-4">15000</td>
                  <td className="py-2 pr-4">12000</td>
                  <td className="py-2">vLLM 上下文窗口</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">gpu_memory_utilization</td>
                  <td className="py-2 pr-4">0.65</td>
                  <td className="py-2 pr-4">0.55</td>
                  <td className="py-2">vLLM 显存占比</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-mono">lr</td>
                  <td className="py-2 pr-4">1e-6</td>
                  <td className="py-2 pr-4">5e-7</td>
                  <td className="py-2">学习率</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 font-mono">max_assistant_turns</td>
                  <td className="py-2 pr-4">4</td>
                  <td className="py-2 pr-4">50</td>
                  <td className="py-2">最大搜索轮数</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Evaluation targets */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 lg:col-span-2">
          <h3 className="font-medium text-gray-900 mb-3">评测目标 (Search-R1, Qwen2.5-3B-Instruct)</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 pr-4 font-medium text-gray-700">数据集</th>
                  <th className="text-right py-2 pr-4 font-medium text-gray-700">基线</th>
                  <th className="text-right py-2 pr-4 font-medium text-gray-700">目标</th>
                  <th className="text-right py-2 pr-4 font-medium text-gray-700">Abs. Gain</th>
                  <th className="text-right py-2 font-medium text-gray-700">Rel. Gain</th>
                </tr>
              </thead>
              <tbody className="text-gray-600">
                {[
                  { name: 'NQ†', baseline: 0.341, target: 0.464 },
                  { name: 'TriviaQA*', baseline: 0.545, target: 0.616 },
                  { name: 'PopQA*', baseline: 0.378, target: 0.424 },
                  { name: 'HotpotQA†', baseline: 0.324, target: 0.423 },
                  { name: '2Wiki*', baseline: 0.319, target: 0.398 },
                  { name: 'Musique*', baseline: 0.103, target: 0.181 },
                  { name: 'Bamboogle*', baseline: 0.264, target: 0.344 },
                ].map((d) => (
                  <tr key={d.name} className="border-b border-gray-100">
                    <td className="py-2 pr-4 font-medium">{d.name}</td>
                    <td className="py-2 pr-4 text-right font-mono">{d.baseline.toFixed(3)}</td>
                    <td className="py-2 pr-4 text-right font-mono text-blue-600">{d.target.toFixed(3)}</td>
                    <td className="py-2 pr-4 text-right font-mono text-green-600">+{(d.target - d.baseline).toFixed(4)}</td>
                    <td className="py-2 text-right font-mono text-green-600">+{((d.target - d.baseline) / d.baseline * 100).toFixed(1)}%</td>
                  </tr>
                ))}
                <tr className="font-medium">
                  <td className="py-2 pr-4">Avg</td>
                  <td className="py-2 pr-4 text-right font-mono">0.325</td>
                  <td className="py-2 pr-4 text-right font-mono text-blue-600">0.407</td>
                  <td className="py-2 pr-4 text-right font-mono text-green-600">+0.082</td>
                  <td className="py-2 text-right font-mono text-green-600">+25.3%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2">† In-domain · * Out-of-domain · 数据来源: RUC-NLPIR/FlashRAG_datasets</p>
        </div>
      </div>
    </section>
  )
}
