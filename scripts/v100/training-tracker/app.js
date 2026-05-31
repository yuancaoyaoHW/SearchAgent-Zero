const { createApp, ref, computed, watch, onMounted, onUnmounted, nextTick } = Vue;

createApp({
    setup() {
        // --- State ---
        const currentTab = ref('live');
        const selectedSymptom = ref('');
        let rewardChartInstance = null;
        let toolChartInstance = null;
        let liveRewardChartInstance = null;
        let gpuHistoryChartInstance = null;

        const tabs = [
            { id: 'live', label: '实时日志', icon: '📡' },
            { id: 'checklist', label: '检查清单', icon: '📋' },
            { id: 'log', label: '训练日志', icon: '📝' },
            { id: 'metrics', label: '训练曲线', icon: '📊' },
            { id: 'gpu', label: 'GPU 监控', icon: '🖥️' },
            { id: 'troubleshoot', label: '问题诊断', icon: '🔧' },
            { id: 'reference', label: '快速参考', icon: '📚' },
        ];

        // =============================================
        // Feature 1: Live Log Stream + Auto Metrics
        // =============================================
        const logWsUrl = ref('ws://localhost:8766');
        const logConnected = ref(false);
        const logAutoScroll = ref(true);
        const logLines = ref([]);
        const liveMetrics = ref([]);
        const alerts = ref([]);
        const alertCount = computed(() => alerts.value.length);
        const logStreamEl = ref(null);
        let logWs = null;
        let logReconnectTimer = null;
        const MAX_LOG_LINES = 500;
        const MAX_LIVE_METRICS = 200;

        const latestMetric = computed(() => {
            if (liveMetrics.value.length === 0) return {};
            return liveMetrics.value[liveMetrics.value.length - 1];
        });

        function connectLogStream() {
            if (logWs) logWs.close();
            try {
                logWs = new WebSocket(logWsUrl.value);
                logWs.onopen = () => {
                    logConnected.value = true;
                    clearTimeout(logReconnectTimer);
                };
                logWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleLogMessage(data);
                };
                logWs.onclose = () => {
                    logConnected.value = false;
                    scheduleLogReconnect();
                };
                logWs.onerror = () => {
                    logConnected.value = false;
                };
            } catch (e) {
                logConnected.value = false;
            }
        }

        function disconnectLogStream() {
            clearTimeout(logReconnectTimer);
            if (logWs) { logWs.close(); logWs = null; }
            logConnected.value = false;
        }

        function scheduleLogReconnect() {
            if (logReconnectTimer) return;
            logReconnectTimer = setTimeout(() => {
                logReconnectTimer = null;
                if (!logConnected.value && logWsUrl.value) {
                    connectLogStream();
                }
            }, 5000);
        }

        function handleLogMessage(data) {
            if (data.type === 'log') {
                const time = data.timestamp ? new Date(data.timestamp).toLocaleTimeString('zh-CN') : '';
                logLines.value.push({ line: data.line, time });
                if (logLines.value.length > MAX_LOG_LINES) logLines.value.shift();
                checkAlerts(data.line);
                if (logAutoScroll.value) {
                    nextTick(() => {
                        const el = logStreamEl.value;
                        if (el) el.scrollTop = el.scrollHeight;
                    });
                }
            } else if (data.type === 'metric') {
                const m = { ...data };
                delete m.type;
                liveMetrics.value.push(m);
                if (liveMetrics.value.length > MAX_LIVE_METRICS) liveMetrics.value.shift();
                nextTick(() => updateLiveChart());
            } else if (data.type === 'history') {
                liveMetrics.value = data.metrics || [];
                nextTick(() => updateLiveChart());
            }
        }

        function checkAlerts(line) {
            const lower = line.toLowerCase();
            if (lower.includes('oom') || lower.includes('out of memory')) {
                addAlert('danger', 'OOM: ' + line.substring(0, 120));
            } else if (lower.includes('nan') && !lower.includes('channel') && !lower.includes('nan_to_num')) {
                addAlert('danger', 'NaN: ' + line.substring(0, 120));
            } else if (lower.includes('error') && !lower.includes('error_rate') && !lower.includes('errors=')) {
                addAlert('warning', 'Error: ' + line.substring(0, 120));
            }
        }

        function addAlert(level, message) {
            alerts.value.unshift({ level, message, time: new Date().toLocaleTimeString('zh-CN') });
            if (alerts.value.length > 50) alerts.value.pop();
            if (Notification.permission === 'granted') {
                new Notification('Training Alert', { body: message });
            }
        }

        function dismissAlert(idx) { alerts.value.splice(idx, 1); }

        function requestNotificationPermission() {
            if ('Notification' in window && Notification.permission === 'default') {
                Notification.requestPermission();
            }
        }

        function logLineClass(line) {
            if (!line) return '';
            const l = line.toLowerCase();
            if (l.includes('error') || l.includes('oom') || l.includes('nan')) return 'log-error';
            if (l.includes('warn')) return 'log-warn';
            if (l.includes('step') || l.includes('reward')) return 'log-metric';
            return '';
        }

        function updateLiveChart() {
            const canvas = document.getElementById('liveRewardChart');
            if (!canvas || liveMetrics.value.length < 2) return;
            if (liveRewardChartInstance) liveRewardChartInstance.destroy();

            const data = liveMetrics.value.filter(m => m.step != null);
            liveRewardChartInstance = new Chart(canvas, {
                type: 'line',
                data: {
                    labels: data.map(m => 'Step ' + m.step),
                    datasets: [
                        {
                            label: 'Reward',
                            data: data.map(m => m.reward),
                            borderColor: '#2563eb',
                            tension: 0.3,
                            yAxisID: 'y',
                        },
                        {
                            label: 'Tool Success %',
                            data: data.map(m => m.tool_success ? m.tool_success * 100 : null),
                            borderColor: '#16a34a',
                            tension: 0.3,
                            yAxisID: 'y1',
                        },
                    ],
                },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: { title: { display: true, text: '实时训练指标' } },
                    scales: {
                        y: { position: 'left', title: { display: true, text: 'Reward' } },
                        y1: { position: 'right', min: 0, max: 100, title: { display: true, text: 'Tool %' }, grid: { drawOnChartArea: false } },
                    },
                },
            });
        }

        // =============================================
        // Feature 3: Retrieval Service Health Probe
        // =============================================
        const retrievalUrl = ref('http://127.0.0.1:8000/health');
        const retrievalStatus = ref('offline');
        const retrievalStatusText = computed(() => {
            if (retrievalStatus.value === 'online') return '在线';
            if (retrievalStatus.value === 'checking') return '检测中...';
            return '离线';
        });
        let retrievalProbeTimer = null;

        async function checkRetrievalHealth() {
            retrievalStatus.value = 'checking';
            try {
                const controller = new AbortController();
                const timeout = setTimeout(() => controller.abort(), 3000);
                const resp = await fetch(retrievalUrl.value, { signal: controller.signal, mode: 'no-cors' });
                clearTimeout(timeout);
                retrievalStatus.value = 'online';
            } catch (e) {
                try {
                    const controller = new AbortController();
                    const timeout = setTimeout(() => controller.abort(), 3000);
                    const base = retrievalUrl.value.replace('/health', '');
                    const resp = await fetch(base + '/retrieve', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: 'test', top_k: 1 }),
                        signal: controller.signal,
                    });
                    clearTimeout(timeout);
                    retrievalStatus.value = resp.ok ? 'online' : 'offline';
                } catch (e2) {
                    retrievalStatus.value = 'offline';
                }
            }
        }

        function startRetrievalProbe() {
            checkRetrievalHealth();
            retrievalProbeTimer = setInterval(checkRetrievalHealth, 30000);
        }

        // =============================================
        // Checklist
        // =============================================
        const phases = ref([
            {
                id: 'env', icon: '🖥️', title: '阶段一：环境安装',
                items: [
                    { id: 'e1', text: '确认 GPU 型号和显存 (nvidia-smi)', hint: '需要 8\u00d7V100-32GB', done: false },
                    { id: 'e2', text: '确认 CUDA 版本 (nvcc --version)', hint: '需要 CUDA 12.x', done: false },
                    { id: 'e3', text: '确认内存 >= 256GB (free -h)', hint: 'CPU offload 需要大内存', done: false },
                    { id: 'e4', text: '确认磁盘空间 >= 500GB (df -h)', hint: '模型+索引+checkpoint', done: false },
                    { id: 'e5', text: '克隆 SearchAgent-Zero 仓库', hint: 'git clone ...', done: false },
                    { id: 'e6', text: '运行 install_train_env.sh', hint: '约 10-15 分钟', done: false },
                    { id: 'e7', text: '验证 PyTorch + vLLM 安装', hint: 'python -c "import vllm"', done: false },
                    { id: 'e8', text: '运行 install_retrieval_env.sh', hint: '约 5 分钟', done: false },
                ]
            },
            {
                id: 'data', icon: '📦', title: '阶段二：数据准备',
                items: [
                    { id: 'd1', text: '运行 download_data.sh', hint: '下载索引+语料+预处理，约 30-60 分钟', done: false },
                    { id: 'd2', text: '验证检索索引文件 (e5_Flat.index ~12GB)', hint: 'ls -lh ...search_data/', done: false },
                    { id: 'd3', text: '验证语料文件 (wiki-18.jsonl)', hint: '解压后约 8GB', done: false },
                    { id: 'd4', text: '验证 Search-R1 训练数据 (train_search_r1.parquet)', hint: '', done: false },
                    { id: 'd5', text: '验证 ASearcher 训练数据 (ASearcher_train.parquet)', hint: '如果跑 ASearch', done: false },
                ]
            },
            {
                id: 'retrieval', icon: '🔍', title: '阶段三：检索服务',
                items: [
                    { id: 'r1', text: '在 tmux 中启动检索服务', hint: 'tmux new -s retriever', done: false },
                    { id: 'r2', text: '等待 "Uvicorn running" 输出', hint: '加载索引需要几分钟', done: false },
                    { id: 'r3', text: 'curl 测试检索服务正常', hint: 'curl http://127.0.0.1:8000/retrieve ...', done: false },
                ]
            },
            {
                id: 'train', icon: '🚀', title: '阶段四：训练启动',
                items: [
                    { id: 't1', text: '设置 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', hint: '', done: false },
                    { id: 't2', text: '设置 WANDB_API_KEY', hint: '可选，用于在线监控', done: false },
                    { id: 't3', text: '启动训练脚本', hint: 'bash scripts/v100/run_search_r1_v100.sh', done: false },
                    { id: 't4', text: '确认训练正常启动（无报错）', hint: '观察前 5 分钟日志', done: false },
                    { id: 't5', text: '确认第一个 step 完成', hint: '日志中出现 step 1 的 reward', done: false },
                ]
            },
            {
                id: 'monitor', icon: '📊', title: '阶段五：训练监控',
                items: [
                    { id: 'm1', text: '检查 GPU 利用率 > 70%', hint: 'nvidia-smi', done: false },
                    { id: 'm2', text: '检查 reward 开始上升', hint: '通常 50-100 步后', done: false },
                    { id: 'm3', text: '检查 tool_call_success_rate > 30%', hint: '日志或 wandb', done: false },
                    { id: 'm4', text: '确认无 OOM 或 NaN', hint: 'grep "OOM\\|NaN" logs/', done: false },
                    { id: 'm5', text: '第一个 checkpoint 保存成功', hint: '默认每 250 步保存', done: false },
                ]
            },
            {
                id: 'eval', icon: '\u2705', title: '阶段六：评估与导出',
                items: [
                    { id: 'v1', text: '训练完成（2 epochs）', hint: '', done: false },
                    { id: 'v2', text: '运行评估脚本', hint: 'run_*_eval.sh', done: false },
                    { id: 'v3', text: '对比 baseline 分数', hint: 'Search-R1 baseline: 0.325', done: false },
                    { id: 'v4', text: '导出最终模型', hint: 'output/*/global_step_*/actor/', done: false },
                ]
            },
        ]);

        // =============================================
        // Training Logs (manual entry)
        // =============================================
        const logs = ref([]);
        const newLog = ref({
            name: '', recipe: 'search_r1', step: null,
            reward: null, toolSuccess: null, notes: ''
        });

        function addLog() {
            if (!newLog.value.name) return;
            logs.value.unshift({
                ...newLog.value,
                time: new Date().toLocaleString('zh-CN'),
            });
            newLog.value = { name: newLog.value.name, recipe: newLog.value.recipe, step: null, reward: null, toolSuccess: null, notes: '' };
            saveState();
            nextTick(() => updateCharts());
        }

        function removeLog(idx) {
            logs.value.splice(idx, 1);
            saveState();
            nextTick(() => updateCharts());
        }

        function rewardClass(reward) {
            if (reward == null) return '';
            if (reward > 0.3) return 'reward-good';
            if (reward > 0.1) return 'reward-ok';
            return 'reward-low';
        }

        // =============================================
        // =============================================
        // Feature 4: Multi-experiment Comparison Charts
        // =============================================
        const selectedExperiments = ref([]);
        const availableExperiments = computed(() => {
            const names = new Set(logs.value.map(l => l.name).filter(Boolean));
            return [...names];
        });

        function toggleExperiment(name) {
            const idx = selectedExperiments.value.indexOf(name);
            if (idx >= 0) {
                selectedExperiments.value.splice(idx, 1);
            } else {
                selectedExperiments.value.push(name);
            }
            nextTick(() => updateCharts());
        }

        function updateCharts() {
            const experiments = selectedExperiments.value.length > 0
                ? selectedExperiments.value
                : availableExperiments.value;

            if (logs.value.length < 2) return;

            const colors = [
                '#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed',
                '#0891b2', '#be185d', '#65a30d', '#c2410c', '#4f46e5',
            ];

            // Reward chart
            const rewardCanvas = document.getElementById('rewardChart');
            if (rewardCanvas) {
                if (rewardChartInstance) rewardChartInstance.destroy();
                const datasets = experiments.map((name, i) => {
                    const expLogs = logs.value
                        .filter(l => l.name === name && l.step != null)
                        .sort((a, b) => a.step - b.step);
                    return {
                        label: name,
                        data: expLogs.map(l => ({ x: l.step, y: l.reward })),
                        borderColor: colors[i % colors.length],
                        tension: 0.3,
                        fill: false,
                    };
                }).filter(ds => ds.data.length > 0);

                if (datasets.length > 0) {
                    rewardChartInstance = new Chart(rewardCanvas, {
                        type: 'line',
                        data: { datasets },
                        options: {
                            responsive: true,
                            plugins: { title: { display: true, text: 'Reward (mean) by Step' } },
                            scales: {
                                x: { type: 'linear', title: { display: true, text: 'Step' } },
                                y: { title: { display: true, text: 'Reward' } },
                            },
                        },
                    });
                }
            }

            // Tool success chart
            const toolCanvas = document.getElementById('toolChart');
            if (toolCanvas) {
                if (toolChartInstance) toolChartInstance.destroy();
                const datasets = experiments.map((name, i) => {
                    const expLogs = logs.value
                        .filter(l => l.name === name && l.step != null && l.toolSuccess != null)
                        .sort((a, b) => a.step - b.step);
                    return {
                        label: name,
                        data: expLogs.map(l => ({ x: l.step, y: l.toolSuccess * 100 })),
                        borderColor: colors[i % colors.length],
                        tension: 0.3,
                        fill: false,
                    };
                }).filter(ds => ds.data.length > 0);

                if (datasets.length > 0) {
                    toolChartInstance = new Chart(toolCanvas, {
                        type: 'line',
                        data: { datasets },
                        options: {
                            responsive: true,
                            plugins: { title: { display: true, text: 'Tool Call Success Rate (%)' } },
                            scales: {
                                x: { type: 'linear', title: { display: true, text: 'Step' } },
                                y: { title: { display: true, text: '%' }, min: 0, max: 100 },
                            },
                        },
                    });
                }
            }
        }

        function importMetricsJson() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = (e) => {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (ev) => {
                    try {
                        const data = JSON.parse(ev.target.result);
                        const metrics = Array.isArray(data) ? data : (data.metrics || []);
                        metrics.forEach(m => {
                            logs.value.push({
                                name: m.experiment || 'imported',
                                recipe: m.recipe || 'search_r1',
                                step: m.step,
                                reward: m.reward || m.reward_mean,
                                toolSuccess: m.tool_success || m.tool_call_success_rate,
                                notes: 'imported from JSON',
                                time: m.timestamp || new Date().toLocaleString('zh-CN'),
                            });
                        });
                        saveState();
                        nextTick(() => updateCharts());
                    } catch (err) {
                        alert('JSON parse error: ' + err.message);
                    }
                };
                reader.readAsText(file);
            };
            input.click();
        }

        // =============================================
        // Feature 2: GPU History Chart + Feature 5: Reconnect
        // =============================================
        const gpuWsUrl = ref('ws://localhost:8765');
        const gpuConnected = ref(false);
        const gpuData = ref([]);
        const gpuHistory = ref([]);
        const MAX_GPU_HISTORY = 120;
        let gpuWs = null;
        let gpuReconnectTimer = null;

        function connectGpuMonitor() {
            if (gpuWs) gpuWs.close();
            try {
                gpuWs = new WebSocket(gpuWsUrl.value);
                gpuWs.onopen = () => {
                    gpuConnected.value = true;
                    clearTimeout(gpuReconnectTimer);
                };
                gpuWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.gpus) {
                        gpuData.value = data.gpus;
                        gpuHistory.value.push({
                            timestamp: data.timestamp || new Date().toISOString(),
                            gpus: data.gpus,
                        });
                        if (gpuHistory.value.length > MAX_GPU_HISTORY) gpuHistory.value.shift();
                        nextTick(() => updateGpuHistoryChart());
                    }
                };
                gpuWs.onclose = () => {
                    gpuConnected.value = false;
                    scheduleGpuReconnect();
                };
                gpuWs.onerror = () => {
                    gpuConnected.value = false;
                };
            } catch (e) {
                gpuConnected.value = false;
            }
        }

        function disconnectGpuMonitor() {
            clearTimeout(gpuReconnectTimer);
            if (gpuWs) { gpuWs.close(); gpuWs = null; }
            gpuConnected.value = false;
        }

        function scheduleGpuReconnect() {
            if (gpuReconnectTimer) return;
            gpuReconnectTimer = setTimeout(() => {
                gpuReconnectTimer = null;
                if (!gpuConnected.value && gpuWsUrl.value) {
                    connectGpuMonitor();
                }
            }, 5000);
        }

        function updateGpuHistoryChart() {
            const canvas = document.getElementById('gpuHistoryChart');
            if (!canvas || gpuHistory.value.length < 2) return;

            if (gpuHistoryChartInstance) gpuHistoryChartInstance.destroy();

            const numGpus = gpuHistory.value[0].gpus.length;
            const colors = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2', '#be185d', '#65a30d'];
            const labels = gpuHistory.value.map(h => {
                const d = new Date(h.timestamp);
                return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            });

            const datasets = [];
            for (let i = 0; i < numGpus; i++) {
                datasets.push({
                    label: 'GPU ' + i + ' Util%',
                    data: gpuHistory.value.map(h => h.gpus[i] ? h.gpus[i].gpu_util : 0),
                    borderColor: colors[i % colors.length],
                    backgroundColor: colors[i % colors.length] + '15',
                    tension: 0.3,
                    fill: true,
                    pointRadius: 0,
                });
            }

            gpuHistoryChartInstance = new Chart(canvas, {
                type: 'line',
                data: { labels, datasets },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: {
                        title: { display: true, text: 'GPU 利用率历史 (%)' },
                        legend: { position: 'bottom' },
                    },
                    scales: {
                        y: { min: 0, max: 100, title: { display: true, text: '%' } },
                        x: { ticks: { maxRotation: 0 } },
                    },
                },
            });
        }

        function gpuUtilClass(util) {
            if (util >= 70) return 'fill-good';
            if (util >= 30) return 'fill-ok';
            return 'fill-low';
        }

        function memUtilClass(util) {
            if (util >= 95) return 'fill-danger';
            if (util >= 80) return 'fill-warning';
            return 'fill-good';
        }

        // =============================================
        // Troubleshooting
        // =============================================
        const symptoms = [
            { id: 'oom', label: 'GPU OOM (显存不足)' },
            { id: 'nan', label: '出现 NaN / Inf' },
            { id: 'reward_zero', label: 'Reward 一直为 0' },
            { id: 'tool_fail', label: 'Tool call 成功率极低' },
            { id: 'slow', label: '训练速度很慢' },
            { id: 'retrieval_down', label: '检索服务无响应' },
            { id: 'kl_explode', label: 'KL divergence 爆炸' },
        ];

        const diagnoses = {
            oom: {
                causes: ['max_model_len 过大', 'gpu_memory_utilization 过高', 'batch_size 过大', '梯度累积占用'],
                fixes: ['降低 gpu_memory_utilization 到 0.55', '减小 max_model_len (V100 建议 \u2264 15000)', '减小 train_batch_size', '启用 gradient_checkpointing'],
            },
            nan: {
                causes: ['学习率过大', '梯度爆炸', 'reward 计算异常', '数据中有异常样本'],
                fixes: ['降低学习率到 5e-7', '添加 gradient clipping (max_grad_norm=1.0)', '检查 reward function 是否有除零', '检查训练数据是否有空样本'],
            },
            reward_zero: {
                causes: ['tool call 格式不对', 'reward function 逻辑错误', '模型未学会搜索', 'prompt 模板问题'],
                fixes: ['检查 system prompt 中的 tool 格式说明', '手动测试 reward function', '增大 rollout.n 提供更多探索', '检查 chat_template 是否正确'],
            },
            tool_fail: {
                causes: ['检索服务未启动', '请求超时', 'tool call 格式解析失败', '网络问题'],
                fixes: ['确认检索服务运行中 (curl 测试)', '增大 tool_timeout', '检查 tool_parser 逻辑', '检查防火墙/端口'],
            },
            slow: {
                causes: ['CPU offload 过多', 'vLLM 未充分利用 GPU', '数据加载瓶颈', '检索服务响应慢'],
                fixes: ['检查 GPU 利用率，调整 gpu_memory_utilization', '增大 vLLM tensor_parallel_size', '使用更快的数据加载 (num_workers)', '优化检索服务或增加副本'],
            },
            retrieval_down: {
                causes: ['服务进程崩溃', '端口被占用', '索引文件损坏', '内存不足'],
                fixes: ['重启检索服务 (检查 tmux session)', '检查端口占用 (lsof -i :8000)', '重新下载索引文件', '确认系统内存充足 (free -h)'],
            },
            kl_explode: {
                causes: ['学习率过大', 'reward scale 过大', 'kl_coef 过小', '训练不稳定'],
                fixes: ['降低学习率', '归一化 reward', '增大 kl_coef (0.01 -> 0.05)', '减小 batch_size 稳定训练'],
            },
        };

        const currentDiagnosis = computed(() => {
            if (!selectedSymptom.value) return null;
            return diagnoses[selectedSymptom.value] || null;
        });

        // Quick reference commands
        const commands = {
            gpu: 'nvidia-smi -l 5',
            memory: 'free -h && df -h',
            process: 'ps aux | grep python | grep -v grep',
            logs: 'tail -f output/*/logs/training.log',
            kill: 'pkill -f "python.*train"',
            retrieval: "curl -s http://127.0.0.1:8000/retrieve -X POST -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"top_k\":3}'",
        };

        // =============================================
        // Persistence
        // =============================================
        function saveState() {
            const state = {
                phases: phases.value,
                logs: logs.value,
                gpuWsUrl: gpuWsUrl.value,
                logWsUrl: logWsUrl.value,
                retrievalUrl: retrievalUrl.value,
            };
            localStorage.setItem('training-tracker-state', JSON.stringify(state));
        }

        function loadState() {
            try {
                const raw = localStorage.getItem('training-tracker-state');
                if (!raw) return;
                const state = JSON.parse(raw);
                if (state.phases) {
                    state.phases.forEach(savedPhase => {
                        const phase = phases.value.find(p => p.id === savedPhase.id);
                        if (phase) {
                            savedPhase.items.forEach(savedItem => {
                                const item = phase.items.find(i => i.id === savedItem.id);
                                if (item) item.done = savedItem.done;
                            });
                        }
                    });
                }
                if (state.logs) logs.value = state.logs;
                if (state.gpuWsUrl) gpuWsUrl.value = state.gpuWsUrl;
                if (state.logWsUrl) logWsUrl.value = state.logWsUrl;
                if (state.retrievalUrl) retrievalUrl.value = state.retrievalUrl;
            } catch (e) {
                console.warn('Failed to load state:', e);
            }
        }

        function exportData() {
            const data = {
                phases: phases.value,
                logs: logs.value,
                liveMetrics: liveMetrics.value,
                exportTime: new Date().toISOString(),
            };
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'training-tracker-' + new Date().toISOString().slice(0, 10) + '.json';
            a.click();
            URL.revokeObjectURL(url);
        }

        function importData() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = (e) => {
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (ev) => {
                    try {
                        const data = JSON.parse(ev.target.result);
                        if (data.phases) {
                            data.phases.forEach(savedPhase => {
                                const phase = phases.value.find(p => p.id === savedPhase.id);
                                if (phase) {
                                    savedPhase.items.forEach(savedItem => {
                                        const item = phase.items.find(i => i.id === savedItem.id);
                                        if (item) item.done = savedItem.done;
                                    });
                                }
                            });
                        }
                        if (data.logs) logs.value = data.logs;
                        if (data.liveMetrics) liveMetrics.value = data.liveMetrics;
                        saveState();
                        nextTick(() => updateCharts());
                    } catch (err) {
                        alert('导入失败: ' + err.message);
                    }
                };
                reader.readAsText(file);
            };
            input.click();
        }

        function clearAll() {
            if (!confirm('确定清空所有数据？此操作不可恢复。')) return;
            phases.value.forEach(p => p.items.forEach(i => i.done = false));
            logs.value = [];
            liveMetrics.value = [];
            alerts.value = [];
            gpuHistory.value = [];
            localStorage.removeItem('training-tracker-state');
        }

        // --- Helpers ---
        function isPhaseComplete(phase) {
            return phase.items.every(i => i.done);
        }

        function phaseProgress(phase) {
            const done = phase.items.filter(i => i.done).length;
            return done + '/' + phase.items.length;
        }

        // --- Lifecycle ---
        onMounted(() => {
            loadState();
            nextTick(() => updateCharts());
            startRetrievalProbe();
            requestNotificationPermission();
        });

        onUnmounted(() => {
            if (retrievalProbeTimer) clearInterval(retrievalProbeTimer);
            if (logReconnectTimer) clearTimeout(logReconnectTimer);
            if (gpuReconnectTimer) clearTimeout(gpuReconnectTimer);
        });

        watch(currentTab, (val) => {
            if (val === 'metrics') nextTick(() => updateCharts());
            if (val === 'gpu') nextTick(() => updateGpuHistoryChart());
            if (val === 'live') nextTick(() => updateLiveChart());
        });

        return {
            currentTab, tabs, phases, logs, newLog, selectedSymptom,
            symptoms, currentDiagnosis, commands,
            addLog, removeLog, rewardClass,
            isPhaseComplete, phaseProgress,
            exportData, importData, importMetricsJson, clearAll, saveState,
            // Live log stream
            logWsUrl, logConnected, logAutoScroll, logLines, liveMetrics,
            latestMetric, alerts, alertCount, logStreamEl,
            connectLogStream, disconnectLogStream, dismissAlert, logLineClass,
            // GPU monitor
            gpuWsUrl, gpuConnected, gpuData, gpuHistory,
            connectGpuMonitor, disconnectGpuMonitor,
            gpuUtilClass, memUtilClass,
            // Retrieval health
            retrievalUrl, retrievalStatus, retrievalStatusText,
            // Multi-experiment
            selectedExperiments, availableExperiments, toggleExperiment,
        };
    }
}).mount('#app');
