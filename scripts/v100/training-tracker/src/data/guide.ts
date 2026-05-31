// 训练指南数据 - 来自 TRAINING_MANUAL.md

export interface GuideSection {
  title: string
  type: 'text' | 'code' | 'table' | 'list' | 'tip' | 'warning'
  content?: string
  lang?: string
  headers?: string[]
  rows?: string[][]
  items?: string[]
}

export interface GuideChapter {
  id: string
  title: string
  icon: string
  sections: GuideSection[]
}

export const GUIDE_CHAPTERS: GuideChapter[] = [
  {
    id: 'prereq',
    title: '前置知识',
    icon: '🧠',
    sections: [
      {
        title: '核心概念',
        type: 'table',
        headers: ['概念', '一句话解释', '为什么重要'],
        rows: [
          ['GRPO', '无 Critic 的强化学习算法，对同一 prompt 采样多个回答，用组内相对奖励更新策略', '本项目的核心训练算法'],
          ['FSDP', 'PyTorch 的分布式数据并行，把模型参数分片到多卡', '让 3B/8B 模型装进 V100'],
          ['vLLM', '高性能 LLM 推理引擎，用 PagedAttention 管理 KV cache', '负责 rollout（生成训练轨迹）'],
          ['AgentLoop', 'verl 的多轮工具调用框架，模型生成 → 调工具 → 拼结果 → 继续生成', '搜索 Agent 的核心循环'],
          ['Rollout', '用当前策略模型生成一批轨迹（prompt → 多轮搜索 → 最终回答）', 'RL 训练的数据来源'],
          ['Reward', '对轨迹打分（答案正确性），用于计算优势函数', '驱动模型学习的信号'],
          ['KV Cache', '推理时缓存的 Key/Value 张量，避免重复计算', '决定能处理多长的上下文'],
          ['Offload', '把参数/优化器状态放到 CPU 内存，节省 GPU 显存', 'V100 显存有限，必须开启'],
        ],
      },
      {
        title: '训练流程总览',
        type: 'text',
        content:
          '环境安装 → 数据准备 → 检索服务 → 开始训练 → Rollout → 训练监控 → 模型评估 → 模型部署',
      },
      {
        title: '训练路线矩阵',
        type: 'table',
        headers: ['路线', '模型', '架构', '搜索轮数', '难度', '预计时间'],
        rows: [
          ['Search-R1（推荐新手）', 'Qwen2.5-3B-Instruct', 'Qwen2.5', '4 轮', '⭐⭐', '2-3 天'],
          ['Search-R1', 'Qwen3-1.7B', 'Qwen3', '4 轮', '⭐⭐', '1-2 天'],
          ['Search-R1', 'Qwen3-4B', 'Qwen3', '4 轮', '⭐⭐⭐', '2-3 天'],
          ['ASearch', 'Qwen3-4B', 'Qwen3', '30-50 轮', '⭐⭐⭐', '3-5 天'],
          ['ASearch（进阶）', 'Qwen3-8B', 'Qwen3', '30-50 轮', '⭐⭐⭐⭐', '5-8 天'],
          ['Search-R1（实验性）', 'Qwen3-30B-A3B', 'Qwen3 MoE', '4 轮', '⭐⭐⭐⭐⭐', '3-5 天'],
        ],
      },
      {
        title: '新手建议',
        type: 'tip',
        content: '先跑通 Search-R1，理解整个流程后再尝试 ASearch。',
      },
    ],
  },
  {
    id: 'qwen3-arch',
    title: 'Qwen3 架构特性',
    icon: '🧬',
    sections: [
      {
        title: 'Qwen3 vs Qwen2.5 对比',
        type: 'table',
        headers: ['特性', 'Qwen2.5', 'Qwen3', '训练影响'],
        rows: [
          ['QK LayerNorm', '❌', '✅ per-head RMSNorm', '训练更稳定，LR 可稍大'],
          ['Sliding Window', '全层固定', '交替层（全局+局部）', '长序列效率更好'],
          ['Thinking Mode', '❌', '✅ <think>...</think>', '需配置开关'],
          ['Tie Embeddings', '✅ (小模型)', '❌ (全部 untied)', '显存多占一份'],
          ['MoE 变体', '无', '30B-A3B', '需要 TP=4'],
        ],
      },
      {
        title: 'V100 资源需求',
        type: 'table',
        headers: ['模型', 'TP', '显存/GPU', '备注'],
        rows: [
          ['Qwen3-1.7B', '1', '~10-12 GB', '非常宽裕'],
          ['Qwen3-4B', '1', '~16-18 GB', '舒适'],
          ['Qwen3-4B (ASearch)', '1', '~20-24 GB', '较紧'],
          ['Qwen3-8B (ASearch)', '2', '~28-30 GB', '接近极限'],
          ['Qwen3-30B-A3B', '4', '~26-30 GB', '实验性'],
        ],
      },
      {
        title: 'Thinking Mode 决策树',
        type: 'list',
        items: [
          '序列短（Search-R1, 4轮）+ 小模型（1.7B/4B）→ 可开启',
          '序列长（ASearch, 50轮）→ 关闭',
          '显存紧张（8B TP=2 / MoE TP=4）→ 关闭',
          '开启方式: ENABLE_THINKING=true bash run_qwen3_xxx.sh',
        ],
      },
      {
        title: 'MoE 注意事项',
        type: 'warning',
        content:
          'Qwen3-30B-A3B 总参数 30B，激活参数仅 3B。V100 上需要 TP=4 加载权重，FSDP 切分 expert。标记为实验性，可能 OOM。如遇 OOM：降低 gpu_memory_utilization 到 0.40，减少 rollout_n 到 2。',
      },
    ],
  },
  {
    id: 'hardware',
    title: '硬件与环境要求',
    icon: '🖥️',
    sections: [
      {
        title: '最低硬件配置',
        type: 'code',
        lang: 'text',
        content:
          'GPU:     8× NVIDIA V100-32GB (sm_70)\nCPU:     32 核以上（Ray worker 需要）\n内存:    256 GB（CPU offload 需要大内存）\n磁盘:    500 GB 可用空间（模型 + 索引 + checkpoint）\n网络:    能访问 HuggingFace（下载模型和数据）',
      },
      {
        title: '软件要求',
        type: 'code',
        lang: 'text',
        content:
          'OS:      Ubuntu 20.04 / 22.04\nCUDA:    12.1 / 12.2 / 12.4\nDriver:  >= 535.x\nconda:   Miniconda 或 Anaconda\ngit:     >= 2.x',
      },
      {
        title: '确认硬件',
        type: 'code',
        lang: 'bash',
        content:
          '# 确认 GPU 型号和显存\nnvidia-smi\n\n# 确认 CUDA 版本\nnvcc --version\n\n# 确认内存（需要 256GB+）\nfree -h\n\n# 确认磁盘空间\ndf -h /home',
      },
    ],
  },
  {
    id: 'install',
    title: '第一阶段：环境安装',
    icon: '📦',
    sections: [
      {
        title: '克隆项目',
        type: 'code',
        lang: 'bash',
        content: 'cd ~\ngit clone https://github.com/NLPJCL/SearchAgent-Zero.git\ncd SearchAgent-Zero',
      },
      {
        title: '安装训练环境',
        type: 'code',
        lang: 'bash',
        content: 'bash scripts/v100/install_train_env.sh',
      },
      {
        title: '脚本会执行',
        type: 'list',
        items: [
          '创建 conda 环境 verl-v100（Python 3.11）',
          '安装 PyTorch 2.4.0 + CUDA 12.1',
          '安装 vLLM 0.6.6.post1（V100 兼容版）',
          '安装 xformers 0.0.28.post3（替代 FlashAttention-2）',
          '安装所有 Python 依赖',
          '以 editable 模式安装 SearchAgent-Zero',
        ],
      },
      {
        title: '预计耗时',
        type: 'tip',
        content: '10-15 分钟',
      },
      {
        title: '验证安装',
        type: 'code',
        lang: 'bash',
        content:
          'conda activate verl-v100\npython -c "import torch; print(torch.cuda.get_device_name(0))"\n# 应输出: Tesla V100-SXM2-32GB 或类似\n\npython -c "import vllm; print(vllm.__version__)"\n# 应输出: 0.6.6.post1',
      },
      {
        title: '安装检索服务环境',
        type: 'code',
        lang: 'bash',
        content: 'bash scripts/v100/install_retrieval_env.sh\n# 独立 conda 环境 retriever-v100，预计 5-10 分钟',
      },
      {
        title: '常见安装问题',
        type: 'table',
        headers: ['问题', '原因', '解决方案'],
        rows: [
          ['pip install vllm 编译失败', 'CUDA 版本不匹配', '确认 nvcc --version 是 12.x'],
          ['xformers 安装报错', 'PyTorch 版本不匹配', '确保先装 torch==2.4.0'],
          ['conda create 很慢', '默认源慢', '换清华源'],
          ['ray 启动失败', '端口冲突', 'ray stop --force 后重试'],
        ],
      },
    ],
  },
  {
    id: 'data',
    title: '第二阶段：数据准备',
    icon: '💾',
    sections: [
      {
        title: '一键下载和预处理',
        type: 'code',
        lang: 'bash',
        content: 'conda activate verl-v100\nbash scripts/v100/download_data.sh',
      },
      {
        title: '脚本会执行',
        type: 'list',
        items: [
          '下载 wiki-18 向量索引（~12GB）',
          '下载 wiki-18 语料库（~3GB）',
          '预处理 Search-R1 训练数据',
          '预处理 ASearcher 训练数据',
        ],
      },
      {
        title: '预计耗时',
        type: 'tip',
        content: '30-60 分钟（取决于网速）',
      },
      {
        title: '验证数据',
        type: 'code',
        lang: 'bash',
        content:
          "# 检查检索索引\nls -lh examples/search_agent_rl/local_dense_retriever/search_data/\n# 应该看到:\n#   e5_Flat.index  (~12GB)\n#   wiki-18.jsonl  (~8GB 解压后)\n\n# 检查训练数据\npython -c \"\nimport pandas as pd\ndf = pd.read_parquet('examples/search_agent_rl/search_r1_processed/train_search_r1.parquet')\nprint(f'Search-R1 训练集: {len(df)} 条')\nprint(f'字段: {list(df.columns)}')\n\"",
      },
      {
        title: '数据格式说明',
        type: 'code',
        lang: 'python',
        content:
          '{\n    "prompt": [{"role": "user", "content": "问题..."}],  # 对话格式\n    "data_source": "nq",                                   # 数据来源\n    "reward_model": {"ground_truth": "答案"},              # 奖励计算用\n    "extra_info": {"split": "train", ...}                  # 元信息\n}',
      },
    ],
  },
  {
    id: 'retrieval',
    title: '第三阶段：检索服务部署',
    icon: '🔍',
    sections: [
      {
        title: '启动检索服务',
        type: 'code',
        lang: 'bash',
        content:
          '# 新建 tmux 会话\ntmux new -s retriever\n\n# 激活检索环境\nconda activate retriever-v100\n\n# 启动（默认用 GPU 0，端口 8000）\nbash scripts/v100/start_retrieval_server.sh',
      },
      {
        title: '等待启动完成',
        type: 'tip',
        content: '看到 "INFO: Uvicorn running on http://0.0.0.0:8000" 即表示启动成功。',
      },
      {
        title: '验证检索服务',
        type: 'code',
        lang: 'bash',
        content:
          'curl -s http://127.0.0.1:8000/retrieve \\\n  -X POST \\\n  -H "Content-Type: application/json" \\\n  -d \'{"query": "What is the capital of France?", "topk": 3}\' | python -m json.tool',
      },
      {
        title: '检索服务架构',
        type: 'text',
        content:
          '训练进程 (8 GPU) → HTTP POST /retrieve → 检索服务 (1 GPU)：e5-base-v2 编码查询 → faiss-gpu 向量检索 → 返回 top-3 文档',
      },
      {
        title: '注意事项',
        type: 'warning',
        content:
          '检索服务和训练共享 GPU 0。训练脚本的 gpu_memory_utilization 已调低给检索服务留空间。如果 OOM，可以把检索服务移到 CPU 模式。',
      },
      {
        title: 'CPU 模式启动（备选）',
        type: 'code',
        lang: 'bash',
        content:
          'CUDA_VISIBLE_DEVICES="" python examples/search_agent_rl/local_dense_retriever/retrieval_server.py \\\n    --index_path ... --corpus_path ... --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2',
      },
    ],
  },
  {
    id: 'training',
    title: '第四阶段：训练启动',
    icon: '🚀',
    sections: [
      {
        title: 'Search-R1 训练（推荐新手首次运行）',
        type: 'code',
        lang: 'bash',
        content:
          '# 新终端\nconda activate verl-v100\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nexport WANDB_API_KEY=your_wandb_api_key_here  # 从 wandb.ai 获取\n\n# 启动训练\nbash scripts/v100/run_search_r1_v100.sh',
      },
      {
        title: '首次运行会',
        type: 'list',
        items: [
          '下载 Qwen2.5-3B-Instruct 模型（~6GB）',
          '启动 Ray cluster',
          '初始化 vLLM 推理引擎',
          '开始第一个 epoch 的 rollout',
        ],
      },
      {
        title: '正常启动标志',
        type: 'code',
        lang: 'text',
        content: '[INFO] Starting PPO training...\n[INFO] Rollout batch 1/X ...',
      },
      {
        title: 'ASearch 训练（进阶）',
        type: 'code',
        lang: 'bash',
        content:
          'conda activate verl-v100\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nexport WANDB_API_KEY=your_wandb_api_key_here\n\nbash scripts/v100/run_asearch_v100.sh',
      },
      {
        title: 'ASearch 注意',
        type: 'warning',
        content: 'ASearch 首次 rollout 会很慢（每条轨迹最多 30-50 轮搜索），这是正常的。',
      },
      {
        title: 'Search-R1 关键参数',
        type: 'table',
        headers: ['参数', '值', '含义'],
        rows: [
          ['data.train_batch_size', '256', '每步采样 256 个 prompt'],
          ['rollout.n', '5', '每个 prompt 生成 5 条轨迹'],
          ['max_response_length', '3000', '最大回答长度（token）'],
          ['max_assistant_turns', '4', '最多 4 轮搜索'],
          ['ppo_mini_batch_size', '128', 'PPO 更新的 mini-batch'],
          ['lr', '1e-6', '学习率'],
          ['total_epochs', '2', '训练 2 个 epoch'],
        ],
      },
      {
        title: 'ASearch 关键参数',
        type: 'table',
        headers: ['参数', '值', '含义'],
        rows: [
          ['data.train_batch_size', '128', '每步采样 128 个 prompt'],
          ['rollout.n', '4', '每个 prompt 生成 4 条轨迹'],
          ['max_response_length', '16384', '最大回答长度'],
          ['max_assistant_turns', '50', '最多 50 轮搜索'],
          ['turn_limit_schedule', '0:30,50:40,...', '逐步放开搜索轮数'],
          ['ppo_mini_batch_size', '32', 'PPO mini-batch'],
          ['lr', '5e-7', '更小的学习率（8B 模型）'],
        ],
      },
      {
        title: '恢复训练（从 checkpoint）',
        type: 'code',
        lang: 'bash',
        content:
          '# 找到最新 checkpoint\nls -lt output/qwen2.5-3b-instruct_searchr1_v100/\n\n# 从 checkpoint 恢复\nbash scripts/v100/run_search_r1_v100.sh \\\n    trainer.resume_from_checkpoint=output/qwen2.5-3b-instruct_searchr1_v100/step_250',
      },
    ],
  },
  {
    id: 'monitor',
    title: '第五阶段：训练监控',
    icon: '📊',
    sections: [
      {
        title: 'Weights & Biases 监控',
        type: 'text',
        content: '训练启动后，打开 https://wandb.ai 查看实时曲线。',
      },
      {
        title: '关键指标解读',
        type: 'table',
        headers: ['指标', '健康范围', '异常信号'],
        rows: [
          ['reward/mean', '持续上升', '长期不涨或下降 → 学习率/batch 问题'],
          ['reward/std', '逐渐减小', '持续很大 → 模型不稳定'],
          ['actor/loss', '小幅波动', '突然飙升 → loss spike'],
          ['actor/entropy', '缓慢下降', '骤降到 0 → 模式坍塌'],
          ['tool_call_success_rate/mean', '> 0.8', '< 0.5 → 工具调用格式学坏了'],
          ['tool_call_turn/mean', '逐步增加', '一直是 1 → 没学会多轮搜索'],
          ['abnormal_trajectory/*', '< 0.2', '> 0.5 → 需要调整过滤策略'],
        ],
      },
      {
        title: '终端日志监控',
        type: 'code',
        lang: 'bash',
        content:
          '# 实时查看日志\ntail -f logs/qwen2.5-3b-instruct_searchr1_v100.log\n\n# 过滤关键信息\ngrep -E "reward|loss|step" logs/qwen2.5-3b-instruct_searchr1_v100.log | tail -20',
      },
      {
        title: 'GPU 监控',
        type: 'code',
        lang: 'bash',
        content:
          '# 实时 GPU 使用率\nwatch -n 1 nvidia-smi\n\n# 更详细的监控（推荐）\nnvidia-smi dmon -s pucvmet -d 5\n# p=power, u=utilization, c=clock, v=violation, m=memory, e=ecc, t=temperature',
      },
      {
        title: 'Ray Dashboard',
        type: 'tip',
        content: 'Ray 会自动启动 dashboard，默认地址: http://localhost:8265，可以看到 worker 状态、任务队列、资源使用。',
      },
    ],
  },
  {
    id: 'eval',
    title: '第六阶段：模型评估',
    icon: '✅',
    sections: [
      {
        title: '评测数据集',
        type: 'table',
        headers: ['数据集', '子集名', '类型', '说明'],
        rows: [
          ['NQ†', 'nq', '单跳问答', 'In-domain'],
          ['TriviaQA*', 'triviaqa', '单跳问答', 'Out-of-domain'],
          ['PopQA*', 'popqa', '长尾知识', 'Out-of-domain'],
          ['HotpotQA†', 'hotpotqa', '多跳推理', 'In-domain'],
          ['2WikiMultihopQA*', '2wikimultihopqa', '多跳推理', 'Out-of-domain'],
          ['MuSiQue*', 'musique', '多跳推理', 'Out-of-domain'],
          ['Bamboogle*', 'bamboogle', '组合推理', 'Out-of-domain'],
        ],
      },
      {
        title: '说明',
        type: 'tip',
        content: '† = In-domain（训练数据中包含同源训练集），* = Out-of-domain（仅用于评测）。评估数据通过 datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "<子集名>") 自动加载。',
      },
      {
        title: '运行评估',
        type: 'code',
        lang: 'bash',
        content:
          'conda activate verl-v100\n\n# 评估 Search-R1 训练的模型\nbash run_qwen2.5_3b_instruct_search_multiturn_SearchR1_eval.sh',
      },
      {
        title: '评测目标',
        type: 'table',
        headers: ['数据集', 'Baseline', '目标', 'Rel. Gain'],
        rows: [
          ['NQ', '0.341', '0.464', '+36.1%'],
          ['TriviaQA', '0.545', '0.616', '+13.1%'],
          ['PopQA', '0.378', '0.424', '+12.1%'],
          ['HotpotQA', '0.324', '0.423', '+30.4%'],
          ['2Wiki', '0.319', '0.398', '+24.7%'],
          ['Musique', '0.103', '0.181', '+75.5%'],
          ['Bamboogle', '0.264', '0.344', '+30.3%'],
          ['Avg', '0.325', '0.407', '+25.3%'],
        ],
      },
      {
        title: '评测通过标准',
        type: 'table',
        headers: ['指标', '最低要求', '优秀'],
        rows: [
          ['平均分', '≥ 0.38', '≥ 0.40'],
          ['所有数据集正向提升', '✅', '✅'],
          ['Musique（最难）', '≥ 0.14', '≥ 0.18'],
          ['NQ（in-domain）', '≥ 0.42', '≥ 0.46'],
        ],
      },
      {
        title: '手动测试模型',
        type: 'code',
        lang: 'python',
        content:
          'from vllm import LLM, SamplingParams\n\nmodel = LLM(\n    model="output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint",\n    dtype="float16",\n    gpu_memory_utilization=0.8,\n)\n\nprompt = "What year was the Eiffel Tower completed?"\noutput = model.generate([prompt], SamplingParams(temperature=0.7, max_tokens=2048))\nprint(output[0].outputs[0].text)',
      },
    ],
  },
  {
    id: 'export',
    title: '第七阶段：模型导出与部署',
    icon: '📤',
    sections: [
      {
        title: '导出 HuggingFace 格式',
        type: 'text',
        content: '训练产出的 checkpoint 已经是 HuggingFace 格式，可以直接加载。',
      },
      {
        title: '加载模型',
        type: 'code',
        lang: 'python',
        content:
          'from transformers import AutoModelForCausalLM, AutoTokenizer\n\nmodel = AutoModelForCausalLM.from_pretrained(\n    "output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint",\n    torch_dtype="auto",\n    device_map="auto",\n)\ntokenizer = AutoTokenizer.from_pretrained(\n    "output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint"\n)',
      },
      {
        title: '上传到 HuggingFace Hub',
        type: 'code',
        lang: 'bash',
        content:
          'pip install huggingface_hub\nhuggingface-cli login\n\npython -c "\nfrom huggingface_hub import HfApi\napi = HfApi()\napi.upload_folder(\n    folder_path=\'output/qwen2.5-3b-instruct_searchr1_v100/best_checkpoint\',\n    repo_id=\'your-username/searchagent-zero-3b\',\n    repo_type=\'model\',\n)\n"',
      },
    ],
  },
  {
    id: 'perf',
    title: '性能分析与问题定位',
    icon: '⚡',
    sections: [
      {
        title: '参考文档',
        type: 'tip',
        content: '详见 PERFORMANCE_GUIDE.md，包含完整的性能分析方法论。',
      },
      {
        title: '快速诊断命令',
        type: 'code',
        lang: 'bash',
        content:
          '# 检查 GPU 利用率\nnvidia-smi dmon -s pucvmet -d 5\n\n# 检查 CPU 瓶颈（offload 相关）\nhtop\n\n# 检查磁盘 IO（checkpoint 写入）\niostat -x 1 5\n\n# 检查网络（检索服务延迟）\ncurl -w "\\n%{time_total}s\\n" -s http://127.0.0.1:8000/retrieve -X POST -H "Content-Type: application/json" -d \'{"query": "test", "topk": 3}\'',
      },
    ],
  },
  {
    id: 'faq',
    title: '常见问题 FAQ',
    icon: '❓',
    sections: [
      {
        title: '训练多久能看到效果？',
        type: 'text',
        content: 'Search-R1 通常在 50-100 步后 reward 开始明显上升。如果 200 步后还没动静，检查检索服务是否正常。',
      },
      {
        title: '可以中途停止再恢复吗？',
        type: 'text',
        content: '可以。Ctrl+C 停止后，用 trainer.resume_from_checkpoint=<path> 恢复。checkpoint 默认每 250 步保存一次。',
      },
      {
        title: '显存不够怎么办？',
        type: 'list',
        items: [
          '降低 gpu_memory_utilization（0.65 → 0.55）',
          '降低 rollout.n（5 → 3）',
          '降低 max_model_len（15000 → 12000）',
          '降低 train_batch_size',
          '确认 gradient_checkpointing 已开启',
        ],
      },
      {
        title: '训练速度太慢？',
        type: 'text',
        content: 'V100 上 Search-R1 每步约 3-5 分钟是正常的。如果超过 10 分钟/步，检查：1) 检索服务是否响应慢；2) CPU offload 是否导致瓶颈；3) 是否有 GPU 空闲。',
      },
      {
        title: '模型生成乱码/不调用工具？',
        type: 'text',
        content: '训练初期正常。如果持续 100+ 步还是这样：1) 检查 tool_call_success_rate 指标；2) 确认 attn_implementation=sdpa 设置正确；3) 尝试降低学习率。',
      },
      {
        title: 'Ray 报错 "No available node"',
        type: 'code',
        lang: 'bash',
        content: 'ray stop --force\n# 等 10 秒\nray start --head --num-gpus=8\n# 然后重新启动训练',
      },
    ],
  },
]
