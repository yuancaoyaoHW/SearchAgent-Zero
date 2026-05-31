# Rollout 数据解读 & 自定义扩展指南

> 如何查看模型生成的轨迹、如何自定义 Reward 函数、如何添加新工具。

---

## 一、Rollout 数据解读

### 1.1 Rollout 数据在哪里

训练脚本中设置了 `trainer.rollout_data_dir`，默认保存在：

```
rollout_data/your_experiment/
├── step_1/
│   ├── batch_0.jsonl
│   └── batch_1.jsonl
├── step_50/
│   └── batch_0.jsonl
└── ...
```

每个 JSONL 文件包含该 step 的所有生成轨迹。

### 1.2 轨迹数据结构

```python
import json

# 读取一条轨迹
with open("rollout_data/experiment/step_50/batch_0.jsonl") as f:
    for line in f:
        traj = json.loads(line)
        break

# 轨迹字段说明
print(traj.keys())
# dict_keys(['prompt', 'response', 'reward', 'data_source', 'extra_info'])
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | str | 原始 prompt（已 apply chat template） |
| `response` | str | 模型完整回复（含多轮工具调用） |
| `reward` | float | 该轨迹的奖励分数 |
| `data_source` | str | 数据来源（如 `searchR1_nq`） |
| `extra_info` | dict | 包含 ground_truth、question 等 |

### 1.3 解读一条完整轨迹

一条成功的多轮搜索轨迹长这样：

```xml
<thought>
The user asks about the capital of Australia. Let me search for this.
</thought>
<tool_call>
{"name": "search", "arguments": {"query_list": ["capital of Australia"]}}
</tool_call>
<tool_response>
Doc 1 (Title: Canberra)
Canberra is the capital city of Australia...
</tool_response>
<thought>
The search results confirm that Canberra is the capital of Australia.
I have enough information to answer.
</thought>
<answer>Canberra</answer>
```

### 1.4 轨迹分析脚本

```python
#!/usr/bin/env python3
"""分析 rollout 数据，统计轨迹质量。"""
import json
import re
from pathlib import Path
from collections import Counter

def analyze_rollout(jsonl_path):
    """分析一个 rollout batch 文件。"""
    stats = {
        "total": 0,
        "rewarded": 0,  # reward > 0
        "avg_reward": 0,
        "avg_turns": 0,
        "format_errors": 0,
        "no_search": 0,
        "answer_found": 0,
    }
    rewards = []
    turns = []

    with open(jsonl_path) as f:
        for line in f:
            traj = json.loads(line)
            stats["total"] += 1
            reward = traj.get("reward", 0)
            rewards.append(reward)
            if reward > 0:
                stats["rewarded"] += 1

            response = traj.get("response", "")

            # 统计搜索轮数
            n_tool_calls = len(re.findall(r"<tool_call>", response))
            turns.append(n_tool_calls)

            # 检查格式
            if "<answer>" in response:
                stats["answer_found"] += 1
            if n_tool_calls == 0:
                stats["no_search"] += 1
            if "<tool_call>" in response and "</tool_call>" not in response:
                stats["format_errors"] += 1

    stats["avg_reward"] = sum(rewards) / len(rewards) if rewards else 0
    stats["avg_turns"] = sum(turns) / len(turns) if turns else 0
    stats["reward_distribution"] = {
        "0": sum(1 for r in rewards if r == 0),
        "0-0.5": sum(1 for r in rewards if 0 < r <= 0.5),
        "0.5-1": sum(1 for r in rewards if 0.5 < r <= 1.0),
    }
    return stats

# 使用示例
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "rollout_data/experiment/step_50/batch_0.jsonl"
    stats = analyze_rollout(path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
```

### 1.5 常见轨迹问题

| 问题 | 轨迹特征 | 原因 | 解决 |
|------|---------|------|------|
| 不搜索直接回答 | 无 `<tool_call>` | 模型没学会工具格式 | 检查 format/template |
| 搜索但不回答 | 有 `<tool_call>` 无 `<answer>` | 序列被截断 | 增大 max_response_length |
| 重复搜索相同内容 | 多个相同 query | 模型陷入循环 | credit assignment 会惩罚 |
| 格式错误 | `<tool_call>` 内容不是 JSON | 模型还在学习 | 训练初期正常 |
| 搜索结果为空 | `<tool_response>` 为空 | 检索服务问题 | 检查检索服务 |

---

## 二、自定义 Reward 函数

### 2.1 Reward 函数架构

```
verl/utils/reward_score/
├── __init__.py                  # 路由：根据 data_source 选择 reward 函数
├── search_r1_like_qa_em.py      # Search Agent 用的 QA Exact Match
├── gsm8k.py                     # 数学题 reward
├── math_reward.py               # MATH 数据集 reward
└── ...
```

核心路由逻辑在 `__init__.py`：

```python
def default_compute_score(data_source, solution_str, ground_truth, extra_info, ...):
    if data_source in ("searchR1_nq", "searchR1_triviaqa", "searchR1_hotpotqa", ...):
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth, extra_info)
        return res
```

### 2.2 Search-R1 Reward 详解

`search_r1_like_qa_em.py` 的评分逻辑：

```python
def compute_score(solution_str, ground_truth, extra_info):
    """
    评分 = format_score + answer_score

    format_score (0 或 format_reward):
        - 轨迹格式正确（通过状态机验证）→ format_reward (默认 0)
        - 格式错误 → 0

    answer_score (0 或 1):
        - 从 <answer>...</answer> 中提取答案
        - 与 ground_truth 做 Exact Match（忽略大小写、标点）
        - 匹配 → 1.0
        - 不匹配 → 0.0

    最终 reward = format_score + answer_score
    """
```

**关键点**：
- 答案提取：从 `<answer>` 和 `</answer>` 之间提取
- 匹配方式：normalize 后的 Exact Match（去标点、去冠词、小写化）
- 格式验证：状态机检查 `<thought>→<tool_call>→<tool_response>→<thought>→<answer>` 的合法序列

### 2.3 自定义 Reward 函数

假设你想添加一个新的评分方式（比如 F1 score 而不是 EM）：

**Step 1**：创建新的 reward 文件

```python
# verl/utils/reward_score/my_custom_reward.py

import re
from collections import Counter

def normalize_answer(s):
    """标准化答案文本。"""
    s = s.lower().strip()
    # 去除标点
    s = re.sub(r'[^\w\s]', '', s)
    # 去除冠词
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def f1_score(prediction, ground_truth):
    """计算 token-level F1 score。"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def extract_answer(text):
    """从 <answer>...</answer> 中提取答案。"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def compute_score(solution_str, ground_truth, extra_info=None):
    """
    自定义 reward：使用 F1 score 而不是 Exact Match。
    返回 0-1 之间的连续值。
    """
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0

    # ground_truth 可能是列表（多个正确答案）
    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]

    # 取所有正确答案中的最高 F1
    best_f1 = max(f1_score(answer, t) for t in targets) if targets else 0.0
    return best_f1
```

**Step 2**：注册到路由

编辑 `verl/utils/reward_score/__init__.py`，添加：

```python
    elif data_source in ("my_custom_qa",):
        from . import my_custom_reward
        res = my_custom_reward.compute_score(solution_str, ground_truth, extra_info)
```

**Step 3**：在数据预处理中设置 `data_source`

```python
# 预处理脚本中
row["data_source"] = "my_custom_qa"
```

### 2.4 Reward 调试技巧

```python
# 手动测试 reward 函数
from verl.utils.reward_score import default_compute_score

result = default_compute_score(
    data_source="searchR1_nq",
    solution_str="<thought>...</thought><answer>Paris</answer>",
    ground_truth={"target": ["Paris", "paris"]},
    extra_info={}
)
print(f"Score: {result}")  # 应该输出 1.0
```

---

## 三、自定义工具（Tool）

### 3.1 工具架构

```
verl/tools/
├── base_tool.py          # BaseTool 抽象基类
├── schemas.py            # 工具 schema 定义
├── search_tool.py        # 搜索工具实现
└── ...
```

工具通过 YAML 配置注册：

```yaml
# config/tool_config/search_tool_config.yaml
tools:
  - class_name: verl.tools.search_tool.SearchTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      num_workers: 60
      rate_limit: 60
      timeout: 20
      type: native
    tool_schema:
      type: function
      function:
        name: search
        description: Searches the web for relevant information.
        parameters:
          type: object
          properties:
            query_list:
              type: array
              item:
                type: string
              description: A list of search queries.
          required:
            - query_list
```

### 3.2 添加新工具

假设你想添加一个 **代码执行工具**：

**Step 1**：实现工具类

```python
# verl/tools/code_executor_tool.py

import subprocess
import tempfile
from typing import Any, Dict, List
from .base_tool import BaseTool
from .schemas import ToolResponse

class CodeExecutorTool(BaseTool):
    """执行 Python 代码并返回结果。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get("timeout", 10)
        self.max_output_length = config.get("max_output_length", 1000)

    async def execute(self, arguments: Dict[str, Any]) -> ToolResponse:
        """执行代码并返回输出。"""
        code = arguments.get("code", "")

        try:
            # 写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tmp_path = f.name

            # 执行（带超时和沙箱）
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout[:self.max_output_length]
            if result.returncode != 0:
                output = f"Error: {result.stderr[:self.max_output_length]}"

            return ToolResponse(content=output, success=True)

        except subprocess.TimeoutExpired:
            return ToolResponse(content="Execution timed out.", success=False)
        except Exception as e:
            return ToolResponse(content=f"Error: {str(e)}", success=False)
```

**Step 2**：创建工具配置

```yaml
# config/tool_config/code_tool_config.yaml
tools:
  - class_name: verl.tools.search_tool.SearchTool
    config:
      retrieval_service_url: http://127.0.0.1:8000/retrieve
      num_workers: 60
      rate_limit: 60
      timeout: 20
      type: native
    tool_schema:
      type: function
      function:
        name: search
        description: Searches the web for relevant information.
        parameters:
          type: object
          properties:
            query_list:
              type: array
              item:
                type: string
          required:
            - query_list

  - class_name: verl.tools.code_executor_tool.CodeExecutorTool
    config:
      timeout: 10
      max_output_length: 1000
      type: native
    tool_schema:
      type: function
      function:
        name: execute_code
        description: Executes Python code and returns the output.
        parameters:
          type: object
          properties:
            code:
              type: string
              description: Python code to execute.
          required:
            - code
```

**Step 3**：在训练脚本中指定新配置

```bash
bash scripts/v100/run_search_r1_v100.sh \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=config/tool_config/code_tool_config.yaml
```

### 3.3 工具开发注意事项

| 要点 | 说明 |
|------|------|
| 异步执行 | `execute` 方法必须是 `async def` |
| 超时控制 | 工具调用不能阻塞太久，否则 rollout 会卡住 |
| 错误处理 | 返回 `ToolResponse(success=False)` 而不是抛异常 |
| 输出长度 | 控制返回内容长度，太长会占用 context window |
| 并发安全 | 多个 rollout worker 会并发调用，确保线程安全 |
| Rate Limiting | 如果调用外部 API，使用 `GlobalRateLimiter` |

### 3.4 工具调试

```python
# 手动测试工具
import asyncio
from verl.tools.search_tool import SearchTool

tool = SearchTool(config={
    "retrieval_service_url": "http://127.0.0.1:8000/retrieve",
    "num_workers": 1,
    "rate_limit": 10,
    "timeout": 20,
    "type": "native",
})

async def test():
    result = await tool.execute({"query_list": ["What is Python?"]})
    print(f"Success: {result.success}")
    print(f"Content: {result.content[:500]}")

asyncio.run(test())
```

---

## 四、自定义 Agent Loop

### 4.1 Agent Loop 架构

```python
# verl/experimental/agent_loop/agent_loop.py

class AgentLoopBase(ABC):
    @abstractmethod
    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        """
        输入: prompt messages + sampling params
        输出: AgentLoopOutput(prompt_ids, response_ids, response_mask)

        response_mask 的作用:
          1 = 模型生成的 token（参与 loss 计算）
          0 = 工具返回的 token（不参与 loss 计算）
        """
        raise NotImplementedError
```

### 4.2 自定义 Agent Loop 示例

```python
# my_agent_loop.py
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

class MyCustomAgentLoop(AgentLoopBase):
    """自定义 Agent Loop：搜索 + 反思 + 再搜索。"""

    async def run(self, sampling_params, **kwargs):
        # 第一轮：生成初始思考和搜索
        response_ids_1 = await self.server_manager.generate(
            request_id=self.request_id,
            prompt_ids=self.prompt_ids,
            sampling_params=sampling_params,
        )

        # 解析工具调用
        tool_calls = self.parse_tool_calls(response_ids_1)

        if tool_calls:
            # 执行工具
            tool_results = await self.execute_tools(tool_calls)

            # 拼接工具结果
            full_ids = self.prompt_ids + response_ids_1 + tool_results_ids

            # 第二轮：基于搜索结果生成最终答案
            response_ids_2 = await self.server_manager.generate(
                request_id=self.request_id,
                prompt_ids=full_ids,
                sampling_params=sampling_params,
            )

            # 构建 mask（工具结果部分为 0）
            mask = [1] * len(response_ids_1) + [0] * len(tool_results_ids) + [1] * len(response_ids_2)

            return AgentLoopOutput(
                prompt_ids=self.prompt_ids,
                response_ids=response_ids_1 + tool_results_ids + response_ids_2,
                response_mask=mask,
            )
        else:
            # 没有工具调用，直接返回
            return AgentLoopOutput(
                prompt_ids=self.prompt_ids,
                response_ids=response_ids_1,
                response_mask=[1] * len(response_ids_1),
            )
```

### 4.3 注册自定义 Agent Loop

创建配置文件：

```yaml
# config/agent_loop/my_custom_agent.yaml
- name: my_agent
  _target_: my_agent_loop.MyCustomAgentLoop
```

在训练脚本中指定：

```bash
bash scripts/v100/run_search_r1_v100.sh \
    actor_rollout_ref.rollout.agent.default_agent_loop=my_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config=config/agent_loop/my_custom_agent.yaml
```
