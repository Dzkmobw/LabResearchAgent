# LabResearchAgent
哦哦哦哦哦哦哦哦

## MVP: Lab Research Harness Agent

这个仓库现在包含一个最小版的硕博实验室调研 Agent。它的目标不是一次性做完整产品，而是把简历里写到的 Agent 工程能力先落成可运行骨架。

### 已实现能力

- **LangGraph 工作流编排**：`clarify_requirements -> generate_research_plan -> discover_professors -> discover_web_sources -> resolve_professor_identity -> collect_publications -> analyze_match -> generate_report -> quality_gate`
- **Pydantic 结构化建模**：申请者、任务契约、教授候选、已确认教授、论文、Evidence、实验室档案和最终报告都定义在 `src/agent/schemas.py`
- **Tool Gateway Harness**：统一管理工具注册、白名单调用、超时、重试、缓存、调用预算、错误分类和工具日志
- **DeepSeek V4 模型接入**：所有 LLM 调用统一走 `src/agent/llm.py`，默认使用 `deepseek-v4-flash`，可切换为 `deepseek-v4-pro`
- **OpenAlex 学术数据接入**：使用 `httpx` 查询作者候选和近年论文
- **MCP Web Search 插槽**：`MCPWebSearchTool` 已接入 Tool Gateway；配置 MCP server 后可用于发现学校官网、实验室主页和招生页面
- **教授身份消歧**：基于姓名、学校 affiliation、OpenAlex author id、论文主题和候选分数做多信号判断
- **Evidence Ledger**：关键事实记录 `source_url`、`source_type`、`retrieved_at`、`confidence` 和 `is_inference`
- **Quality Gate**：检查教授身份、来源数量、论文、官方招生来源和 unsupported evidence
- **Trace 与 Eval 骨架**：节点 trace、工具日志会进入 state；`evals/` 下包含最小 LabResearchBench 骨架

### 最小调用示例

```python
import asyncio
from agent.graph import graph


async def main():
    result = await graph.ainvoke(
        {
            "run_spec": {
                "target_country": "Japan",
                "degree": "phd",
                "research_interests": ["natural language processing", "speech recognition"],
                "target_schools": ["Kyoto University"],
                "target_professor": "Tatsuya Kawahara",
                "lab_count": 1,
                "publication_years": 5,
                "max_tool_calls": 6,
            }
        }
    )
    print(result["status"].value)
    print(result["report"].executive_summary)


asyncio.run(main())
```

### DeepSeek V4 配置

所有模型调用都统一使用 DeepSeek，不接 OpenAI。把 key 写在 `.env`：

```bash
DEEPSEEK_API_KEY=你的_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
LAB_AGENT_MODEL=deepseek-v4-flash
LAB_AGENT_THINKING=disabled
```

如果你想用更强的版本，把模型改成：

```bash
LAB_AGENT_MODEL=deepseek-v4-pro
```

配置好以后，可以直接给自然语言请求，Agent 会先尝试用 DeepSeek 解析成 `ResearchRunSpec`。如果没有 key 或信息缺失，会进入 `needs_clarification`。

### MCP Web Search 配置

当前默认不假装联网搜索。如果没有配置 MCP，`discover_web_sources` 会返回 `mcp_not_configured`，最终报告通常是 `partial`，并提示缺少官方招生来源。

后续可以安装 MCP adapter：

```bash
uv sync --extra mcp
```

并通过环境变量提供 MCP server 配置：

```bash
export LAB_AGENT_MCP_SERVERS='{"search":{"transport":"stdio","command":"your-search-mcp-server","args":[]}}'
export LAB_AGENT_MCP_SEARCH_TOOL='web_search'
```

### LangSmith Trace

图节点使用 `langsmith.traceable` 标记。如果设置了 LangSmith 环境变量，运行时可以在 LangSmith 中查看节点、工具、证据和失败原因。

### Eval

`evals/` 提供最小评测骨架：

```bash
python -m evals.run_evals
```

当前只用于展示评测闭环，不包含真实效果指标。只有跑完稳定测试集后，才能在 README 或简历中补充准确率、覆盖率、耗时等数字。
