# Lab Research Agent 技术面试 Q&A

这份文档按“面试官追问源码细节”的标准写。回答时要记住：当前项目是 MVP，不要吹成完整生产系统。最稳的口径是：

> 这是一个基于 LangGraph 的实验室调研 Harness Agent。MVP 已经实现了确定性工作流、DeepSeek V4 模型入口、OpenAlex 真实学术数据接入、Pydantic 结构化状态、Tool Gateway、Evidence Ledger、教授身份消歧、Quality Gate 和 Eval 骨架。MCP Web Search 是可配置插槽，配置 server 后才会真实搜索网页。

## 1. 总体架构

### Q1: 你这个项目的核心架构是什么？

可以按四层讲：

```text
User / API Input
    ↓
LangGraph Workflow
    ↓
Harness Layer
    ↓
External Tools / LLM / Data Sources
```

对应源码：

- LangGraph 图入口：`src/agent/graph.py`
- 图注册：`langgraph.json` 里的 `"agent": "./src/agent/graph.py:graph"`
- 状态模型：`src/agent/state.py` 的 `ResearchState`
- 结构化 Schema：`src/agent/schemas.py`
- 工具控制层：`src/agent/harness/tool_gateway.py`
- OpenAlex 工具：`src/agent/tools/openalex.py`
- MCP Web Search 插槽：`src/agent/tools/mcp_web_search.py`
- DeepSeek V4 入口：`src/agent/llm.py`
- Quality Gate：`src/agent/harness/quality_gate.py`
- Eval 骨架：`evals/`

面试回答：

> 我没有把所有逻辑塞进一个 Prompt，而是把 Agent 拆成可观测的状态图。LLM 只负责自然语言解析和报告表达，工具调用、预算、缓存、证据校验和状态路由都写成确定性代码。

### Q2: 用户请求进入系统后发生了什么？

如果用户直接传结构化 `run_spec`，流程是：

```text
run_spec
→ clarify_requirements
→ generate_research_plan
→ discover_professors
→ discover_web_sources
→ resolve_professor_identity
→ collect_publications
→ analyze_match
→ generate_report
→ quality_gate
→ report/status
```

如果用户只输入自然语言 `user_request`：

```text
user_request
→ DeepSeek V4 解析成 ResearchRunSpec
→ 后续进入同一条 LangGraph 流程
```

如果没有配置 `DEEPSEEK_API_KEY`，自然语言输入会返回：

```text
status = needs_clarification
missing_fields = ["DEEPSEEK_API_KEY"]
```

### Q3: 这个项目目前是不是 ReAct Agent？

不是。当前更准确说是：

> 这是一个带 LLM 入口的 Harness Workflow，不是模型自由决定下一步工具调用的 ReAct Agent。

当前工具路由是确定性的：

```text
OpenAlex author search
→ MCP Web Search slot
→ identity resolver
→ OpenAlex works search
```

为什么这样设计：

- 教授调研更看重可靠性和可复现。
- 身份消歧和来源校验不适合完全交给模型。
- 确定性路由更容易做 trace eval。
- 后续可以加 model-based planning，但所有工具仍然经过 Tool Gateway。

## 2. LangGraph 工作流

### Q4: 你是怎么创建 LangGraph 图的？

源码在 `src/agent/graph.py` 底部：

```python
builder = StateGraph(ResearchState, context_schema=Context)
builder.add_node("clarify_requirements", clarify_requirements)
builder.add_node("generate_research_plan", generate_research_plan)
builder.add_node("discover_professors", discover_professors)
builder.add_node("discover_web_sources", discover_web_sources)
builder.add_node("resolve_professor_identity", resolve_professor_identity)
builder.add_node("collect_publications", collect_publications)
builder.add_node("analyze_match", analyze_match)
builder.add_node("generate_report", generate_report)
builder.add_node("quality_gate", quality_gate)

builder.add_edge(START, "clarify_requirements")
builder.add_conditional_edges("clarify_requirements", route_after_clarification)
...
graph = builder.compile(name="Lab Research Agent")
```

重点解释：

- `StateGraph(ResearchState)`：所有节点共享同一个业务状态。
- `context_schema=Context`：保留运行时配置入口。
- `add_node`：注册每个任务节点。
- `add_edge`：定义确定性执行顺序。
- `add_conditional_edges`：输入不完整时提前结束。
- `compile(name=...)`：生成 LangGraph 可运行对象。

### Q5: 图入口是怎么注册给 LangGraph server 的？

`langgraph.json`：

```json
{
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env"
}
```

意思是：

- assistant id 叫 `agent`
- Python 对象在 `src/agent/graph.py`
- 对象名是 `graph`
- 启动 `langgraph dev` 时会读取 `.env`

启动后调用：

```bash
curl -X POST http://127.0.0.1:2024/runs/wait \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {
      "user_request": "日本 NLP 博士，看看京都大学 Tatsuya Kawahara 教授"
    }
  }'
```

### Q6: 每个节点具体负责什么？

当前节点职责：

- `clarify_requirements`: 校验输入；如果只有自然语言，则用 DeepSeek 抽取 `ResearchRunSpec`
- `generate_research_plan`: 生成确定性调研计划
- `discover_professors`: 通过 Tool Gateway 调 OpenAlex `/authors`
- `discover_web_sources`: 通过 MCP 插槽搜索官网/实验室/招生页面
- `resolve_professor_identity`: 根据候选教授打分，判断是否 ambiguous
- `collect_publications`: 用 OpenAlex author id 查询 `/works`
- `analyze_match`: 用论文标题和关键词匹配申请者研究兴趣
- `generate_report`: 生成 `LabProfile` 和 `LabComparisonReport`
- `quality_gate`: 检查身份、来源、论文、官方招生信息和证据完整性

### Q7: LangGraph 的状态里保存了什么？

`ResearchState` 在 `src/agent/state.py`，核心字段：

```python
run_id: str
user_request: str
applicant: ApplicantProfile | None
run_spec: ResearchRunSpec | None
research_plan: list[str]
professor_candidates: list[ProfessorCandidate]
resolved_professor: ResolvedProfessor | None
publications: list[Publication]
research_trend: ResearchTrend | None
lab_profile: LabProfile | None
report: LabComparisonReport | None
evidence: list[Evidence]
tool_call_count: int
tool_logs: list[ToolCallLog]
trace: list[TraceEvent]
quality_issues: list[QualityIssue]
status: RunStatus
```

回答口径：

> 我没有只用 messages 做状态，而是直接存业务对象。这样每个节点读取的是结构化字段，后续做质量检查和 Eval 会更稳定。

### Q8: 你怎么处理异常状态？

用 `RunStatus` 枚举：

```text
created
needs_clarification
planned
searching
resolved
needs_review
partial
complete
failed
```

例子：

- 缺少任务契约：`needs_clarification`
- MCP 没配置但 OpenAlex 成功：通常 `partial`
- 教授候选分数低或候选接近：`needs_review`
- Quality Gate 发现硬错误：`failed`
- 所有关键检查通过：`complete`

## 3. Pydantic 结构化建模

### Q9: 为什么要用 Pydantic？

Agent 系统里有三类东西很容易不稳定：

- 用户输入不完整
- LLM 输出格式不稳定
- 外部 API 返回字段不统一

Pydantic 在这里是结构化契约层：

- 校验输入：`ResearchRunSpec`
- 统一工具输出：`ToolResult`
- 统一事实证据：`Evidence`
- 统一最终报告：`LabComparisonReport`

### Q10: `ResearchRunSpec` 约束了什么？

在 `src/agent/schemas.py`：

```python
class ResearchRunSpec(BaseModel):
    target_country: str
    degree: Literal["master", "phd"]
    research_interests: list[str] = Field(min_length=1)
    target_schools: list[str] = Field(default_factory=list)
    target_professor: str | None = None
    target_lab: str | None = None
    lab_count: int = Field(default=1, ge=1, le=10)
    publication_years: int = Field(default=5, ge=1, le=10)
    max_tool_calls: int = Field(default=8, ge=1, le=50)
```

还有一个 `model_validator`：

```python
if not target_professor and not target_lab and not target_schools:
    raise ValueError(...)
```

面试回答：

> 这个契约保证 Agent 至少知道目标国家、申请学位、研究兴趣，以及教授/实验室/学校中的一个。缺失时不让模型猜，而是进入澄清分支。

### Q11: `Evidence` 为什么要单独建模？

`Evidence` 是这个项目可靠性的核心。字段包括：

```python
claim: str
source_url: str | None
source_title: str | None
source_type: SourceType
retrieved_at: datetime
confidence: float
is_inference: bool
supports: list[str]
metadata: dict[str, Any]
```

重点：

- `source_url`: 事实从哪里来
- `source_type`: 官网、学术 API、网页搜索、论文、模型推断等
- `is_inference`: 区分事实和推断
- `supports`: 一个推断可以指向多个支撑证据 id

回答口径：

> 我不让报告直接引用裸文本，而是所有关键事实都先进入 Evidence Ledger。最终报告引用的是 Evidence，而不是模型临时编出来的内容。

## 4. DeepSeek V4 接入

### Q12: DeepSeek V4 是怎么接入的？

所有 LLM 调用集中在 `src/agent/llm.py` 的 `DeepSeekChatClient`：

```python
self.api_key = os.getenv("DEEPSEEK_API_KEY")
self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
self.model = os.getenv("LAB_AGENT_MODEL", "deepseek-v4-flash")
```

调用方式：

```text
POST {DEEPSEEK_BASE_URL}/chat/completions
Authorization: Bearer {DEEPSEEK_API_KEY}
```

配置放在 `.env`：

```bash
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
LAB_AGENT_MODEL=deepseek-v4-flash
```

### Q13: DeepSeek 现在具体负责哪些任务？

当前只负责两个 LLM 任务：

1. `infer_research_run_spec`

把自然语言变成结构化任务，例如：

```text
"我想申请日本 NLP 博士，看看京都大学 Tatsuya Kawahara 教授是否匹配"
```

转换成：

```json
{
  "target_country": "Japan",
  "degree": "phd",
  "research_interests": ["natural language processing"],
  "target_schools": ["Kyoto University"],
  "target_professor": "Tatsuya Kawahara"
}
```

2. `draft_report_text`

把结构化 `LabProfile` 写成更自然的 `executive_summary` 和 `match_rationale`。

### Q14: 为什么不在每个节点直接 import 模型？

当前做法：

```text
graph.py
→ infer_research_run_spec()
→ DeepSeekChatClient
```

和：

```text
graph.py
→ draft_report_text()
→ DeepSeekChatClient
```

好处：

- 所有 LLM 配置集中
- 后面换模型只改一个文件
- 更容易记录 token、latency、cost
- 更容易做 fallback

### Q15: 为什么不让 DeepSeek 直接决定调用什么工具？

MVP 先不这么做。原因：

- 教授调研的失败成本主要来自身份错配和无来源结论。
- 如果让模型自由路由，trace 会更复杂，debug 成本更高。
- 当前目标是先把可靠业务闭环跑通。

回答：

> 我把 LLM 放在语言理解和表达层，把控制逻辑放在 Harness 层。后续可以加 model-based planning，但工具调用仍然要经过 Tool Gateway。

## 5. Tool Gateway / Harness

### Q16: Tool Gateway 解决了什么问题？

如果节点直接调用工具，会出现这些问题：

- 每个工具自己处理超时和重试，逻辑重复
- 调用次数不可控，可能无限跑
- 工具失败时错误格式不统一
- 难以统计 latency、attempts、cache hit
- 不方便做 Trace Eval

所以我加了 `ToolGateway`，路径是 `src/agent/harness/tool_gateway.py`。

### Q17: Tool Gateway 怎么实现？

核心结构：

```python
ToolGateway(
    max_tool_calls=spec.max_tool_calls,
    tool_timeout_seconds=spec.tool_timeout_seconds,
    max_retries=spec.max_retries,
    tool_call_count=state.tool_call_count,
)
```

使用方式：

```python
gateway.register("openalex.search_authors", openalex.search_authors)
call = await gateway.call("openalex.search_authors", {...})
```

`call()` 里做了这些事：

- 检查工具是否注册
- 基于工具名和参数生成 cache key
- 检查 `max_tool_calls`
- 使用 `asyncio.wait_for` 控制超时
- 根据 `max_retries` 重试
- 生成 `ToolCallLog`
- 返回标准化 `GatewayCallResult`

### Q18: cache key 怎么做的？

当前是内存缓存 `_CACHE`：

```python
payload = json.dumps(arguments, sort_keys=True, default=str)
hashlib.sha256(f"{name}:{payload}".encode()).hexdigest()
```

局限：

- 只是进程内缓存，服务重启后会丢失
- 没有 TTL
- 后续可以换成 SQLite 或 diskcache

面试补充：

> MVP 先用 in-memory cache 验证 Harness 设计，后续可以替换成持久化缓存，不影响节点逻辑。

### Q19: 失败怎么分类？

当前分类：

- `tool_not_registered`: 工具没注册
- `budget_exhausted`: 超过调用预算
- `timeout`: 工具调用超时
- `tool_error`: 其他异常
- `mcp_not_configured`: 没配 MCP server
- `mcp_adapter_missing`: 没安装 MCP adapter
- `mcp_search_tool_missing`: MCP server 里找不到 search 工具

这些错误会进入 `tool_logs` 和 `errors`，最终可以被 Trace/Eval 分析。

## 6. OpenAlex 学术数据接入

### Q20: OpenAlex 接了哪些接口？

路径：`src/agent/tools/openalex.py`

两个主要方法：

```python
search_authors()
fetch_author_works()
```

`search_authors` 调：

```text
GET https://api.openalex.org/authors?search={query}&per-page={limit}
```

`fetch_author_works` 调：

```text
GET https://api.openalex.org/works
filter=authorships.author.id:{author_id},from_publication_date:{year}-01-01
sort=publication_date:desc
```

### Q21: OpenAlex 返回结果怎么标准化？

作者结果转成 `ProfessorCandidate`：

```python
candidate_id
display_name
alternative_names
affiliations
source_ids
topics
works_count
cited_by_count
evidence_ids
```

论文结果转成 `Publication`：

```python
publication_id
title
year
venue
doi
cited_by_count
url
authors
keywords
source_evidence_ids
```

每个作者候选和论文都会生成对应 Evidence：

```python
source_type = SourceType.ACADEMIC_API
source_title = "OpenAlex Authors API" 或 "OpenAlex Works API"
source_url = 实际请求 URL
```

### Q22: 为什么 OpenAlex 适合 MVP？

优点：

- 有公开 API
- 能查作者、机构、论文
- 有稳定 id
- 适合程序化调用和 Eval
- 比 Google Scholar 更可复现

局限：

- 作者消歧不一定完美
- 机构信息可能滞后
- 论文主题标签可能不够细
- 招生信息无法从 OpenAlex 获得

## 7. MCP Web Search

### Q23: MCP 在项目里具体怎么用？

MCP 不是数据源，而是工具协议。当前项目把它作为 Web Search 插槽：

```text
discover_web_sources
→ ToolGateway
→ MCPWebSearchTool
→ MultiServerMCPClient
→ 具体 MCP search server
```

源码在 `src/agent/tools/mcp_web_search.py`。

### Q24: MCP 怎么配置？

环境变量：

```bash
LAB_AGENT_MCP_SERVERS='{"search":{"transport":"stdio","command":"your-search-mcp-server","args":[]}}'
LAB_AGENT_MCP_SEARCH_TOOL='web_search'
```

依赖：

```bash
uv sync --extra mcp
```

代码会：

- 读取 `LAB_AGENT_MCP_SERVERS`
- 初始化 `MultiServerMCPClient`
- 调 `client.get_tools()`
- 优先找 `LAB_AGENT_MCP_SEARCH_TOOL`
- 找不到时找名字里包含 `search` 或 `web` 的工具
- 调 `tool.ainvoke({"query": query, "max_results": limit})`

### Q25: 没有配置 MCP 时会怎样？

不会报死，也不会假装成功。返回：

```python
ToolResult(
    tool_name="mcp.web_search",
    status="skipped",
    error_type="mcp_not_configured",
)
```

然后 Quality Gate 通常会发现：

```text
missing_official_admission_source
```

最终状态通常是 `partial`，表示有学术数据，但官网招生来源缺失。

## 8. 教授身份消歧

### Q26: 为什么身份消歧是核心难点？

实验室调研里最容易出错的是“人找错了”。特别是日本教授场景：

- 同名教授
- 英文名、罗马字、日文名不一致
- 教授换学校
- OpenAlex 作者聚合错误
- 同一个领域有多个相似名字

如果身份错了，后面的论文、方向匹配、招生判断全部都会错。

### Q27: 当前身份消歧怎么打分？

源码在 `graph.py` 的 `_score_candidate()`。

当前分数由几类信号组成：

```text
base score: 0.05
exact name match: +0.45
partial name match: +0.30
school affiliation match: +0.30
topic / interest match: +0.15
has works_count: +0.03
has cited_by_count: +0.02
max score: 0.95
```

歧义判断：

```python
is_ambiguous = top_score < 0.6 or top_score - second_score < 0.12
```

如果 ambiguous：

- `resolved_professor.is_ambiguous = True`
- trace 记录 `needs_review`
- Quality Gate 产生 `professor_identity_ambiguous`
- 最终状态可能变成 `needs_review`

### Q28: 为什么不用 LLM 做身份消歧？

当前 MVP 用确定性规则，因为身份消歧是高风险步骤。LLM 可以辅助解释，但不能单独决定。

回答：

> 我把身份消歧做成可解释打分，保留每个候选的 affiliation、source id、topic 和 evidence。这样出错时可以定位是哪个信号弱，而不是只看到模型一句“我认为是这个人”。

后续可以加 LLM reranker，但前提是：

- 候选来自可信数据源
- LLM 输出必须包含理由
- Quality Gate 仍检查 ambiguous 状态
- 用户可以人工确认

## 9. 研究方向匹配

### Q29: 匹配度现在怎么计算？

当前是一个确定性 MVP：

```text
论文标题 + OpenAlex keywords
→ 拼成 corpus
→ 和用户 research_interests 做关键词匹配
→ 计算 matched_interests / total_interests
```

如果有论文但没有直接命中，给一个弱匹配分：

```python
if publications and score == 0:
    score = 0.2
```

输出：

- `ResearchTrend.top_keywords`
- `ResearchTrend.matched_interests`
- `LabProfile.match_score`
- `LabProfile.match_rationale`
- 一条 `is_inference=True` 的 Evidence

### Q30: 这个匹配算法有什么不足？

主动承认：

- 只是关键词级别，不是语义匹配
- 没有用 embedding
- 没有读 abstract/full text
- 没有区分一作/通讯作者
- 没有区分教授本人近期主线和合作论文

下一步增强：

- 用 Semantic Scholar abstract
- 用 embedding 计算兴趣和论文摘要相似度
- 给近五年论文加时间权重
- 加推荐阅读模块，推荐最相关的 3-5 篇论文
- 区分 strong match / adjacent match / weak match

### Q31: 为什么匹配度要标记成 inference？

因为“是否匹配申请者方向”不是来源直接给出的事实，而是系统根据论文和兴趣推断出来的。

所以它必须是：

```python
source_type = MODEL_INFERENCE
is_inference = True
supports = publication evidence ids
```

回答：

> 我把事实和推断分开。论文列表是学术 API 事实，匹配度是系统推断。报告里不能把推断写成官网事实。

## 10. Evidence Ledger 和来源校验

### Q32: Evidence Ledger 在代码里怎么流动？

每个工具返回 `ToolResult`：

```python
ToolResult(
    tool_name="openalex.search_authors",
    items=[...],
    evidence=[...],
)
```

节点拿到结果后：

```python
"evidence": state.evidence + call.result.evidence
```

最终报告：

```python
LabComparisonReport(
    evidence=state.evidence,
    profiles=[profile],
)
```

### Q33: 你怎么区分事实和推断？

OpenAlex 返回论文：

```python
source_type = SourceType.ACADEMIC_API
is_inference = False
source_url = request_url
```

身份消歧结果：

```python
source_type = SourceType.MODEL_INFERENCE
is_inference = True
source_url = None
supports = candidate.evidence_ids
```

匹配度结论：

```python
is_inference = True
supports = publication.source_evidence_ids
```

回答：

> 只要不是来源直接声明的事实，就必须标记为 inference，并指向支撑它的 Evidence。

### Q34: 来源优先级怎么设计？

当前代码已有 `SourceType`，下一步会用它做 source ranking：

```text
OFFICIAL > ACADEMIC_API > PAPER > WEB_PAGE > WEB_SEARCH > AGGREGATOR > MODEL_INFERENCE
```

场景：

- 招生要求：必须优先官网
- 论文：学术 API 或论文 DOI
- 教授任职机构：官网 + OpenAlex affiliation 交叉验证
- 匹配度：推断，必须引用论文 evidence

## 11. Quality Gate

### Q35: Quality Gate 检查哪些东西？

路径：`src/agent/harness/quality_gate.py`

当前检查：

1. 是否生成 `LabProfile`
2. 是否有教授身份
3. 教授身份是否 ambiguous
4. evidence 数量是否低于 `min_sources_per_lab`
5. 是否有近年论文
6. 如果要求官方招生来源，是否存在 `SourceType.OFFICIAL` 的招生 Evidence
7. 非推断 Evidence 是否缺少 `source_url`

输出是 `QualityIssue`：

```python
code: str
severity: "info" | "warning" | "error"
message: str
suggested_action: str
```

### Q36: Quality Gate 怎么决定最终状态？

在 `graph.py` 的 `quality_gate()`：

```python
if has_error:
    final_status = RunStatus.FAILED
elif needs_review:
    final_status = RunStatus.NEEDS_REVIEW
elif issues:
    final_status = RunStatus.PARTIAL
else:
    final_status = RunStatus.COMPLETE
```

例子：

- OpenAlex 成功，但没配 MCP，缺官网招生来源：`partial`
- 教授候选分数接近：`needs_review`
- 没生成 profile：`failed`
- 论文、身份、官网证据都齐：`complete`

### Q37: 为什么需要 Quality Gate，不让模型自己检查？

模型检查容易不稳定，也可能放过自己的幻觉。Quality Gate 是确定性规则，能稳定复现。

回答：

> 我把格式、来源、身份状态、预算状态这类可规则化的检查写成代码。LLM 可以帮忙写报告，但不能决定没有来源的事实是否合格。

## 12. Trace 和 LangSmith

### Q38: 你现在记录了哪些 trace？

本地状态里有两种 trace：

1. Node trace: `TraceEvent`

```python
node
status
message
metadata
created_at
```

2. Tool trace: `ToolCallLog`

```python
tool_name
arguments_summary
status
attempts
duration_ms
cache_hit
error_type
message
created_at
```

另外，节点用 `@traceable` 标记；配置 LangSmith 后可以在平台里看到运行链路。

### Q39: LangSmith 在项目里具体做什么？

LangSmith 是可观测性和 Eval 平台，不是模型 API。

它适合记录：

- 哪个节点先执行
- LLM 调了几次
- 工具调用参数摘要
- 工具是否失败
- Evidence 是否进入报告
- Quality Gate 为什么给 partial/needs_review

当前项目已经在节点上加了 `@traceable`，后续可以把 Eval 结果也上传到 LangSmith Dataset。

### Q40: 为什么本地也要存 trace，不只依赖 LangSmith？

因为：

- 没有 LangSmith key 时也能 debug
- Eval 脚本可以直接读 graph output
- 本地 trace 更容易做断言
- 面试 demo 不依赖外部平台

回答：

> LangSmith 负责平台可观测，本地 TraceEvent/ToolCallLog 负责可复现数据结构，两者不是互相替代。

## 13. Eval 设计

### Q41: Eval Harness 当前实现了什么？

当前在 `evals/` 下：

- `dataset.jsonl`: 最小测试样本
- `evaluators.py`: 确定性 evaluator
- `run_evals.py`: 批量跑图并输出结果

当前 evaluator 检查：

- status 是否符合预期
- 是否有教授身份
- 是否有论文
- 是否有 evidence
- 是否有 trace
- 是否有 tool_logs

### Q42: 你真正想评估哪些指标？

后续 LabResearchBench 会分三层：

Output Eval:

- 教授身份是否正确
- 论文列表是否属于正确作者
- 推荐/匹配理由是否被论文支持
- 招生信息是否来自官网
- 关键事实引用覆盖率

Trace Eval:

- 是否先身份消歧再分析论文
- 是否调用了 OpenAlex / MCP Web Search
- 是否在 ambiguous 时进入 needs_review
- 是否遵守 max_tool_calls
- 是否没有无限重试

Robustness Eval:

- MCP 未配置时是否 partial
- API 失败时是否记录 error
- 官网打不开是否标缺失
- 同名教授是否不强行 complete
- 缺少用户约束是否 needs_clarification

### Q43: 为什么 Eval 不只是看最终答案？

因为 Agent 的错误可能藏在过程里。

例子：

- 最终报告看起来对，但没有调用官网来源
- 找到论文了，但其实作者是同名教授
- 报告说“招生中”，但没有官方来源
- 工具失败了，但模型硬编了结果

所以要同时看 output 和 trace。

### Q44: 如果面试官问“你有真实指标吗”怎么答？

不要编。

可以答：

> 现在仓库里有 Eval Harness 骨架和少量样本，用于保证流程可跑。完整 30 条 LabResearchBench 和正式指标还在下一阶段。我不会在 README 或简历里写没跑过的准确率。

## 14. Harness 和可靠性

### Q45: 你说的 Harness 到底是什么？

Harness 是模型外部的运行控制层，不是另一个 Agent。

在这个项目里，Harness 包括：

- `ResearchRunSpec`: 输入契约
- `ToolGateway`: 工具调用控制
- `Evidence`: 证据结构
- `QualityGate`: 输出质量检查
- `ToolCallLog` / `TraceEvent`: 可观测性
- `RunStatus`: 运行状态

回答：

> LLM 负责理解和生成，LangGraph 负责编排，Harness 负责约束、预算、重试、缓存、证据和验证。

### Q46: 为什么 Harness 比多 Agent 更重要？

因为这个项目的主要风险不是“能力不够多”，而是“结果不可信”。

多 Agent 会增加复杂度，但不一定减少幻觉。Harness 可以先解决：

- 工具失败怎么处理
- 证据怎么追踪
- 预算怎么控制
- 身份不确定怎么标记
- 报告不合格怎么拦截

后续如果要加多 Agent，也应该建立在 Harness 之上。

### Q47: partial / needs_review 有什么意义？

它们是可靠性的体现。

- `partial`: 已经拿到部分证据，但缺少关键来源，比如官网招生页面
- `needs_review`: 系统发现身份歧义，需要人工确认

回答：

> 我不希望 Agent 为了输出完整报告而编造缺失信息。partial 和 needs_review 是让系统显式表达不确定性。

## 15. 自进化

### Q48: 你说的 Agent 自进化是什么？

在这个项目里，自进化不是自动训练模型，也不是自动改代码。

更准确说是：

```text
失败 Trace
→ Quality Gate 问题
→ Eval 失败类型
→ 生成候选改进建议
→ 人工确认
→ 更新 policy / prompt / tool routing
→ 回归 Eval
```

例子：

```text
失败：同名教授识别错
原因：只看姓名，没有检查 affiliation
候选经验：身份消歧必须比较学校官网、OpenAlex author id、affiliation、论文主题
验证：同名教授 eval case 通过后再保留
```

### Q49: 当前是否已经实现自进化？

要诚实：

> 当前已经有 Trace、Quality Gate 和 Eval 骨架，这是自进化闭环的数据基础。自动生成 improvement proposal 和 policy versioning 还没做。

下一步文件可以是：

```text
src/agent/harness/experience.py
```

定义：

```python
ExperienceCandidate:
    run_id
    failure_type
    root_cause
    proposed_rule
    affected_component
    status
    eval_before
    eval_after
```

## 16. Sandbox 问题

### Q50: 这个项目现在用了 sandbox 吗？

没有。当前没有实现 sandbox。

如果被问到，答：

> 我评估过 sandbox，但 MVP 阶段没有引入。因为当前没有执行用户代码，也没有解析不可信 PDF 的复杂流程。现阶段更需要把 Tool Gateway、Evidence、身份消歧和 Quality Gate 做扎实。

### Q51: 后续哪里适合加 sandbox？

适合放进 sandbox 的是：

- 不可信 HTML 清洗
- PDF 招生简章解析
- Playwright 动态网页截图
- LLM 生成的临时分析脚本
- 异常网页/PDF 的 Eval 样本

不适合放进 sandbox 的是：

- LangGraph 主流程
- Pydantic 校验
- Tool Gateway
- Evidence Ledger
- Quality Gate

口径：

> Sandbox 是可选增强，不是当前 MVP 的核心。简历里如果没实现，就不写。

## 17. Google Scholar / 数据源

### Q52: 为什么不用 Google Scholar？

因为它不适合作为自动化 Agent 的核心数据源：

- 没有稳定公开 API
- 自动访问容易被限制
- 结果排序不够可复现
- author identity 不适合严肃消歧
- 不利于 Eval

所以优先 OpenAlex、Semantic Scholar、ORCID、DBLP、CiNii、KAKEN、J-STAGE。

面试回答：

> 我不是不用 Google Scholar，而是不把它作为程序化依赖。这个项目强调可复现和可评测，所以优先使用有 API 和稳定 ID 的来源。

### Q53: 为什么先接 OpenAlex，不先接 Semantic Scholar？

OpenAlex 对 MVP 更方便：

- 作者搜索简单
- works filter 支持 author id
- 返回 affiliations、works_count、cited_by_count
- API 使用门槛低

Semantic Scholar 更适合下一步补：

- abstract
- citation intent
- recommended papers
- paper details

## 18. 当前不足和下一步

### Q54: 当前项目最大的技术不足是什么？

最主要不足：

1. MCP Web Search 还是配置插槽，没有默认绑定具体搜索 server
2. 官网正文抽取还没做，所以招生信息通常缺失
3. 匹配度还是关键词匹配，没有 embedding/abstract 语义分析
4. Eval 样本太少，还没有真实指标
5. 还没有 Human-in-the-loop interrupt 节点
6. 缓存是内存级，没有持久化

回答：

> 当前 MVP 先验证架构闭环，下一阶段重点是真实网页抽取、Semantic Scholar 补充摘要、Human-in-the-loop 和完整 Eval。

### Q55: 下一步最应该做什么？

最优先三步：

1. 接一个真实 MCP Web Search server，让 `discover_web_sources` 真的返回学校官网/实验室主页/招生页面。
2. 增加网页抽取工具，把搜索结果页面抓取正文，转成 `Evidence`，识别 official source。
3. 增加 Human-in-the-loop，当 `resolved_professor.is_ambiguous=True` 时，用 LangGraph interrupt 让用户确认候选教授。

## 19. 面试高频追问

### Q56: 如果 OpenAlex API 挂了怎么办？

当前 Tool Gateway 会：

- 捕获异常
- 根据 retry 配置重试
- 记录 `tool_error` 或 `timeout`
- 把错误写入 `tool_logs`
- 让流程尽量返回 partial

后续增强：

- Semantic Scholar fallback
- 本地缓存 fallback
- source-level confidence 降级

### Q57: 如果 MCP 没配，系统还能跑吗？

能跑。MCP 工具会返回 `skipped / mcp_not_configured`，OpenAlex 仍然可以查教授和论文。最终报告大概率是 `partial`，因为缺少官方网页和招生来源。

这体现了 graceful degradation。

### Q58: 如何避免 hallucination？

当前主要靠四件事：

1. 结构化 Schema 限制输出字段
2. Evidence Ledger 要求关键事实绑定来源
3. `is_inference` 区分事实和推断
4. Quality Gate 拦截无来源事实和缺失官方来源

回答：

> 我没有假设 LLM 不会幻觉，而是把 LLM 输出放进结构化和证据校验流程里。

### Q59: 你怎么证明工具调用过程合理？

靠 `ToolCallLog` 和 Trace Eval。

每次工具调用记录：

```text
tool_name
arguments_summary
status
attempts
duration_ms
cache_hit
error_type
```

Eval 可以断言：

- OpenAlex 是否被调用
- MCP 是否被调用或明确 skipped
- 是否超过 max_tool_calls
- ambiguous 时是否进入 needs_review

### Q60: 这个项目为什么不是普通 RAG？

普通 RAG 通常是：

```text
检索文档 → 塞给模型 → 生成回答
```

这个项目是：

```text
任务契约
→ 工具调用
→ 身份消歧
→ 证据建模
→ 多源校验
→ 匹配度推断
→ Quality Gate
→ Trace/Eval
```

RAG 可以作为其中的网页/论文上下文检索模块，但不是系统全部。

## 20. 简历口径

### Q61: 一句话介绍项目

我做了一个面向硕博申请的实验室调研 Agent，用 LangGraph 编排调研流程，用 Pydantic 约束结构化状态，用 DeepSeek V4 做需求解析和报告表达，用 Tool Gateway 管理 OpenAlex 和 MCP 工具调用，并通过 Evidence Ledger、教授身份消歧、Quality Gate 和 Eval 骨架提升结果可靠性。

### Q62: 技术亮点怎么讲？

可以按这个顺序：

1. LangGraph: 把调研拆成多个可追踪节点
2. Pydantic: 定义状态和报告结构
3. Tool Gateway: 控制外部工具调用、预算、重试和缓存
4. OpenAlex: 获取真实教授和论文数据
5. MCP: 预留标准化 Web Search 接入
6. Identity Resolver: 做教授身份消歧
7. Evidence Ledger: 追踪事实来源
8. Quality Gate: 检查报告是否可信
9. LangSmith/Eval: 记录 trace 并做评测闭环

### Q63: 哪些话不要说？

不要说：

- 已经完成生产级 Agent
- 已经完整接入 Google Scholar
- 已经有 30 条高质量 Eval 和真实指标
- 已经完成全自动自进化
- 已经完成完整网页招生信息抽取
- 已经实现 sandbox

可以说：

> 当前是 MVP，已经跑通核心工程骨架和 OpenAlex 真实数据接入；MCP、官网抽取、Human-in-the-loop 和完整 Eval 是下一阶段。

## 21. 面试时最强回答模板

### Q64: 你觉得这个项目最有技术含量的地方是什么？

推荐回答：

> 我觉得不是单个模型调用，而是把一个容易幻觉的 research task 拆成可控流程。比如教授匹配这个场景，最怕找错人和无来源结论。所以我做了三个控制点：第一，Tool Gateway 统一管理外部调用和失败；第二，Evidence Ledger 把事实和来源绑定；第三，Quality Gate 在报告输出前检查身份、来源和推断标记。这样即使结果不完整，系统也会返回 partial 或 needs_review，而不是编一个完整答案。

### Q65: 如果给你两周继续做，你会怎么排优先级？

推荐回答：

> 第一周我会接真实 MCP Web Search 和网页抽取，把学校官网、实验室主页和招生页面转成 Evidence。第二周我会扩展 Eval，重点覆盖同名教授、官网缺失、招生信息冲突和 API 失败。因为这两个方向直接提升系统可信度，比先加多 Agent 更有价值。

### Q66: 这个项目对 AI 产品实习有什么价值？

推荐回答：

> 它体现的是从 demo 到可用 Agent 的工程意识。不是只会调模型，而是考虑用户输入契约、工具失败、身份歧义、证据引用、质量检查和评测闭环。这些能力在真实 AI 产品里很重要，因为用户最终关心的是结果能不能信、出了错能不能定位。
