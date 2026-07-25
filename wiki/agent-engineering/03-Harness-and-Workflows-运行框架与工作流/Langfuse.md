# Langfuse：LLM 应用可观测、评测与 Prompt 管理平台

> 调研快照：2026-07-14。GitHub 约 31.1k Stars，最新 OSS Release 为 v3.213.0。Stars 和版本会持续变化，本文中的数字只代表调研当日状态。

Langfuse 是一个开源 AI Engineering 平台。它不只是展示日志，而是把 LLM 应用的运行轨迹、成本与延迟、质量评分、数据集实验和 Prompt 版本串成持续改进闭环。它可以作为 [Harness Engineering](Harness%20Engineering.md) 中“评估与观测”能力的一种工程实现。

## 一句话结论

如果一个 LLM 或 Agent 项目已经从 Demo 进入持续迭代阶段，需要回答“这次请求经历了什么、为什么失败、新版本是否更好、哪个 Prompt 正在生产运行”，Langfuse 很适合作为统一的可观测与评测平台。

它不替代业务代码、测试系统或 Agent Harness，而是保存这些系统产生的 Trace、指标和 Score，并提供分析、实验和版本关联能力。对于只有少量调用、无需回归评测的个人脚本，直接接入可能比简单结构化日志更重。

## 项目概览

| 维度 | 结论 |
| --- | --- |
| 项目定位 | 开源 AI Engineering 平台 |
| 核心能力 | LLM/Agent Tracing、Metrics、Evaluation、Datasets、Experiments、Prompt Management |
| 主要接入方式 | Python SDK v4、JS/TS SDK v5、OpenTelemetry、框架原生集成 |
| 部署方式 | Langfuse Cloud 或自托管 |
| 开源模式 | Open Core：核心代码 MIT，企业目录使用商业许可证 |
| 典型对象 | Session、Trace、Observation、Score、Dataset、Prompt |

## 核心闭环

Langfuse 的价值不在某一个独立功能，而在于把生产数据转化为下一轮改进依据：

1. **采集**：记录一次用户请求、Agent 运行或 RAG 问答的完整 Trace。
2. **定位**：查看模型调用、工具调用、检索、重试、错误、token、成本和延迟。
3. **评分**：用人工标注、用户反馈、规则或 LLM-as-a-Judge 生成 Score。
4. **沉淀**：把生产中的失败样本或边界案例加入 Dataset。
5. **实验**：离线运行新模型、新 Prompt 或新流程，对比不同 Dataset Run。
6. **发布**：将验证通过的 Prompt 版本或应用版本投入生产。
7. **监控**：继续在线评测，发现新失败样本后回到数据集。

这形成了“生产观测 → 失败样本 → 离线实验 → 发布 → 在线监控”的循环。

## 数据模型

### Session、Trace 与 Observation

Langfuse 的运行时数据可以理解为三层：

~~~mermaid
flowchart LR
    S["Session：一次会话或连续任务"] --> T1["Trace：一次请求或 Agent run"]
    S --> T2["Trace：下一轮请求或 run"]
    T1 --> O1["Observation：检索"]
    T1 --> O2["Observation：模型生成"]
    T1 --> O3["Observation：工具调用"]
~~~

- **Session**：聚合多条相关 Trace，例如一段多轮对话或一个长期 Coding Session。
- **Trace**：一个自包含工作单元，例如一次对话轮次、一次 Agent run 或一次数据处理任务。
- **Observation**：Trace 中的单个步骤，可以嵌套形成树。

常用 Observation 类型包括：

- **generation**：模型生成，适合记录模型、输入输出、token、成本和首 token 延迟。
- **tool**：工具调用，例如读文件、运行命令或访问外部 API。
- **retriever**：检索步骤，例如向量召回或知识库查询。
- **span**：普通处理步骤，例如解析、重排、路由和验证。
- **event**：没有持续时间的离散事件。

Trace 还可携带 user、session、environment、release、version、tag 和 metadata，用于筛选、聚合和版本对比。

### Score

Score 是 Langfuse 保存评测结果的统一对象，可以关联 Trace、Observation、Session 或 Dataset Run。它支持数值、类别、布尔和文本等类型。

Score 的来源可以是：

- 用户点赞或点踩；
- 人工标注与 Annotation Queue；
- 确定性的 Python/TypeScript 规则；
- LLM-as-a-Judge；
- 外部评测系统通过 SDK 或 API 回写。

因此，Langfuse 负责统一保存和分析评测结果，但不意味着所有质量指标都由 Langfuse 自动计算。

### Dataset 与 Experiment

- **Dataset**：一组稳定的测试样本。
- **Dataset Item**：输入、可选期望输出、metadata，以及可选的来源 Trace。
- **Task**：被测应用逻辑。
- **Evaluator**：对单条结果或整次实验评分的函数。
- **Dataset Run**：一个版本在整个 Dataset 上的执行结果。

将 Dataset Item 关联回生产 Trace，可以解释样本为什么进入测试集；比较多个 Dataset Run，则可以判断模型、Prompt 或流程变更是否真的带来收益。

### Prompt

Langfuse 支持 Text Prompt 和 Chat Prompt，并提供：

- 不可变的版本历史；
- production、latest 和自定义标签；
- 变量、消息占位符与 Prompt 组合；
- Playground 与数据集实验；
- Prompt 与生成 Observation 的关联；
- SDK 客户端缓存和回退。

生产代码应按稳定标签读取 Prompt，而不是写死版本号。发布时移动 production 标签，异常时将标签指回旧版本即可回滚。由于 SDK 有缓存，新版本不保证在每个实例上瞬时生效，应根据业务要求设置 TTL 或预取与回退策略。

## 接入方式

### SDK 与 OpenTelemetry

官方主要维护 Python SDK v4 和 JS/TS SDK v5，其他语言可以通过 OpenTelemetry 接入。Langfuse SDK 构建在 OpenTelemetry 之上，因此能够：

- 复用上下文传播和嵌套 Span；
- 同一份遥测同时输出到 Langfuse 与其他可观测平台；
- 接收框架自动生成的 LLM、工具和检索 Trace；
- 在应用后台批量发送数据，减少请求链路额外延迟。

官方集成覆盖 OpenAI SDK、Anthropic、LangChain/LangGraph、LlamaIndex、LiteLLM、OpenAI Agents SDK、Claude Agent SDK、Ragas、Dify、Ragflow 等。选择集成时，应优先使用官方或框架原生集成，再对缺失的业务步骤做手工 Observation。

[AgentScope 2.0](../04-References-%E9%A1%B9%E7%9B%AE%E5%8F%82%E8%80%83/AgentScope.md) 提供 TracingMiddleware，为 Agent Reply、Model Call 和 Tool Execution 生成 OpenTelemetry Span；配置指向 Langfuse 的 OTLP Exporter 后即可接入，但 Exporter 鉴权、Span 映射和敏感字段脱敏需要显式处理。

### 最小 Python 示例

安装并配置：

~~~bash
pip install langfuse

export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
~~~

创建一个 Observation：

~~~python
from langfuse import get_client

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="process-request",
) as span:
    span.update(input={"question": "Langfuse 是什么？"})
    answer = "一个开源 AI Engineering 平台"
    span.update(output={"answer": answer})

# CLI、批处理和其他短生命周期进程退出前必须主动刷新。
langfuse.flush()
~~~

生产接入还应补充 environment、release、session_id、user_id、稳定 Observation 名称、错误状态和必要 metadata。

## Trace 设计原则

Langfuse 的评测器、Dashboard、实验和 Saved View 都依赖 Trace 结构，因此 Trace Schema 应像 API 一样设计。

- **边界稳定**：一个 Trace 对应一个自包含工作单元；多轮过程用 Session 聚合。
- **名称低基数且稳定**：使用 retrieve-context、run-tests、generate-answer，不把用户 ID、模型名和重试次数拼进名称。
- **类型准确**：模型调用使用 generation，工具使用 tool，检索使用 retriever。
- **输入输出可读**：根 Observation 保存审阅者最需要看到的输入和最终输出，原始载荷放 metadata。
- **版本可追踪**：记录应用 release、模型、Prompt 版本、工具策略和索引版本。
- **短任务主动 flush**：后台批处理尚未发送完时进程退出会丢 Trace。
- **控制噪声**：不把所有 HTTP、数据库和框架内部 Span 都当成业务步骤。

这里的机器运行 Trace 与 专家 Trace 要求 所讨论的任务沟通 Trace 不相同：前者由程序自动采集执行证据，后者强调用户和专家之间如何把目标、约束、反馈与验收表达清楚。两者结合后，既能读懂“要做什么”，也能验证“实际做了什么”。

## 自托管架构

Langfuse Cloud 最容易开始；自托管适合数据驻留、内网或深度运维控制要求较高的场景。

~~~mermaid
flowchart LR
    SDK["SDK / OTLP / API"] --> WEB["Langfuse Web"]
    WEB --> S3["S3 / Blob Storage：原始事件与附件"]
    WEB --> REDIS["Redis / Valkey：缓存与队列"]
    WEB --> PG["Postgres：事务数据"]
    REDIS --> WORKER["Langfuse Worker"]
    S3 --> WORKER
    WORKER --> CH["ClickHouse：Trace、Observation、Score"]
    WORKER --> PG
    WORKER -.-> LLM["LLM API / Gateway：Playground 与评测的可选依赖"]
~~~

核心组件包括：

- **Langfuse Web**：UI 和 API；
- **Langfuse Worker**：异步处理事件和后台任务；
- **Postgres**：项目、Prompt、Dataset 等事务数据；
- **ClickHouse**：Trace、Observation、Score 和分析查询；
- **Redis/Valkey**：队列与缓存；
- **S3/Blob Storage**：原始事件、多模态内容和批量导出。

官方 Docker Compose 适合本地试用和低规模部署，但没有高可用、水平扩展和完整备份能力。官方 VM 示例建议至少 4 核、16 GiB 内存和约 100 GiB 存储；生产高可用场景应考虑 Kubernetes 或云 Terraform 方案，并分别备份 Postgres、ClickHouse 和对象存储。

Postgres 与 ClickHouse 必须使用 UTC 时区，否则可能出现查询为空或时间范围错误。自托管虽然避免把原始 Trace 交给 SaaS，但也意味着需要承担数据库、队列、对象存储、升级、迁移、监控和容量规划成本。

## 许可证与数据安全

Langfuse 采用 Open Core：

- 根 LICENSE 声明，除 ee、web/src/ee、worker/src/ee 等企业目录外，代码使用 MIT Expat License。
- 企业目录由 Langfuse Enterprise License 管理，生产使用需要有效商业许可证。
- 官方说明核心功能和 API 在 OSS 中可用，核心数据规模不因 OSS 许可证被人为限制。

需要企业许可证的功能包括项目级 RBAC、受保护的 Prompt 标签、数据保留策略、审计日志、服务端数据脱敏、UI 定制、组织创建控制、SCIM/组织管理 API 等。具体边界可能随版本变化，部署前应再次核对官方 License Key 页面。

数据安全实践：

- Prompt、上下文、工具返回和模型输出可能含源码、个人信息或密钥，不应默认全量上传。
- 在 SDK 导出前做字段级脱敏；OSS 可做客户端 masking，服务端 ingestion masking 属于企业功能。
- 对高流量项目设置 sampling、batch 和保留策略。
- 用 environment 隔离 development、staging 和 production。
- 自托管 OSS 会发送聚合部署遥测，但不发送原始 Trace、Prompt、Observation、Score 或 Dataset 内容；可设置 TELEMETRY_ENABLED=false 关闭。企业版许可证遥测不能关闭。

## 优势与限制

### 优势

- Trace、评测、Prompt、Dataset 和成本指标共享同一数据模型，减少工具拼接。
- OpenTelemetry 降低厂商锁定，可以和现有基础设施监控并存。
- 同时支持在线质量监控与离线回归实验。
- Cloud 和自托管共用代码库，迁移路径相对明确。
- Prompt 版本能关联到真实 Trace 和评测结果，不只保存文本历史。

### 限制与成本

- 自托管生产架构比普通日志系统更重，至少涉及四类存储/中间件。
- 高质量结果依赖良好的 Trace Schema；自动采集不等于可用的可观测性。
- LLM-as-a-Judge 仍可能偏置、漂移并产生额外成本，需要规则和人工抽检校准。
- 全量记录输入输出可能引发隐私、合规和存储成本问题。
- SDK 和平台升级较快，应锁定版本并阅读迁移指南。
- Langfuse 能看到和比较执行结果，但不会替代测试脚本、权限沙箱、业务验收或事故响应。

## MiniCode 接入设计（建议，尚未实现）

MiniCode 幻觉评测与版本控制 已经提出 Trace、回放、固定任务集、成本和版本对比。Langfuse 可以承载其中的观测与评测数据，但下述内容是后续设计建议，不代表 MiniCode 当前已经依赖或接入 Langfuse。

### 对象映射

| MiniCode 概念 | Langfuse 对象 | 建议 |
| --- | --- | --- |
| .port_sessions 会话 | Session | session_id 沿用本地会话 ID |
| 一次 Agent run | Trace | 根输入为用户任务，根输出为最终报告 |
| 模型请求 | generation | 记录模型、token、成本、Prompt 版本和 stop reason |
| 文件、Shell、MCP 调用 | tool | 记录工具名、参数摘要、状态、时长和错误 |
| Memory/Skill 召回 | retriever | 记录候选数量、命中项和召回策略版本 |
| Planner、Reviewer、Verifier | span 或 generation | 保留角色、输入输出与父子关系 |
| 应用和策略版本 | release、version、metadata | 记录 Git commit、工具策略和模型配置 |

Observation 名称应保持稳定，例如 plan-task、read-file、run-tests、review-result、verify-evidence。动态路径、用户 ID 和重试序号放入 metadata。

### 评测闭环

1. 从失败、人工接管和高成本 Trace 中抽取 Dataset Item。
2. 保存初始仓库、任务、预期行为、验证脚本和风险标签。
3. 新版本 Agent 在固定 Dataset 上离线运行。
4. 写入 task_success、tests_passed、hallucination、policy_violation 等 Score。
5. 同时比较 token、成本、延迟、工具调用次数和失败类型。
6. CI 中设置关键 Score 门槛，未达到门槛不发布。
7. 灰度阶段继续在线评分，将新边界案例补回 Dataset。

源码、Shell 输出和密钥不应无筛选进入 Trace。建议只记录必要摘要、内容哈希和证据路径，对可能含密钥的环境变量与文件内容在客户端脱敏。

## RAG 接入设计（建议，尚未实现）

对于 知识库问答优化与编排，建议把一次用户问题到最终回答定义为一个 Trace，多轮问答通过 Session 关联。

核心 Observation：

1. **retrieve-context / retriever**：记录查询、索引版本、数据源、Top-K、过滤条件和召回文档 ID。
2. **rerank-context / span**：记录候选数量、重排器版本、最终上下文及分数。
3. **generate-answer / generation**：记录 Prompt 版本、模型、引用上下文、回答、token、成本和延迟。

建议 Score：

- context_relevance：召回上下文与问题的相关性；
- context_recall：应检索证据是否被覆盖；
- faithfulness：回答是否忠于检索证据；
- answer_relevance：回答是否真正解决问题；
- citation_correctness：引用是否指向支持结论的片段；
- user_feedback：用户反馈。

Ragas 等外部评测器可以计算 RAG 指标，再通过 SDK/API 将结果写成 Langfuse Score。Langfuse 负责统一关联、展示和比较，不应把外部评测结果误写成平台自动生成。

当线上出现低忠实度、错误引用或旧索引命中时，可以把对应 Trace 转为 Dataset Item，保留 sourceTraceId，并用不同 Top-K、重排器、Prompt、模型或索引版本运行对照实验。

## 选型建议

适合优先采用 Langfuse：

- 已有稳定业务流量，需要定位 LLM/Agent 失败原因；
- 正在做 Prompt、模型、检索或 Agent 策略的持续回归；
- 需要把成本、延迟和质量放到同一版本维度比较；
- 需要 Cloud 快速起步，或有明确自托管数据要求；
- 团队愿意维护稳定 Trace Schema 和评测数据集。

可以暂缓：

- 仍处于一次性 Demo，调用链很短；
- 没有评测集、版本管理或线上监控需求；
- 无法安全处理 Trace 中的敏感数据；
- 不愿承担自托管组件成本，又不能使用 SaaS。

建议 MiniCode 或 RAG 项目先用 Langfuse Cloud/本地 Docker Compose 做小规模验证，确认 Trace Schema、数据量、脱敏和评测流程有效后，再决定是否生产自托管。

## 官方资料

- [Langfuse GitHub 仓库](https://github.com/langfuse/langfuse)
- [v3.213.0 Release](https://github.com/langfuse/langfuse/releases/tag/v3.213.0)
- [项目根 License](https://github.com/langfuse/langfuse/blob/main/LICENSE)
- [Enterprise License](https://github.com/langfuse/langfuse/blob/main/ee/LICENSE)
- [官方文档总览](https://langfuse.com/docs)
- [Observability 数据模型](https://langfuse.com/docs/observability/data-model)
- [Trace 最佳实践](https://langfuse.com/docs/observability/best-practices)
- [SDK 文档](https://langfuse.com/docs/observability/sdk/overview)
- [Evaluation 核心概念](https://langfuse.com/docs/evaluation/core-concepts)
- [Prompt Management 数据模型](https://langfuse.com/docs/prompt-management/data-model)
- [Self-hosting 架构](https://langfuse.com/self-hosting)
- [Enterprise 功能边界](https://langfuse.com/self-hosting/license-key)
- [Self-hosted Telemetry](https://langfuse.com/self-hosting/security/telemetry)

## 相关笔记

- **实现工具**：[Harness Engineering](Harness%20Engineering.md) — Langfuse 将 Harness 的 Trace、指标、评测集和版本比较落实为统一平台。
- **项目应用**：MiniCode 幻觉评测与版本控制 — 可用 Langfuse 承载 Agent 运行轨迹、离线回归和线上质量评分。
- **对比**：专家 Trace 要求 — 专家 Trace 强调任务沟通质量，Langfuse Trace 强调机器运行证据。
- **项目应用**：知识库问答优化与编排 — 可观测检索、重排和生成步骤，并用 Score 驱动 RAG 优化闭环。
- **观测集成**：[AgentScope 2.0](../04-References-%E9%A1%B9%E7%9B%AE%E5%8F%82%E8%80%83/AgentScope.md) — AgentScope 的 OpenTelemetry Span 可导出到 Langfuse 统一分析 Agent、模型和工具调用。
