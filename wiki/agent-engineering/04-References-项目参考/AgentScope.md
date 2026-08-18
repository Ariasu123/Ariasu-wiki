# AgentScope 2.0：生产型 Agent 框架与多智能体服务

> 调研快照：2026-07-14。GitHub 约 27.9k Stars，最新稳定 Release 为 v2.0.4，Python 要求 3.11+，许可证为 Apache-2.0。官方 latest 文档当日指向 2.0.5dev，因此本文会区分“稳定版已有能力”和“main/latest 文档中的开发中能力”。

AgentScope 是阿里巴巴通义实验室 SysML 团队开源的 Python Agent 框架。它把 [Agent 核心问题](../01-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/Agent_16.md) 中的 ReAct、工具、上下文、状态、权限、人机协作和多 Agent 通信落实为可组合的运行时，并提供 FastAPI Agent Service 将单个 Agent 扩展成多租户、多会话服务。

## 一句话结论

AgentScope 2.0 的核心价值不是提供更多 Prompt 模板，而是给自主 Agent 建立一套明确的运行协议：

- Agent 用统一 ReAct 循环驱动模型与工具；
- Message 保存完整结果，Event 提供可重放的流式过程；
- AgentState 显式保存上下文、权限和暂停状态；
- Middleware 在关键生命周期注入追踪、预算、记忆和 RAG；
- Workspace 隔离工具、文件、MCP 和 Skill；
- Agent Service 负责会话、持久化、事件流、后台任务和多 Agent 团队。

它适合需要在 Python 中深度定制 Agent Runtime，同时希望复用生产服务骨架的团队。对于简单聊天机器人、固定 DAG 工作流或只想直接使用现成个人助理的场景，AgentScope 可能过重。

## 项目快照

| 维度 | 结论 |
| --- | --- |
| 项目 | agentscope-ai/agentscope |
| 维护方 | Alibaba Tongyi Lab SysML Team |
| 当前稳定版 | v2.0.4，发布于 2026-07-07 |
| 主要语言 | Python 3.11+ |
| 许可证 | Apache License 2.0 |
| 核心定位 | 可扩展 Agent Runtime + 多租户 Agent Service |
| 基础范式 | 模型原生推理与工具调用驱动的 ReAct |
| 关键扩展 | Event、Permission、Middleware、Workspace、RAG、Long-term Memory、Agent Team、OpenTelemetry |

## AgentScope 2.0 的分层

~~~mermaid
flowchart TB
    UI["CLI / Web UI / API Client"] --> SERVICE["Agent Service：会话、SSE、调度、后台任务"]
    SERVICE --> AGENT["Agent：ReAct 推理—行动循环"]
    AGENT --> EVENT["Message + Event"]
    AGENT --> MW["Middleware"]
    AGENT --> STATE["AgentState"]
    AGENT --> TOOLKIT["Toolkit：Tools / MCP / Skills"]
    TOOLKIT --> WS["Workspace / Sandbox"]
    MW --> MODEL["Model + Formatter + Credential"]
    MW --> MEMORY["Memory / RAG / Tracing / Budget"]
    SERVICE --> TEAM["Agent Team：Leader + Worker Sessions"]
    SERVICE --> REDIS["Redis Storage + Message Bus"]
~~~

可以把它分成三层：

1. **Agent 内核层**：Agent、Model、Message、Event、Toolkit、Permission、Context、State、Middleware。
2. **执行环境层**：Workspace、Sandbox、MCP Gateway、Skill、RAG 与长期记忆。
3. **服务层**：FastAPI Agent Service、Redis 持久化与消息总线、SSE、调度、后台任务、Agent Team 和 Web UI 示例。

这与 [Harness Engineering](../03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Harness%20Engineering.md) 的思路一致：模型负责决策，Harness 负责工具、上下文、状态、权限、恢复和运行环境。AgentScope 主要覆盖运行时 Harness；独立评测集和发布门禁仍需要项目自己建设或接入外部平台。

## Agent 内核

### 显式状态的 ReAct 循环

官方把 Agent 描述为“无状态的推理—行动循环引擎”。这里的无状态不是说对话不能持续，而是循环逻辑与状态分离：上下文、PermissionContext、任务状态和当前 reply 位置保存在 AgentState 中。

每次 reply 或 reply_stream 大致执行：

1. 接收 Message 或恢复 Event；
2. 写入上下文，并在需要时压缩；
3. 调用模型进行推理；
4. 没有工具调用则返回最终 Message；
5. 有工具调用则做权限判断；
6. ALLOW 时执行，DENY 时把错误反馈给模型；
7. ASK 或外部执行时暂停并发出 Event；
8. 收到确认或外部结果后，从一致状态继续。

Agent 还支持取消正在执行的协程，或中断处于 HITL 等待状态的回复。中断后会修复工具状态，使下一条消息可以继续，而不是留下半完成的上下文。

### Message 与 Event

- **Message** 是 Agent 间通信和持久化单位，一次完整回复对应一个 Message。
- **Event** 是前端交互和流式过程单位，记录文本增量、思考块、工具调用、工具结果、权限请求和外部执行。
- 同一 reply 的 Event 共享 reply_id，并能通过 append_event 重建最终 Message。

常用内容块包括 TextBlock、DataBlock、ThinkingBlock、ToolCallBlock、ToolResultBlock 和 HintBlock。HintBlock 用于注入团队消息、定时任务和后台工具结果等系统提示。

这种“事件是过程、消息是结果”的设计，使流式 UI、断线重放、人机确认和状态持久化共享同一套数据协议。

## 工具、MCP 与 Skill

Toolkit 将以下能力统一为 Agent 可调用工具：

- 内置 Bash、Read、Write、Edit、Glob、Grep 和任务管理工具；
- 自定义 ToolBase 或函数工具；
- Stdio/HTTP MCP Client；
- 文件系统或沙箱中的 Skill；
- ToolGroup 与按需激活；
- 外部执行工具。

Agent 会根据工具的并发安全标记决定并行或顺序执行。外部执行工具不会在 Agent 进程内直接运行，而是发出 RequireExternalExecutionEvent，等待外部系统返回结果后恢复。

MCP、Skill 和工具组是能力装配机制，不等于权限控制；真正执行前仍会经过 Permission System。

## 权限与 Human-in-the-Loop

每次工具调用都会得到 ALLOW、DENY 或 ASK。决策由规则、全局模式和工具对实际参数的安全检查共同产生。

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| DEFAULT | 未匹配规则时通常询问用户 | 默认交互式使用 |
| EXPLORE | 只允许只读操作 | 搜索、分析和规划 |
| ACCEPT_EDITS | 工作目录内的编辑可自动通过，高风险操作仍询问 | 用户在场的开发任务 |
| DONT_ASK | 将所有 ASK 转为 DENY | 无人值守和定时任务 |
| BYPASS | 跳过大部分安全询问，仅保留显式 deny/ask 规则和工具 DENY | 已有强沙箱的可信环境 |

BYPASS 不代表“更智能”，而是主动放弃内置安全确认。官方文档明确指出，危险路径和命令注入类安全 ASK 在 BYPASS 下也会被跳过，因此只能在可靠沙箱和额外 deny 规则下使用。

当决策为 ASK 时，Agent 暂停并发出 RequireUserConfirmEvent；用户可以允许、拒绝、修改参数，或接受建议规则，让后续同类调用自动处理。

## Context、State 与长期记忆

### 上下文压缩

ContextConfig 控制：

- 达到模型上下文比例后自动压缩；
- 保留最近消息的比例；
- 工具结果最大 token；
- 超长结果是否通过 Offloader 落盘。

压缩后的历史和超长工具输出可以移入 Workspace，Prompt 只保留摘要和路径，需要时再读取。这降低上下文膨胀，但摘要仍可能遗漏证据，关键任务应保留原始文件引用。

### 状态持久化

AgentState 是可序列化的 Pydantic 模型。Agent Service 的 RedisStorage 按 user_id、agent_id、session_id 保存状态，让服务可以在不同请求间重建 Agent。

框架提供状态结构和 Redis 参考实现，但业务仍需决定：

- 会话保存和过期策略；
- 多租户密钥与数据隔离；
- 状态 Schema 升级；
- 失败恢复和幂等；
- 敏感内容加密。

### 长期记忆

v2.0.4 已包含 Agentic Memory、Mem0 和 ReMe 的 Middleware 集成。它们在 reply/reasoning 生命周期中召回或写入跨会话记忆。

长期记忆不是上下文压缩的替代品：压缩解决单次会话 token 压力，长期记忆解决跨会话信息沉淀。记忆写入还需要去重、置信度、过期和隐私策略。

## Middleware：可组合运行时扩展

Middleware 可围绕完整 reply、单轮 reasoning、单次 acting、模型调用、上下文压缩和系统 Prompt 注入逻辑，而不修改 Agent 核心代码。

内置 Middleware 包括：

- TracingMiddleware：OpenTelemetry Trace；
- ReplyBudgetControlMiddleware：单次回复 token 预算；
- RAGMiddleware：知识库检索；
- AgenticMemoryMiddleware、Mem0Middleware、ReMeMiddleware：长期记忆；
- TTSMiddleware：语音合成。

Middleware 很适合日志、审计、缓存、策略和数据脱敏，但顺序会影响行为。多个 Middleware 同时修改 Prompt、工具或事件时，应测试执行顺序和异常传播。

## Workspace 与沙箱

Workspace 为 Agent 提供统一的文件系统、内置工具、MCP、Skill 和 Context Offloader。相同 Agent 代码可以切换执行后端。

官方 main/latest 文档展示了 Local、Docker、E2B、Kubernetes、OpenSandbox 和 Daytona 等后端；其中部分是 v2.0.4 之后进入 main 的开发中能力，使用稳定包前必须核对对应版本。

安全边界取决于后端：

- LocalWorkspace 直接操作宿主机，权限系统不是操作系统级沙箱；
- Docker/E2B/OpenSandbox/Daytona 等提供更强隔离，但仍要限制网络、挂载、密钥和资源；
- WorkspaceManager 可以按 agent、session 或 user 设置隔离粒度；
- MCP Gateway 让宿主 Agent 访问沙箱内部 MCP Server。

## RAG 能力

AgentScope 2.0 的 RAG 模块提供 Parser、Chunker、Embedding、Vector Store 和 KnowledgeBase 抽象。

当前文档明确展示：

- Text、PDF、PPT、Image Parser；
- ApproxTokenChunker；
- Qdrant 与 Milvus Lite；
- 文档级插入、查询、列举和删除；
- metadata_filter 多租户过滤；
- 文本和多模态检索；
- RAGMiddleware 的 static 与 agentic 两种模式。

static 模式在推理前固定检索并注入上下文；agentic 模式把检索暴露为工具，由模型判断何时查询。它适合快速搭建 Agent 内知识访问，但不是完整的知识库质量平台：清洗、版本一致性、混合检索、重排、引用验证和评测仍需要额外实现。

## Agent Service

Agent Service 是基于 FastAPI 的托管层。它把 Agent 变成多租户、多会话 HTTP 服务，主要提供：

- Agent、Credential、Model、Session、Workspace 和 Knowledge Base API；
- 每个 Session 的 SSE Event Stream 与历史重放；
- RedisStorage 和 RedisMessageBus；
- 会话锁、Inbox、Wakeup 与跨进程消息；
- 后台工具卸载和结果唤醒；
- Cron 调度；
- Agent 中断；
- Workspace 生命周期和 TTL；
- 示例 Web UI。

最小部署需要 Storage、MessageBus 和 WorkspaceManager。Redis 同时承担持久化、锁、事件回放、Inbox 和 Wakeup 等职责，因此生产环境需要认真设计 Redis 高可用、容量、备份和故障恢复。

重要边界：

- 默认 X-User-ID 只是占位 Header，不是身份认证；生产必须替换为 JWT、OAuth 或其他认证。
- 官方将多进程/多节点分布式部署标为 WIP，不能只因使用 Redis 就假设已经具备完整生产弹性。
- 示例 Web UI 是参考实现，不等于完整的权限管理、审计、配额和运营后台。

## Agent Team

Agent Team 建立在 Agent Service 之上。用户面对的 Session 是 Leader，每个 Worker 也是独立 Session，拥有自己的 State、Workspace 绑定和 Event Stream。

Leader 通过 TeamCreate、AgentCreate、AgentInvite、TeamSay 和 TeamDelete 等工具创建或管理团队。成员消息写入 Redis Message Bus 的 Inbox，Wakeup Dispatcher 在任意服务进程中唤醒目标 Session，InboxMiddleware 再把消息作为 HintBlock 注入下一轮推理。

特点：

- Worker 可以并发运行，而不是 Leader 协程里的嵌套函数；
- SubAgentTemplate 可为 explorer、coder 等角色预设 Prompt、Permission、Context 和 Task；
- Leader 负责动态分工和汇总；
- 通信依赖内部 Message Bus；协议适配可通过 Service Middleware 扩展。

与 MiniCode 多智能体 相比，二者都采用 Leader/主 Agent 控制子 Agent、角色权限收窄和独立 Session。AgentScope 进一步提供 Redis Inbox/Wakeup、并发 Worker、服务化状态和 Web Event Stream；MiniCode 更轻、更贴近本地 Coding Agent，当前以串行或拓扑分批委派为主。

## OpenTelemetry 与 Langfuse

TracingMiddleware 会为 Agent Reply、Model Call 和 Tool Execution 创建层级化 OpenTelemetry Span，并记录 Session、reply、模型、token、消息、工具参数、结果、HITL 和外部执行状态。

这意味着 AgentScope 可以接入 [Langfuse](Langfuse.md)：为 OpenTelemetry 配置指向 Langfuse OTLP Endpoint 的 Exporter，再在 Agent 上启用 TracingMiddleware。它不是 AgentScope 内置的一键 Langfuse Connector，Exporter、鉴权、Span 映射和脱敏仍需显式配置。

模型输入输出、工具参数和结果可能含源码、个人信息和密钥。开启 TracingMiddleware 前应检查采集字段，并在 Exporter 或 Middleware 层做过滤、采样和脱敏。

## 最小示例

安装：

~~~bash
uv pip install agentscope
# 或 pip install agentscope
~~~

创建并调用一个 Agent：

~~~python
import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import DashScopeCredential
from agentscope.message import UserMsg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, Read, Grep


async def main() -> None:
    agent = Agent(
        name="researcher",
        system_prompt="你是一个只基于证据回答的研究助手。",
        model=DashScopeChatModel(
            credential=DashScopeCredential(
                api_key=os.environ["DASHSCOPE_API_KEY"],
            ),
            model="qwen-plus",
        ),
        toolkit=Toolkit(tools=[Read(), Grep()]),
    )

    result = await agent.reply(
        UserMsg(name="user", content="分析当前项目的 README。"),
    )
    print(result.get_text_content())


asyncio.run(main())
~~~

真实项目还应补充 PermissionContext、Workspace、Middleware、状态持久化和异常处理，而不是直接给 Bash、Write 等高风险工具开启 BYPASS。

## 与 Hermes Agent、MiniCode 的区别

| 对比项 | AgentScope | Hermes Agent | MiniCode |
| --- | --- | --- | --- |
| 主要形态 | Python SDK + Agent Service 框架 | 可直接运行的通用个人 Agent 平台 | 自研本地 Coding Agent 项目 |
| 核心优势 | Event、Permission、Middleware、Workspace、服务化和团队 | 多入口、长期任务、个人自动化生态 | 可控源码、Coding 工具链和面试项目深度 |
| 多 Agent | Redis Message Bus 驱动的 Leader/Worker Session | 更偏平台任务编排 | 主 Agent 受控委派子 Agent |
| 扩展方式 | Python 抽象、Middleware、Tool、MCP、Skill | Skill、MCP、插件和消息入口 | 自研 Runtime、Tool、Plugin、MCP |
| 适合 | 构建自有 Agent 产品与服务 | 快速使用个人通用 Agent | 学习、定制和演进 Coding Agent |

Hermes Agent 更像已经装配好的 Agent Operating Layer；AgentScope 更像构建这类产品的框架和服务骨架。两者都不是单纯聊天 UI，但抽象层级和目标用户不同。

## 优势

- Agent 内核、Event、State 和 Service 边界清晰，适合前后端分离。
- Permission、HITL、Workspace 和中断是核心能力，而非示例附加项。
- Middleware 覆盖模型、工具、上下文和完整 reply 生命周期。
- Toolkit 统一内置工具、MCP、Skill 和工具组。
- Agent Team 复用 Session、Inbox 和 Wakeup，服务化路径较完整。
- Apache-2.0 便于企业修改和二次分发。

## 限制与风险

- 2.0 是破坏性升级，1.x 的 ReActAgent、Hook、State 和教程不能直接照搬。
- latest 文档当前指向 2.0.5dev，可能早于稳定包；应锁定版本并使用对应文档。
- PyPI 将项目标记为 Beta，且分布式部署在官方文档中仍为 WIP。
- Agent Service 没有内置真实身份认证，生产安全需要自行补齐。
- 框架抽象较多，Middleware、Event、State、Workspace 和 Service 的组合增加学习与排障成本。
- LocalWorkspace 不是强沙箱，BYPASS 模式会跳过关键安全 ASK。
- 内置 RAG 更适合基础检索，复杂知识库仍需重排、版本控制和评测。
- Tracing 会采集完整消息和工具数据，默认全量输出存在隐私风险。
- AgentScope 是运行时框架，不替代离线评测、线上质量监控和 Prompt 版本平台。

## 选型建议

适合采用：

- 需要 Python 原生、可深度扩展的 Agent Runtime；
- 需要流式前端、工具确认、中断与恢复；
- 需要多租户、多会话、后台任务和计划调度；
- 需要 Docker/E2B 等可替换 Workspace；
- 计划构建 Leader/Worker 多智能体产品。

可以暂缓：

- 只是简单问答或少量工具调用；
- 业务本质是固定、确定性的 DAG；
- 团队无法维护 Redis、Workspace、安全策略和状态迁移；
- 已有成熟 Runtime，只缺可观测或评测平台；
- 需要立即获得经过验证的多节点高可用方案。

建议先用稳定版完成单 Agent + DEFAULT/EXPLORE 权限 + Docker Workspace + Tracing 的纵向验证，再引入 RAG、长期记忆和 Agent Team。不要一开始同时启用所有抽象。

## 官方资料

- [AgentScope GitHub 仓库](https://github.com/agentscope-ai/agentscope)
- [中文 README](https://github.com/agentscope-ai/agentscope/blob/main/README_zh.md)
- [v2.0.4 Release](https://github.com/agentscope-ai/agentscope/releases/tag/v2.0.4)
- [Apache-2.0 License](https://github.com/agentscope-ai/agentscope/blob/main/LICENSE)
- [AgentScope 2.0 文档](https://docs.agentscope.io/latest/en/)
- [Agent 核心与 ReAct 循环](https://docs.agentscope.io/latest/en/building-blocks/agent)
- [Message 与 Event](https://docs.agentscope.io/latest/en/building-blocks/message-and-event)
- [Middleware 与 OpenTelemetry](https://docs.agentscope.io/latest/en/building-blocks/middleware)
- [Permission System](https://docs.agentscope.io/latest/en/building-blocks/permission-system)
- [Workspace](https://docs.agentscope.io/latest/en/building-blocks/workspace)
- [RAG](https://docs.agentscope.io/latest/en/building-blocks/rag)
- [Agent Service](https://docs.agentscope.io/latest/en/deploy/agent-service)
- [Agent Team](https://docs.agentscope.io/latest/en/deploy/agent-team)
- [1.x 到 2.0 Changelog](https://docs.agentscope.io/latest/en/others/change-log)

## 相关笔记

- **理论基础**：[Agent 核心问题](../01-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/Agent_16.md) — AgentScope 将 ReAct、工具、状态、记忆和多 Agent 协作落实为框架抽象。
- **框架实现**：[Harness Engineering](../03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Harness%20Engineering.md) — Permission、Workspace、Context、State 和 Middleware 构成具体 Harness。
- **项目对照**：MiniCode 多智能体 — 可比较 Leader/Worker、Session、权限收窄和消息总线设计。
- **观测集成**：[Langfuse](Langfuse.md) — AgentScope 的 OpenTelemetry Span 可导出到 Langfuse 做 Trace、评测和版本分析。
