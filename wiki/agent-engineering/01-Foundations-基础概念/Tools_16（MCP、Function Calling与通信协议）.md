
## 一、总览

关联阅读：可结合 [[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Agent_16|Agent 核心问题]] 理解；Tools 与协议是 Agent 执行闭环的行动接口。


这 16 个问题围绕 LLM 工具调用体系展开：从 Function Calling 的底层机制，到 MCP 的工具生态标准化，再到 Agent Skill、A2A、多种传输协议和 LLM Gateway。

可以用一句话概括：

> Function Calling 解决“模型怎么发起工具调用”，MCP 解决“工具怎么标准化接入”，Skill 解决“Agent 怎么按流程完成任务”，A2A 解决“多个 Agent 怎么协作”。

学习路径建议：

| 模块 | 对应问题 | 核心主题 |
| --- | --- | --- |
| Function Calling | 1-3 | 工具调用原理、模型如何学会调用、训练数据和 SFT/RLHF |
| MCP | 4-8、13 | MCP 定义、组成、和 Function Calling 的关系、选型、传输方式 |
| Skill | 9-11 | Agent Skill、MCP vs Skill、Function Calling/Skill/MCP 层级关系 |
| 多 Agent 协议 | 12 | A2A 协议、Agent 间协作、和 MCP 的区别 |
| 通信协议 | 14-15 | SSE、WebSocket、WebRTC 的区别和适用场景 |
| 网关工程 | 16 | LLM Gateway、多模型统一、配额、成本和语义缓存 |

---

## 1. 什么是 Function Calling？原理是什么？

### 核心回答

Function Calling 是一种让模型以结构化方式发起工具调用的机制。**开发者用 JSON Schema 描述工具**，模型判断需要工具时，不直接输出自然语言答案，而是**输出 `tool_calls` JSON**，告诉宿主程序要调用哪个函数、传什么参数。**真正执行工具的是应用代码，模型只负责决策。**
### 细节

完整流程通常是两轮对话：

- 开发者把工具 schema 随请求传给模型。
- 模型根据用户问题和工具描述判断是否需要调用工具。
- 如果需要工具，**模型输出结构化 `tool_calls`，通常包含**函数名和参数。
- 宿主代码解析 JSON，真正执行函数或 API。
- 工具结果作为 `tool` 消息写回对话。
- 模型基于工具结果生成最终答案。
**并行工具调用**：
模型可以在一次响应里同时输出多个 `tool_calls`，**用 Python 的 `asyncio.gather` 异步任务或者多线程**

---

## 2. LLM 是如何学会调用外部工具的？

### 核心回答

LLM 的工具调用能力不是模型参数变大后自然出现的，而是通过专项训练学出来的。**训练上主要靠 SFT 和 RLHF：SFT 教模型“怎么调工具”，RLHF 教模型“什么时候该调、什么时候不该调”。**
### 细节

SFT 阶段：

- 给模型大量工具调用示范对话。
- 样本包含工具定义、用户问题、模型的结构化调用、工具结果、最终答案。
- 模型通过模仿学会识别工具 schema、判断是否调用、输出规范 JSON。

RLHF 阶段：

- 人类标注不同回答的偏好。
- 奖励模型学习哪种行为更好。
- 主模型通过优化奖励，学会工具调用边界。
- 避免模型“什么问题都调工具”或“该调不调”。

运行时：

- 应用把工具 schema 传给模型。
- 模型输出 `tool_calls`。
- 应用执行工具。
- 工具结果返回给模型。
- 模型再生成最终回答。
### 面试提醒

SFT 解决“会不会调”，RLHF 解决“该不该调”。工具调用能力不是纯预训练自然涌现出来的。

---

## 3. 大模型的 Function Call 能力是怎么训练出来的？

### 核心回答

Function Call 能力主要通过包含完整工具调用流程的数据训练出来。**SFT 阶段让模型学会输出结构化调用格式，RLHF 阶段让模型学会调用边界和偏好。
### 细节

一条典型 SFT 样本会包含：

- `system`：工具定义和使用规则。
- `user`：用户问题。
- `assistant`：模型输出 `tool_calls` JSON。
- `tool`：工具执行结果。
- `assistant`：基于工具结果生成最终答案。

训练数据需要覆盖多种场景：

- 单工具调用。
- 多工具并行调用。
- 多轮工具调用。
- 工具调用失败后的重试。
- 不需要工具时直接回答。
- 参数缺失时追问用户。

数据来源：

- 人工标注：质量高，成本高。
- Self-Instruct/模型生成：成本低，量大，但要过滤幻觉和错误调用。
- 蒸馏：用更强模型生成工具调用样本训练小模型。
### 面试提醒

不要说“预训练就会 Function Call”。预训练最多让模型会描述调用意图，结构化工具调用需要专门 SFT 数据训练。

---

## 4. 什么是 MCP？核心内容是什么？

### 核心回答

MCP 是 Model Context Protocol，模型上下文协议，由 Anthropic 推出的开放协议。它解决的是工具接入碎片化问题：工具提供方按协议实现 MCP Server，支持 MCP 的 AI 客户端就能标准化接入，实现“一次实现，到处复用”。

### 细节
MCP 的核心设计：

- **Client-Server 架构。
- **工具提供方实现 MCP Server。
- AI 应用侧作为 Host，**内部运行 MCP Client。
- 一个 Host 可以连接多个 MCP Server。
- 底层消息格式使用 **JSON-RPC 2.0。

MCP Server 能暴露三类能力：

- **Tools：有副作用的操作，类似function call的函数
- **Resources：只读数据，模型可以从中读取数据
- **Prompts：可复用提示词模板。

Host 启动后会通过 Client 连接各个 MCP Server，并发现它们暴露的 tools、resources 和 prompts，然后把工具描述整理后提供给模型。模型在推理过程中如果需要工具，会输出 tool call，包括工具名和参数。Host 解析这个 tool call，根据工具注册表找到对应的 MCP Client，再由 Client 通过 MCP 协议调用 Server。Server 执行工具后把结构化结果返回给 Client，Client 交回 Host，Host 再把工具结果作为上下文重新喂给模型，让模型继续推理或生成最终回答。
### 面试提醒

MCP 不是 Function Calling 的替代品。Function Calling 是模型发起调用的结构化语言，MCP 是工具接入和复用的生态协议。

[[30-Agent-Engineering-Agent工程/04-References-项目参考/ByteRover|ByteRover]] 提供了具体组合案例：Skill 说明 Agent 何时查询或策展项目记忆，MCP 暴露 `brv-query` 和 `brv-curate` 工具，底层仍由宿主处理模型的 Function Calling。

---

## 5. MCP 由哪几部分组成？

### 核心回答

MCP 可以从三层理解：角色层、能力层、协议层。角色层包括 Host、Client、Server；能力层包括 Tools、Resources、Prompts；协议层包括 JSON-RPC 2.0 和 stdio/Streamable HTTP 等传输方式。
### 细节

角色层：

- **Host**：AI 应用本身，比如 Claude Desktop、IDE、Agent 平台。
- **Client**：Host 内部负责和 MCP Server 通信的模块。
- **Server**：工具提供方实现的独立进程或服务。

能力层：

- **Tools**：执行操作，有副作用，需要权限控制。
- **Resources**：只读数据，无副作用，适合给模型补上下文。
- **Prompts**：提示词模板，用于标准化常用任务。

协议层：

- **JSON**-**RPC 2.0**：统一消息格式。
- **stdio**：本地场景，Client 启动 Server 子进程，通过标准输入输出通信。
- **Streamable** **HTTP**：远程场景，Server 作为 HTTP 服务部署。
### 面试提醒

不要漏掉 Host。MCP 不是简单的 Client + Server 二元结构，Host 是宿主应用，一个 Host 内可以有多个 Client 连接多个 Server。

---

## 6. MCP 和 Function Calling 有什么区别？

### 核心回答

Function Calling 和 MCP 不是竞争关系，而是不同层级的东西。**Function Calling 解决“模型如何表达工具调用请求”，MCP 解决“工具如何标准化打包、注册、发现和复用”**。
### 细节

| 维度 | Function Calling | MCP |
| --- | --- | --- |
| 解决问题 | 模型怎么调用函数 | 工具怎么标准化接入 |
| 层级 | 调用协议/消息格式 | 工具生态协议 |
| 输出 | `tool_calls` JSON | Tools/Resources/Prompts 能力集合 |
| 执行方 | 宿主代码执行函数 | MCP Server 提供工具服务 |
| 复用性 | 通常绑定当前应用 | 一次实现，多客户端复用 |

### 面试提醒

一句话：Function Calling 是“调用语言”，MCP 是“工具生态协议”。MCP 不是 FC 的替代品，而是在更高层管理工具。

---

## 7. 什么场景用 Function Calling，什么场景用 MCP？

### 核心回答

如果只是单个应用接少量工具，工具不需要复用，Function Calling 更简单直接。如果工具要跨项目、跨团队复用，数量多、管理复杂，或者已有现成 MCP Server，就更适合使用 MCP。
### 细节

适合 Function Calling 的场景：

- 快速原型验证。
- 单个应用只接一两个工具。
- 工具只服务当前业务，不需要复用。
- 需要精细控制执行逻辑。
- 部署环境不方便运行额外 Server。

适合 MCP 的场景：

- 工具要跨项目复用。
- 工具有**多个 AI 客户端**要接入。
- 工具数量多，手写 schema 和执行逻辑难维护。
- **团队协作，需要统一工具规范。
- **社区已有成熟 MCP Server。
- Agent 系统工具来源复杂，需要统一管理。

判断核心：

> 这个工具会不会在当前应用之外被复用？

会复用，倾向 MCP；只服务一个小应用，Function Calling 足够。
### 面试提醒

不要一刀切。**Function Calling 简单可控**，**MCP 标准化和复用性强**。选型要看工具规模、复用需求、部署环境和团队协作。

---

## 8. 为什么有些推理模型不支持 MCP 协议？

### 核心回答

根本原因是**推理模型的连续思考范式**和**工具调用的暂停交互范式**存在冲突。推理模型往往需要先完整生成一段 thinking，而工具调用需要模型中途暂停、等待外部执行结果，再继续生成。
### 细节

冲突点：

- 推理模型强调连续思维链，思考过程不能随意中断。
- 工具调用天然是多轮交互：输出调用请求 -> 暂停 -> 等工具结果 -> 继续生成。
- 如果在思考链中间打断，可能破坏推理上下文。
- MCP 底层通常依赖 Function Calling，推理模型如果不擅长 FC，自然也难以支持 MCP。

后续折中方案：

- **让工具调用发生在思考阶段结束后。
- 保证内部推理过程完整，再决定是否调用工具。
- 支持 interleaved thinking，在多次工具调用之间穿插思考，但需要特殊模型和协议支持。

### 面试提醒

不要把“不支持 MCP”简单理解成“没适配”。这是模型生成范式和工具调用交互模式之间的结构性冲突。

---

## 9. Skill 是什么？

### 核心回答

Agent Skill 是把**指令、脚本、模板、参考资料**打包成可复用能力模块的机制。它不是单纯保存 prompt，而是 Agent 能**自动发现、按需加载**、需要时调用资源的一套“操作手册 + 工具箱”。
### 细节

一个 Skill 通常是一个文件夹，里面可以包含：

- `SKILL.md`：说明这个 Skill 做什么、什么时候用、怎么用。
- scripts：可执行脚本。
- references：参考资料。
- assets：示例、资源、样板文件。

Skill 的关键设计：

- 自动发现：Agent 能根据元数据知道有哪些 Skill。
- 按需加载：只有需要时才加载完整说明skill.md。
- **渐进式加载：先读简短描述，再按需读详细指令和资源。
- 上下文友好：避免所有能力一次性塞进 context window。

Skill 和普通 Prompt 的区别：

- Prompt 是一次性临时指令。
- Skill 是持久化、可复用、可发现的能力包。
### 面试提醒

Skill 不是“保存好的 prompt”。它更像一个可复用 SOP，包含指令、流程、脚本和模板，并且支持自动发现和按需加载。Slash Command 需要手动触发，Skill 可以被 Agent 自动发现

---

## 10. MCP 和 Agent Skill 的区别是什么？

### 核心回答

MCP 和 Skill 不是同类概念，而是互补关系。MCP 解决“Agent 怎么**获得外部工具和数据访问能力**”，Skill 解决“Agent 拿到工具后该按什么**流程**完成任务”。
### 细节

| 维度   | MCP                     | Skill              |
| ---- | ----------------------- | ------------------ |
| 定位   | 外部能力接入协议                | **可复用任务能力包（使用说明）   |
| 解决问题 | Agent 能调用哪些工具/数据        | Agent 该如何完成某类任务    |
| 粒度   | 原子操作，如查库、写文件、调 API      | 工作流程，如写报告、审代码、生成简历 |
| 内容   | Tools、Resources、Prompts | 指令、脚本、模板、参考资料      |
| 类比   | 工具箱/电脑软件                | 操作手册/SOP           |

两者协作方式：

- Skill 定义**流程和标准。
- 流程执行中需要**外部能力时调用 MCP 工具。
- MCP 提供“能做什么”，Skill 定义“怎么做才好”。
### 面试提醒

一句话：MCP 是能力，Skill 是方法。MCP 让 Agent 有工具，Skill 教 Agent 用工具完成任务。

---

## 11. Function Calling、Skill、MCP 有什么区别？
### 核心回答

Function Calling、MCP、Skill 位于不同层级。**Function Calling 是底层调用语言，MCP 是中层工具标准化协议，Skill 是上层任务流程和知识封装**。
### 细节

三者关系：
- Function Calling：模型输出结构化 JSON，触发函数调用。
- MCP：把工具、资源、提示词标准化封装成可发现、可复用的服务。
- Skill：把完成某类任务的流程、标准、脚本和模板封装成能力包。

类比：
- Function Calling 是语言。
- MCP 是工具箱。
- Skill 是操作手册。

完整场景：
- Agent 根据 Skill 知道任务流程。
- 某个步骤需要外部数据时，通过 MCP 发现工具。
- 模型用 Function Calling 发起具体工具调用。
- 宿主程序执行工具，并把结果返回给模型。

### 面试提醒

不要把三者当成平行竞争方案。它们是从底层调用、中层工具生态、上层任务流程的三层协作关系。

---

## 12. 什么是 A2A 协议？它和 MCP 的区别是什么？

### 核心回答

A2A 是 **Agent-to-Agent 协议，解决多个 AI Agent 之间如何通信和协作的问题**。MCP 是单个 Agent 向下连接工具和数据，A2A 是多个 Agent 横向协作和任务委托。
### 细节

A2A 解决的问题：
- 单个 Agent 上下文有限。
- 单个 Agent 专业能力有限。
- 复杂任务需要多个专业 Agent 协同。
- Agent 之间需要能力发现、任务委托、状态跟踪和结果返回。

#### A2A 核心机制

- **Agent Card：声明 Agent 能力、接口和可承接任务**。
	 - Agent 名称和描述；
	- 服务地址；
	- 支持的 Skill 列表；
	- 每个 Skill 的描述和输入示例；
	- 是否支持流式响应；
	- 是否支持异步回调
	Agent Card 里最关键的是 Skill 列表，每个 Skill 描述一类能力
	调度 Agent 用这些 Skill 描述来做任务路由决策

这套机制让整个多 Agent 系统变得可插拔：新加一个 Agent，发布它的 Agent Card，调度 Agent 就能自动发现和利用它，完全不需要改调度 Agent 的代码。

- **Task 状态机：支持异步任务、长任务、任务状态流转。
- 流式结果：长任务可以逐步返回进展。
- 能力发现：一个 Agent 可以发现另一个 Agent 能做什么。

### 面试提醒

MCP 和 A2A 是互补关系。在复杂多 Agent 系统里，Agent 可以通过 MCP 调工具，也可以通过 A2A 委托其他 Agent。

---

## 13. MCP 协议通常采用什么通信方式？

### 核心回答

MCP 的**消息格式统一使用 JSON-RPC 2.0，传输方式主要有 stdio 和 Streamable HTTP**。**stdio 适合本地工具，Streamable HTTP 适合远程共享服务**。
### 细节

stdio：

- Client 把 Server 作为本地子进程启动。
- 通过标准输入输出通信。
- 不走网络，不需要开端口。
- 延迟低，安全面小。
- 适合本地文件系统、Git、本地脚本等工具。

Streamable HTTP：

- Server 作为独立 HTTP 服务部署。
- 多个 Client 可以共享同一个 Server。
- 适合团队统一部署、远程 API、共享工具服务。
- **单端点处理请求和流式响应**。Streamable HTTP 的做法是把这两条通道合并成一个端点：Client 照样 POST 发请求，Server 根据情况决定返回「一个普通 JSON」还是「一条 SSE 流」，不需要 Client 提前开另一条连接。
### 面试提醒

要区分消息格式和传输方式：JSON-RPC 2.0 决定消息长什么样，stdio/HTTP 决定消息怎么传。

---

## 14. WebSocket 和 SSE 的区别及局限性是什么？

### 核心回答

**SSE 是服务端到客户端的单向推送，WebSocket 是全双工通信。LLM 文本流式输出大多数场景用 SSE 就够了，因为模型只需要持续向客户端推 token；真正需要双向实时交互时才需要 WebSocket**
### 细节

| 维度   | SSE                   | WebSocket            |
| ---- | --------------------- | -------------------- |
| 通信方向 | 服务端单向推送               | 双向全双工                |
| 协议基础 | HTTP 原生               | HTTP Upgrade 后建立独立连接 |
| 适合场景 | LLM 流式输出、通知、进度推送      | 双向实时协作、实时控制          |
| 优点   | 简单、轻量、易运维、自动重连        | 双向实时、灵活              |

LLM 文本对话中，用户发一次请求，模型持续返回 token，这正好符合 SSE 的单向推送模型。

**SSE 的三个局限性要记住**：
- HTTP/1.1 下同域名连接数上限（6 条）
- 只支持文本格式（传二进制要 Base64 编码膨胀 33%）
- 单向性导致的双通道架构复杂度

**WebSocket 的三个局限性**：
- 有状态导致横向扩展麻烦（需要 Redis 等共享存储做连接状态外移）
- 容易被企业代理和防火墙拦截（Upgrade 握手被当异常请求拒掉）
- 没有内置的请求-响应配对机制（需要自己维护请求 ID 映射）
### 面试提醒

不要说 WebSocket 更强所以一定更好。协议选型看通信模式：单向流式输出用 SSE，真正双向实时才上 WebSocket。

---

## 15. 为什么要用 WebRTC？它和 WebSocket 在 AI 对话流中的核心差异是什么？

### 核心回答

WebRTC 适合实时语音/音视频，因为它**基于 UDP，能用少量丢包换低延迟**。WebSocket 基于 **TCP，可靠但会因丢包重传导致延迟堆积**，不适合对延迟极敏感的实时语音。

### 细节

**WebSocket：

- 基于 TCP。
- 丢包会重传。
- 保证可靠和有序。
- 网络抖动时，后续数据会被阻塞。
- 适合文本消息、控制信令、普通实时通信。

**WebRTC**：

- 基于 UDP。
- 不强制等待重传。
- 内置丢包隐藏、抖动缓冲、自适应码率。
- 内置回声消除、噪声抑制、自动增益控制。
- 支持 NAT 穿透，使用 ICE/STUN/TURN。

语音场景的核心原则：

> 可以容忍少量丢包，不能容忍持续延迟。

WebRTC 建连时仍可能使用 WebSocket 做信令通道，二者不是完全替代关系。

### 面试提醒

一句话：WebSocket 适合可靠文本双向通信，WebRTC 适合低延迟音视频。实时语音产品选 WebRTC，是因为 TCP 的重传机制不适合语音。

---

## 16. 有没有用过大模型网关框架？网关层解决什么问题？

### 核心回答

LLM Gateway 是**架在应用和模型 API 之间的中间层**，统一管理多模型调用、API Key、限流配额、成本追踪、缓存、安全过滤和路由。它不是普通反向代理，而是为 LLM 场景设计的治理层。
### 细节

常见功能：

- **多模型统一接口**：业务代码只调用网关，底层可切换 OpenAI、Claude、Qwen、DeepSeek 等模型。
- API Key 集中管理：避免密钥散落在各个服务。
- **限流和配额**：按团队、应用、用户设置 token 预算。
- **成本追踪**：记录每个服务的 token 用量和费用。
- 模型路由：按任务类型、成本、延迟、失败率选择模型。
- **语义缓存**：语义相近的问题命中缓存，减少模型调用。
- **安全过滤**：Prompt 注入检测、敏感信息脱敏、输出审查。
- 失败兜底：主模型失败时自动切换备用模型。

常见框架：

- LiteLLM。
- One API。
- 自研统一模型网关。

### 面试提醒

不要把 LLM Gateway 说成普通负载均衡。普通 API 网关管流量，LLM 网关还要管 token、成本、模型路由、语义缓存和 Prompt 安全。

---

## 二、核心对比表

### Function Calling、MCP、Skill、A2A

| 概念 | 解决问题 | 层级 | 类比 |
| --- | --- | --- | --- |
| Function Calling | 模型怎么发起函数调用 | 底层调用协议 | 调用语言 |
| MCP | 工具怎么标准化接入和复用 | 工具生态协议 | 工具箱 |
| Skill | Agent 如何按流程完成任务 | 任务知识/流程模块 | 操作手册 |
| A2A | 多个 Agent 如何通信协作 | Agent 间通信协议 | Agent 微服务协议 |

### MCP 三层结构

| 层级 | 组成 | 说明 |
| --- | --- | --- |
| 角色层 | Host / Client / Server | Host 是宿主应用，Client 负责通信，Server 提供工具 |
| 能力层 | Tools / Resources / Prompts | Tools 有副作用，Resources 只读，Prompts 是模板 |
| 协议层 | JSON-RPC 2.0 + stdio / Streamable HTTP | JSON-RPC 定消息格式，传输层负责消息传递 |

### 通信协议选型

| 场景 | 推荐协议 |
| --- | --- |
| LLM 文本流式输出 | SSE |
| 双向实时文本/控制消息 | WebSocket |
| 实时语音、音视频通话 | WebRTC |
| MCP 本地工具 | stdio |
| MCP 远程共享服务 | Streamable HTTP |

## 相关笔记

- **系统位置**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Agent_16|Agent 核心问题]] — Tools 与协议是 Agent 执行闭环的行动接口。
- **对比**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/A2A|A2A]] — 工具协议和 A2A 分别解决不同层级的互操作。
- **对比**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Skill_10|Skill_10]] — 工具协议解释了 Skill 最终调用动作的运行边界。
- **项目应用**：[[50-Projects-项目/01-MiniCode/Tool、MCP与Skill|MiniCode 工具系统]] — MiniCode 的工具注册、MCP 接入和权限控制落地了协议知识。
- **运行机制**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/主循环Query Loop|Claude Code Query Loop]] — Query Loop 负责识别 tool_calls、并行执行并写回结果。
- **应用案例**：[[30-Agent-Engineering-Agent工程/04-References-项目参考/ByteRover|ByteRover]] — 用 Skill 传递记忆工作流，并通过 MCP 暴露查询与策展工具。
