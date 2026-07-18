## 一. 什么是 A2A 协议？

关联阅读：可结合 [[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Agent_16|Agent 核心问题]] 理解；理解单 Agent 循环后才能判断 Agent 间通信边界。


### 1. 核心回答

A2A 是 **Agent-to-Agent 协议**，主要解决多个 AI Agent 之间如何 **发现能力、通信协作、任务委托、状态跟踪和结果返回** 的问题。
- Agent 能力发现
- Agent 任务委托
- Task 生命周期状态
- 长任务异步执行
- 任务结果 artifacts
- 多 Agent 协作

```text
MCP 解决的是：一个 Agent 如何使用外部工具、数据源、API
A2A 解决的是：多个 Agent 如何互相发现、委托任务、协同完成复杂工作
```

---

### 2. 为什么需要 A2A？

在复杂任务中，单个 Agent 往往不够用，主要有几个问题：

- 单个 Agent 的上下文窗口有限，无法长期保存和处理所有任务信息。
- 单个 Agent 的专业能力有限
- 复杂任务通常需要多个专业 Agent 协同完成。
- 多 Agent 之间需要统一的能力发现、任务委托、状态跟踪和结果交付机制。
- 长任务不能一直同步等待，需要支持异步执行、轮询和回调。

所以 A2A 的核心价值是：

> 让多个专业 Agent 可以像团队成员一样协作，每个 Agent 只暴露自己的能力和结果，不暴露内部实现。

---

### 3. A2A 的核心机制

A2A 主要可以从三个方面理解：

```text
Agent Card：发现 Agent 能力
Task：任务委托和生命周期管理
Artifacts：任务完成后的结果产物
```

---

## 二. Agent Card：Agent 的能力名片

Agent Card 可以理解为每个 Agent 对外暴露的一张“能力名片”。

它告诉调度 Agent：

```text
我是谁
我能做什么
我支持哪些 Skill
我怎么被调用
我返回什么格式
我是否支持异步
我是否支持流式输出
```

常见字段包括：

- Agent 名称和描述。
- 服务地址。
- 支持的 Skill 列表。
- 每个 Skill 的描述和输入示例。
- 返回格式，例如 JSON、Markdown、文件等。
- 是否支持流式响应。
- 是否支持异步回调。
- 认证方式和版本信息。

Agent Card 里最关键的是 **Skill 列表**。  
**调度 Agent 会根据用户任务和各个 Agent 的 Skill 描述进行匹配，决定把任务交给哪个 Agent**。

---

### Agent Card 的发现方式

通常每个 Agent 会通过一个固定路径暴露自己的 Agent Card，例如：

```text
/.well-known/agent-card.json
```

这套机制让多 Agent 系统变得可插拔：

> 新加一个 Agent，只需要发布它的 Agent Card，调度 Agent 就能发现并使用它，不需要改调度 Agent 的核心代码。

---

## 三. Task：A2A 任务协作的基本单位

在 A2A 中，任务协作的基本单位是 **Task**。

调度 Agent 想让另一个 Agent 做事，本质上是创建一个 Task，然后委托给被调 Agent。

流程如下：

```text
调度 Agent
  ↓
创建 Task
  ↓
委托给被调 Agent
  ↓
被调 Agent 执行 Task
  ↓
产出 artifacts
  ↓
调度 Agent 获取结果
  ↓
返回给用户
```

这里的 Task 可以理解成：

> 一个可跟踪、可查询、可异步执行的任务对象。

---

### Task 生命周期状态机

Task 有完整的生命周期状态管理，常见状态包括：

```text
submitted → working → completed / failed
```

#### submitted：已提交

表示任务已经被创建并提交，等待处理。

```text
含义：任务已创建
触发：调度 Agent 提交任务
```

#### working：执行中

表示接收方 Agent 已经接收任务，并开始执行。

```text
含义：任务执行中
触发：被调 Agent 开始处理任务
```

#### completed：成功

表示任务执行成功，并产生结果。

```text
含义：任务完成
触发：执行成功
结果：返回 artifacts
```

#### failed：失败

表示任务执行失败。

```text
含义：任务失败
触发：执行失败
结果：返回错误信息或失败状态
```

---

### 为什么需要 Task 状态机？

因为 A2A 主要是为 **长时间任务** 设计的。

这些任务可能要执行几十秒、几分钟甚至更久，不适合让调度 Agent 一直同步等待。

所以 A2A 用 Task 状态机来支持异步协作：

> 调度 Agent 提交任务后，可以先去处理其他事情，之后再通过状态查询或回调机制获取结果。

---

## 四. Artifacts：任务产物

Artifacts 是 Task 执行完成后的结果产物，返回摘要。

它可以是：

- 文本
- Markdown 报告。
- JSON 数据。
- 文件。
- 图片。
- 表格。
- 链接。
- 结构化分析结果。

调度 Agent 不需要知道被调 Agent 内部用了什么工具、调用了几次 LLM、怎么完成任务，只需要拿到最终 artifacts。

这体现了 A2A 的解耦思想：

> 被调 Agent 内部实现对外不可见，调度 Agent 只关心任务状态和最终结果。

---

## 五. A2A 如何获取任务结果？

A2A 支持两种常见方式：

```text
Polling：主动轮询
Push Notification：完成后回调
```

---

### 5.1 主动轮询 Polling

**调度 Agent 定期查询 Task 状态。

流程：

```text
调度 Agent
  ↓ 定期查询
被调 Agent
  ↓ 返回 Task 状态
submitted / working / completed / failed
```

例如：

```text
每隔 5 秒查询一次任务是否完成
```

适合简单场景，优点是实现简单。

缺点是：

- 频繁查询会浪费资源
- 状态更新不是实时的
- 长任务较多时会增加系统压力
---

### 5.2 Push Notification 回调

被调 Agent 完成任务后，主动通知调度 Agent。
流程：

```text
被调 Agent 完成任务
  ↓
主动回调调度 Agent
  ↓
通知任务完成
  ↓
返回 artifacts
```

适合长任务或任务量较大的场景。

优点是：

- 不需要频繁轮询。
- 任务完成后可以及时通知。
- 更适合异步任务系统。

---

## 六. A2A 能解决多 Agent 冲突吗？

A2A 可以 **缓解多 Agent 协作冲突**，但不能自动解决所有冲突。

### A2A 能缓解的冲突

#### 1）能力选择冲突

多个 Agent 都声称自己能做某个任务时，Agent Card 可以帮助调度 Agent 判断谁更合适。
调度 Agent 可以根据任务需求和 Skills 匹配度选择 Agent。
#### 2）任务状态冲突

通过 Task 状态机，调度 Agent 可以清楚知道任务当前状态：

```text
submitted
working
completed
failed
```

避免出现“不知道任务到底有没有开始、有没有完成、是否失败”的问题。
#### 3）同步等待冲突

通过异步 Task、Polling 和 Push Notification，调度 Agent 不需要一直阻塞等待长任务完成。
#### 4）结果交付格式冲突

A2A 用 artifacts 统一表示任务产物，减少不同 Agent 返回结果格式不统一的问题。

---

### A2A 不能自动解决的冲突

#### 1）结论冲突

A2A 只负责让两个 Agent 返回结果，不会自动判断谁对谁错。

这种需要：
- 仲裁 Agent。
- 规则优先级。
- 置信度评分。
- 证据链检查。
- 人工确认。
---

#### 2）共享资源写冲突

例如两个 Agent 同时修改同一个文件或数据库记录。

A2A 本身不负责数据库事务或文件锁，需要额外设计：

- 锁机制。
- 版本号。
- 乐观并发控制。
- 权限隔离。
- 回滚机制。
---

#### 3）权限冲突

某个 Agent 想访问它不该访问的数据时，A2A 不会自动完成权限治理。

需要系统侧实现：

- 访问控制。
- token scope。
- 权限校验。
- 审计日志
---

#### 4）任务依赖冲突

例如：
```text
报告生成必须等市场分析和财务分析都完成
财务分析必须等数据抽取完成
```

这类依赖关系需要 Orchestrator 设计工作流，例如：

- DAG 编排。
- 状态机。
- 任务队列。
- 依赖检查。
- 失败重试
---

### 总结

A2A 解决的是：

```text
怎么协作
怎么委托
怎么跟踪状态
怎么返回结果
```

但不自动解决：

```text
谁的结论更可信
谁先执行
谁有权限
多个 Agent 同时写资源怎么办
```

这些需要上层 Orchestrator 和治理策略解决。

## 相关笔记

- **前置**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Agent_16|Agent 核心问题]] — 理解单 Agent 循环后才能判断 Agent 间通信边界。
- **协议对比**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Tools_16（MCP、Function Calling与通信协议）|Tools、MCP 与 Function Calling]] — A2A 面向 Agent 间通信，MCP／工具协议面向 Agent 与能力连接。
- **实现参考**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/多Agent机制|Claude Code 多 Agent]] — Claude Code 的多 Agent 机制展示任务委派与结果回传。
- **项目实例**：[[50-Projects-项目/01-MiniCode/多智能体|MiniCode 多智能体]] — MiniCode 多智能体展示 Agent 协作的本地实现。
