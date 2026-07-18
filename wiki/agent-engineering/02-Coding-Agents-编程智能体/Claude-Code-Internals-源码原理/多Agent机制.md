
Claude Code 的多 Agent 系统包含三套机制：

1. **常规 Subagent**：主 Agent 临时派遣子 Agent。
2. **Fork Subagent**：复制父 Agent 上下文的轻量分身。
3. **Coordinator 模式**：协调者统一调度多个 Worker。

---

## 1. 常规 Subagent

关联阅读：可结合 [[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/A2A|A2A]] 理解；多 Agent 实现可用 A2A 的角色与消息边界进行分析。


主 Agent 通过 `Agent` 工具创建独立的子 Agent。每个 Subagent 都有：

- 独立上下文
- 独立工具池
- 独立 ID 和生命周期
- 独立执行循环

适合代码探索、资料调研、规划和专项分析，可以避免大量中间内容污染主 Agent 的上下文。

### 工具隔离

Claude Code 根据 Agent 类型过滤工具：

- 禁止 Subagent 继续创建子 Agent，避免递归失控。
- 禁止直接询问用户，保留主 Agent 的对话权。
- 禁止修改主任务列表和切换 Plan Mode。
- 后台 Agent 采用更严格的工具白名单。

### 上下文隔离

上下文不是完全复制或完全共享，而是按字段处理：

- 文件读取缓存：复制，避免污染父 Agent。
- 全局 UI 状态：禁止子 Agent 修改。
- 后台任务注册：保持共享，便于回收进程。
- Agent ID：独立生成。
- 嵌套深度：在父 Agent 基础上加一，用于防止层级失控。

---

## 2. 父子通信

### 默认模式

默认 Subagent 更像一次重量级工具调用：

```
父 Agent 派发任务
→ 子 Agent 独立执行
→ 返回 tool_result
→ 父 Agent 继续处理
```

父 Agent不能在执行过程中追加指令。

如果任务运行超过约两分钟，可以自动转入后台。完成后，系统把结果包装为 XML 通知，并作为一条消息注入父 Agent 的对话队列。

### Agent Teams 模式

开启团队模式后，父子之间支持双向通信：

- 父 → 子：通过 `SendMessage` 写入子 Agent 的消息队列。
- 子 → 父：通过任务通知返回结果。
- 已完成的子 Agent 可以重新唤醒并继续工作。

需要注意：完整双向通信是团队模式提供的能力，默认模式主要是子 Agent 完成后通知父 Agent。

---

## 3. Fork Subagent

Fork Subagent 直接继承父 Agent 的：

- System Prompt
- 用户和系统上下文
- 工具定义及顺序
- 对话历史前缀

这些内容保持字节级一致，从而命中 Prompt Cache，降低输入成本和首 Token 延迟。

适合：

- 需要完整继承父上下文
- 临时尝试另一条方案
- 生成总结或 PR 描述
- 不希望结果污染父 Agent 主循环

如果子 Agent需要专门角色和定制工具，应使用常规 Subagent。

---

## 4. Coordinator 模式

Coordinator 模式用于大型并行任务。主 Agent 转变为纯协调者，只负责：

1. 拆分任务并派遣 Worker。
2. 接收并理解结果。
3. 编写明确的实施规格。
4. 汇总最终答案。

典型流程：

```
并行调研
→ Coordinator 汇总和决策
→ Worker 执行实现
→ 新 Worker 独立验证
→ Coordinator 输出结果
```

Worker 采用扁平结构，不能继续创建团队或调度其他 Worker，防止形成无限递归树。

### 继续还是新建 Worker

- 新任务与已有上下文高度相关：继续使用原 Worker。
- 任务无关或 Worker 已经跑偏：创建新 Worker。
- 代码验证和审查：使用新的 Worker，避免让实现者自我检查。

---

## 核心设计原则

- 上下文隔离需要按字段处理，不能简单地全部共享或全部复制。
- 不同 Agent 应具有不同的工具权限。
- 默认通信与团队双向通信要明确区分。
- Fork 通过缓存复用降低多 Agent 成本。
- 多个独立任务应尽量并行。
- Coordinator 必须理解和合成结果，不能只是转发。
- 系统保持“一个协调者 + 多个 Worker”的扁平结构。

来源：[Claude Code 多 Agent 详解](https://xiaolinnote.com/claudecode/source/cc_multi_agent.html)

## 相关笔记

- **协议视角**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/A2A|A2A]] — 多 Agent 实现可用 A2A 的角色与消息边界进行分析。
- **总览**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/Claude Code架构设计|Claude Code 架构]] — 架构笔记为多 Agent 的委派机制提供系统边界。
- **对照实现**：[[50-Projects-项目/01-MiniCode/多智能体|MiniCode 多智能体]] — 两者都涉及任务拆分、委派、隔离上下文和结果回传。
