
`queryLoop` 是 Claude Code Agent 的主循环，负责不断执行：

```
调用模型 → 解析响应 → 执行工具 → 回传结果 → 再次调用模型
```

直到模型不再请求工具，或者触发异常退出条件。

### 一、四层调用链

关联阅读：可结合 [[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Tools_16（MCP、Function Calling与通信协议）|Tools、MCP 与 Function Calling]] 理解；主循环以 Tool Calling 协议驱动行动与观察。


```
ask
  ↓
QueryEngine.submitMessage
  ↓
query
  ↓
queryLoop
```

- **ask**：SDK 的便捷调用入口。
- **QueryEngine**：管理消息历史、文件缓存、权限记录等会话状态。
- **query**：使用**异步生成器**持续向外传递事件。
- **queryLoop**：真正执行 Agent 循环。

各层通过 `yield*` 传递事件，因此模型输出、工具状态和错误信息都能实时显示。

### 二、每轮执行流程

每轮循环主要完成五件事：

1. **准备消息**：整理完整对话历史。
2. **流式调用模型**：实时输出文本，并收集 `tool_use`。
3. **判断是否继续**：检查模型是否请求工具。
4. **执行工具**：运行本轮产生的全部工具调用。
5. **回填结果**：把 `tool_result` 加入消息历史，开始下一轮。

### 三、循环终止判断

核心判断非常简单：

- 存在 `tool_use`：执行工具并继续循环。
- 不存在 `tool_use`：认为模型已完成回答，结束循环。

除此之外，还有多种结构化退出原因：

- `completed`：正常完成
- `max_turns`：超过最大轮数
- `aborted_streaming`：流式输出被中断
- `aborted_tools`：工具执行被中断
- `prompt_too_long`：上下文压缩后仍然过长
- `max_output_tokens_recovery`：多次续写仍失败
- `stop_hook_prevented`：被 Hook 拦截
- `image_error`：图片格式错误

每一种**异常退出，都不是一个简单的 `throw new Error`，而是需要：

- 第一，识别这种错误状态（每种错误的特征不一样）；
- 第二，做必要的清理（比如取消正在跑的工具、解锁资源）；
- 第三，**返回结构化的 `reason`**，让外层调用者知道是什么原因结束的，方便上报、重试或者给用户友好提示。
### 四、流式并行执行

Claude Code **使用 `StreamingToolExecutor`：

- 模型一旦生成完整的 `tool_use`，工具立即在后台启动。
- 不需要等待模型整段响应结束。
- 模型生成与工具执行时间可以重叠。

并发规则：

- `Read`、`Grep`、`Glob` 等**只读工具可以并行。
- `Edit`、`Write`、`Bash` 等**修改状态的工具必须串行。
- 工具未声明属性时，默认**按写入工具处理**，保证安全。

### 五、显式状态管理

**跨轮信息**统一保存**在 `State` 对象**中，例如：

- 已执行轮数
- 是否进行过上下文压缩
- 输出截断恢复次数
- 权限和工具执行状态

通过**计数器和防重复标志，避免无限压缩、无限续写和死循环。

### 六、错误恢复机制

**上下文过长**

先正常调用 API；只有返回 `prompt_too_long` 时才压缩并重试。同一轮最多压缩一次。

**工具执行中断**

Anthropic API 要求**每个 `tool_use` 都对应一个 `tool_result`**。如果工具没有执行完成，系统会生成一个包含错误信息的结果进行配对，保证下一轮请求仍然合法。

**模型输出被截断**（模型每一次返回都有个最大 token 数限制。Claude Code 常规默认是 32k）

1. 首次截断时，提高输出 Token 上限并静默重试。
2. 仍然截断时，提示模型从断点直接续写。
3. 最多续写三次，随后结构化退出。

### 核心设计思想

1. **边执行边输出**：通过异步生成器降低用户等待感。
2. **显式管理状态**：让循环过程可调试、可恢复。
3. **引擎与工具解耦**：新增工具不需要修改主循环。
4. **优先恢复错误**：尽量让异常在后台解决，而不是直接终止会话。

一句话概括：Query Loop 看起来只是一个 `while` 循环，但真正的工程价值在于流式传递、工具并发、状态管理和异常恢复。

来源：[Claude Code 主循环 Query 详解](https://xiaolinnote.com/claudecode/source/cc_query_loop.html)

## 相关笔记

- **接口基础**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Tools_16（MCP、Function Calling与通信协议）|Tools、MCP 与 Function Calling]] — 主循环以 Tool Calling 协议驱动行动与观察。
- **总览**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/Claude Code架构设计|Claude Code 架构]] — 架构设计说明主循环与模型、工具和状态的关系。
- **工程原则**：[[30-Agent-Engineering-Agent工程/03-Harness-and-Workflows-运行框架与工作流/Loop Engineering|Loop Engineering]] — Loop Engineering 关注主循环中的反馈、验证和停止条件。
- **项目应用**：[[50-Projects-项目/01-MiniCode/MiniCode_architecture|MiniCode 架构]] — MiniCode 的模型—工具—观察循环对应 Claude Code 主循环。
