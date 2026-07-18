
### 1. 四层架构

关联阅读：可结合 [[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/主循环Query Loop|Claude Code Query Loop]] 理解；Query Loop 是 Claude Code 总体架构的运行主干。


- **引擎层**：组织上下文、调用模型、分发工具、控制循环结束。
- **工具层**：提供文件读写、Shell、搜索、子 Agent 等能力，并标注是否只读、危险、可并行。
- **服务层**：负责模型 API、MCP 和上下文压缩等公共能力。
- **安全治理层**：负责权限确认、Hooks、命令分析和危险操作拦截。
![](../../_assets/Coding-Agents-%E7%BC%96%E7%A8%8B%E6%99%BA%E8%83%BD%E4%BD%93/Pasted%20image%2020260621164614.png)
### 2. Tool-Use Loop

Claude Code 没有采用传统 ReAct，而是使用更简单的工具循环：

```
调用模型 → 返回 tool_use → 执行工具 → 回传结果 → 再次调用模型
                         ↓
                    返回 end_turn → 结束
```

推理由模型内部完成，应用层只负责模型调用和工具执行，减少了解析复杂度与额外 Token。

### 3. Plan Mode

复杂任务可以先规划、后执行：

1. 进入 Plan Mode。
2. 权限切换为只读，只允许探索代码。
3. 将方案写入计划文件。
4. 用户审批后恢复写入权限并执行。

它本质上仍是通过工具切换能力，不需要修改 Agent 主循环。

### 4. System Prompt

System Prompt 由多个模块动态组装，包含：

- 角色与安全边界
- 行为准则
- 操作安全
- 工具使用规则
- Git 操作规范
- 输出规则
- 当前系统与项目环境注入
#### 动态分割与三级缓存

System Prompt 被分为两部分：

```
固定部分：
角色、安全规则、行为准则、Git 规范、输出风格
---------------- 动态边界 ----------------
动态部分：
环境信息、CLAUDE.md、记忆、MCP 配置
```

固定前缀可以**复用 Prompt Cache**，动态内容则按用户和项目实时生成。文章将其概括为三级缓存：

1. **全局缓存**：所有用户共享的固定内容。
2. **组织缓存**：同一组织内部共享。
3. **会话缓存**：当前会话内复用。

**核心思想：通过动态组装、明确行为边界和分层缓存，用尽可能低的 Token 成本，让模型稳定、安全地执行软件工程任务。**
### 5. 记忆系统

记忆分为四类：

- `User`：用户背景
- `Feedback`：用户偏好和行为反馈
- `Project`：项目动态和决策
- `Reference`：外部系统入口

每条记忆单独存为 Markdown 文件，`MEMORY.md` 只保存索引。系统使用较小模型筛选相关记忆，再按需加载详情，同时提醒模型验证可能已经过时的信息。

### 6. 上下文压缩

Claude Code 采用五级渐进压缩：

1. 大型工具结果存入磁盘，只保留预览。
2. 删除过早且无用的消息。
3. 裁剪可以重新获取的旧工具输出。
4. 接近窗口上限时生成压缩视图。
5. 最后才对完整对话生成结构化摘要。

压缩完成后，还会恢复最近读取的文件、活跃 Skill 和执行计划，避免 Agent 因压缩而失去工作状态。

## 相关笔记

- **组成**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/主循环Query Loop|Claude Code Query Loop]] — Query Loop 是 Claude Code 总体架构的运行主干。
- **组成**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/上下文管理与五层压缩|Claude Code 上下文压缩]] — 上下文压缩是架构中的预算与会话治理模块。
- **组成**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/代码检索总结|Claude Code 代码检索]] — 代码检索是编码 Agent 获取仓库证据的关键能力。
- **组成**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/多Agent机制|Claude Code 多 Agent]] — 多 Agent 模块扩展了单循环的任务分解与并行执行。
- **组成**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Claude-Code-Internals-源码原理/长期记忆机制|Claude Code 长期记忆]] — 长期记忆负责把稳定信息带入后续会话。
- **迁移参考**：[[50-Projects-项目/01-MiniCode/MiniCode_architecture|MiniCode 架构]] — MiniCode 借鉴编码 Agent 的模型调用、工具循环与状态治理。
