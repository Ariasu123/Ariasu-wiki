
**Loop Engineering** 是把“人反复给 Agent 下指令”的过程，设计成一个能够自动运行的循环系统。

人的职责从亲自执行每一步，转变为定义：

- 什么时候触发
- 每轮执行什么任务
- 如何验证结果
- 什么时候停止
- 失败后如何重试或转人工

### 核心组成

关联阅读：可结合 [Claude Code Query Loop](../02-Coding-Agents-%E7%BC%96%E7%A8%8B%E6%99%BA%E8%83%BD%E4%BD%93/Claude-Code-Internals-%E6%BA%90%E7%A0%81%E5%8E%9F%E7%90%86/%E4%B8%BB%E5%BE%AA%E7%8E%AFQuery%20Loop.md) 理解；Claude Code Query Loop 是循环工程原则的具体实现。


1. **自动化触发**：通过**定时任务或事件启动循环。
2. **Worktree**：为并行 Agent 提供独立工作目录，避免代码冲突。
3. **Skill**：保存**项目规范和操作经验**，减少重复解释。
4. **Connector**：连接 GitHub、工单、监控等外部系统。
5. **Sub-agent**：将执行和检查分开，避免 Agent 自我评价过于宽松。
6. **持久化记忆**：使用状态文件或任务看板记录进度，让下一轮可以继续。（**把记忆放在磁盘上，而不是上下文里**。一个 markdown 文件、一个任务看板，什么都行，只要它活在单次对话之外，记录着「做完了什么、下一步是什么」。
### 典型流程

```
发现任务 → 分配任务 → 执行修改 → 独立检查
→ 提交结果 → 记录状态 → 继续下一轮
```

无法解决的问题进入待办队列，由人工处理。一个好的 Loop 应尽量只让人参与**循环设计、异常处理和最终验收**。
### 风险

- Agent 判断“完成”只是声明，最终验证仍需可靠测试或**人工确认。
- 自动生成代码过快，容易产生“理解债”。
- 长时间循环和多个 Sub-agent 会增加 Token 成本。
- 过度依赖自动结果，可能导致人放弃独立判断。


> Loop engineering 不是某个单独命令，而是一种把 Agent 任务设计成持续发现、执行、验证和推进任务的自动循环的工程方法。Claude Code 的 `/loop` 和 Codex 的 `/goal` 都是这种思想的具体实现。区别在于，`/loop` 是 cadence-driven，按时间或节奏重复执行；`/goal` 是 condition-driven，围绕一个可验证目标持续推进，直到验收条件成立或遇到阻塞。

更短版：

> `/loop` 解决“定期做什么”，`/goal` 解决“做到什么程度才停”，Loop engineering 解决“如何设计这个自动执行闭环”。

来源：[小林面试笔记](https://xiaolinnote.com/agent/engineering/loop-engineering.html)

## 相关笔记

- **源码实例**：[Claude Code Query Loop](../02-Coding-Agents-%E7%BC%96%E7%A8%8B%E6%99%BA%E8%83%BD%E4%BD%93/Claude-Code-Internals-%E6%BA%90%E7%A0%81%E5%8E%9F%E7%90%86/%E4%B8%BB%E5%BE%AA%E7%8E%AFQuery%20Loop.md) — Claude Code Query Loop 是循环工程原则的具体实现。
- **组成**：[Harness Engineering](Harness%20Engineering.md) — 可靠循环需要 Harness 的验证、权限和反馈环境。
- **案例**：[OpenClaw架构与常见问题](OpenClaw%E6%9E%B6%E6%9E%84%E4%B8%8E%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98.md) — OpenClaw 的持续执行、工具循环和故障恢复体现循环工程问题。
