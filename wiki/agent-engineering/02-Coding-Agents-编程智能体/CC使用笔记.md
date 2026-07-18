---
date: 2026-03-20
tags:
  - ClaudeCode
published: true
categories:
  - vibe coding
title: Claude Code 使用心得
---

# Claude Code 使用心得：常用命令、技巧与速查版

关联阅读：可结合 [[10-LLM-大语言模型/04-Engineering-工程实践/Python环境配置|Python环境配置]] 理解；编码 Agent 使用流程需要稳定的 Python 和命令行环境。


如果你刚开始接触 Claude Code，最容易卡住的往往不是不会写 Prompt，而是不知道哪些入口最值得先记、长会话该怎么收束、什么时候该先规划再执行。

这篇笔记把我自己高频使用的命令、快捷键、插件和学习方向，整理成一篇博客：
前半部分讲思路，后半部分给速查。

> [!summary]
> 如果你只想先记住最常用的 6 个入口：
> - `/compact`：压缩上下文，适合长对话
> - `/clear`：清空当前会话
> - `/resume`：续接历史任务
> - `/plan`：先让 Claude 出方案
> - `/agent` / `/agents`：调用子智能体并行处理
> - `/doctor`：排查环境和依赖问题

## 为什么值得把 Claude Code 用顺手

和普通聊天式 AI 相比，Claude Code 更像是终端里的 AI 搭子：它不是只负责回答问题，而是能直接参与到“读代码、改代码、跑命令、排查问题”的完整流程里。

真正拉开效率差距的，通常不是模型本身，而是你会不会用好下面这几类能力：

- 会话管理：避免上下文越来越乱
- 计划模式：先想清楚，再改代码
- 子智能体：把任务拆开并行处理
- Skills / 插件：把高频动作工具化

## 快捷键速查

| 快捷键           | 用途                    | 什么时候最有用              |
| ------------- | --------------------- | -------------------- |
| `Ctrl + B`    | 将任务转到后台运行             | 命令较慢，不想阻塞当前会话时       |
| `Ctrl + J`    | 换行但不提交                | 写长 Prompt、分段列需求时     |
| `Esc Esc`     | 回滚输入，连按两次删除整行对话框内容    | 输入写乱时快速清空            |
| `Shift + Tab` | 在普通模式、自动同意模式、计划模式之间切换 | 需要在“先规划”和“直接执行”之间切换时 |

## 常用命令，我会这样记

### 1. 会话与上下文管理

这组命令解决的是：对话聊久了，Claude 开始发散、回复越来越长，甚至偏离重点。

- `/compact`：压缩对话历史，保留重点，适合长任务中途整理上下文
- `/clear`：直接清空当前会话，适合彻底切换主题
- `/rewind`：回到之前的检查点，适合“这一步走歪了，想撤回”
- `/resume`：恢复历史会话，适合接着昨天没做完的任务继续

> [!tip]
> 如果你感觉 Claude 开始“话变多、重点变少、速度变慢”，先试一次 `/compact`，通常比你手动重述背景更省事。

### 2. 项目与环境配置

这组命令更像是把工作台先搭好。

- `/init`：在当前目录初始化项目，并生成 `CLAUDE.md`，适合沉淀项目级规范
- `/config`：打开配置面板，快速调整全局设置
- `/mcp`：管理 MCP 服务，让 Claude 连接更多外部工具和上下文
- `/allowed-tools`：管理工具权限，减少重复确认

### 3. 任务规划与并行处理

这是 Claude Code 最容易被低估的一块，也是最能提升效率的一块。

- `/plan`：先输出完整思路和改动步骤，再决定是否执行
- `/agent` / `/agents`：把子任务分给 subagent 并行处理
- `/tasks`：查看后台任务状态，管理长时间运行的工作

> [!note]
> 我自己的习惯是：需求不清、改动面大、涉及多个文件时，先 `/plan`；任务能拆分时，再交给 `/agent` 或 `/agents` 去并行跑。

### 4. 状态与诊断

这组命令适合处理“我怀疑不是我写错了，而是环境有问题”的场景。

- `/cost` 或 `/cos`：查看当前 Session 的成本与耗时
- `/usage`：查看 API 总体额度使用情况和限流状态
- `/doctor`：检查当前终端环境、依赖和安装状态

## 插件与 Skills，哪些值得先上手

### Skills

我当前重点关注的是这几类技能：

- `openspec`：适合 OpenAPI 规范和接口文档类工作流
- `helloagent`：适合理解 Agent / Sub-agent 的基本思路
- `skill-seeker`：适合搜索、生成和整理技能文档
- `autogen`：偏多智能体方向

其中 `helloagent` 可以参考：

- [GitHub - jjyaoao/HelloAgents](https://github.com/jjyaoao/HelloAgents)
- [GitHub - hellowind777/helloagents](https://github.com/hellowind777/helloagents/tree/main)

### 我在用的插件记录

#### `context7`

- 跟 Claude Code 对话时，可以在 Prompt 结尾补一句：`use context7`
- 适合需要补充上下文、文档或知识参考的场景

#### `document skills`

- 可以隐式触发，也可以显式点名触发

例如：

- 隐式触发：帮我把 `report.pdf` 里的表格提取成 CSV
- 显式触发：请使用 PDF skill 帮我把 `report.pdf` 里的表格提取成 CSV
- 显式触发：用 docx 技能帮我创建一个标准的商业合同模板

### Claude Code 配置

我目前有记录的配置偏好：

- `statusline`
  - `nerd_font` 字体
  - `cometix` 主题

## 一套够用的上手流程

如果你不想一次记太多，可以先记住下面这条工作流：

1. 用 `/init` 给项目建立规则
2. 长对话中用 `/compact` 收束上下文
3. 遇到复杂需求先 `/plan`
4. 能拆开的任务交给 `/agent` / `/agents`
5. 环境不对劲时先跑 `/doctor`

> [!success]
> 对大多数日常开发场景来说，`/init → /plan → /agent(s) → /compact` 这条链路已经足够高频、足够实用。

## 延伸学习方向

如果你准备继续往下深挖，我建议优先看下面这些方向。

### Skills 方向

- `openspec`
  - [博客教学](https://www.aivi.fyi//llms/introduce-OpenSpec)
  - [YouTube 教学](https://www.youtube.com/watch?v=ANjiJQQIBo0)
  - [L 站 openspec 实战](https://linux.do/t/topic/1445627)
  - [OPSX 任务拆分](https://linux.do/t/topic/1701121)
- `skill-seeker`
  - [博客教学](https://www.aivi.fyi/aiagents/introduce-Skill-Seeker)
  - [YouTube 教学](https://www.youtube.com/watch?v=K6-sR__mA3s&t=9s)
- `Trellis`
  - [L 站讨论](https://linux.do/t/topic/1539636)
  - [GitHub - mindfold-ai/Trellis](https://github.com/mindfold-ai/Trellis)
### 多智能体方向

- [GitHub - datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents)
- `autogen`

## 速查版

如果你只是临时回来翻一下命令，直接看这一节就够了。

| 场景 | 首选命令 |
|------|----------|
| 对话太长，想保留重点继续聊 | `/compact` |
| 想彻底切到新话题 | `/clear` |
| 想恢复之前没做完的任务 | `/resume` |
| 动手前先看实现方案 | `/plan` |
| 想把任务并行拆开处理 | `/agent` / `/agents` |
| 查看后台任务列表 | `/tasks` |
| 初始化当前项目规范 | `/init` |
| 管理工具权限 | `/allowed-tools` |
| 管理 MCP 服务 | `/mcp` |
| 管理插件 / Skills | `/plugins` |
| 查看本次会话成本 | `/cost` |
| 查看额度与限流 | `/usage` |
| 排查环境问题 | `/doctor` |

> [!abstract]
> 一句话总结：
> Claude Code 真正好用的地方，不只是“能回答”，而是它能在正确的时机帮你**压缩上下文、先做规划、并行处理任务、接入更多工具**。

## 相关笔记

- **应用**：[[10-LLM-大语言模型/04-Engineering-工程实践/Python环境配置|Python环境配置]] — 编码 Agent 使用流程需要稳定的 Python 和命令行环境。
- **基础使用**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/AI 辅助编程生态工具(Claude Code 篇)|Claude Code 生态工具]] — 使用笔记为生态工具提供实际工作流上下文。
- **对比**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/Codex_Guide|Codex_Guide]] — 两份指南可比较 Claude Code 与 Codex 的命令、上下文和扩展方式。
- **延伸**：[[30-Agent-Engineering-Agent工程/03-Harness-and-Workflows-运行框架与工作流/从 OpenSpec、Superpowers 到 TRELLIS 的工程化演进|OpenSpec、Superpowers 与 TRELLIS]] — 日常命令和插件使用可进一步升级为规范驱动工作流。
- **参考**：[[30-Agent-Engineering-Agent工程/04-References-项目参考/Claude Code 认证专家操作指南|Claude Code 认证指南]] — 认证指南提供更严格的 Claude Code 操作与 Trace 要求。
