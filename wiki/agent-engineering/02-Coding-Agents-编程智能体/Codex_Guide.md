# Codex 进阶使用方法速查

关联阅读：可结合 [[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/AI 辅助编程生态工具(Claude Code 篇)|Claude Code 生态工具]] 理解；Codex 指南提供另一种编码 Agent 使用基线。


## 1. 核心功能

| 功能 | 作用 | 适合场景 |
|---|---|---|
| Annotate | 在浏览器页面上点选或框选 UI，并留下修改意见 | 前端样式、页面错位、交互问题 |
| Fork | 从当前会话（可任选之前的某个对话位置）复制一个新分支 | 多方案尝试、重构试错、风险改动 |
| Archive | 归档当前会话，保留记录但清理侧边栏 | 项目完成、阶段结束、历史暂存 |

## 2. 常用 Slash Command

| 命令             | 作用                    |
| -------------- | --------------------- |
| `/plan`        | 进入规划模式                |
| `/fork`        | 创建会话分支                |
| `/archive`     | 归档当前会话                |
| `/diff`        | 查看当前代码改动              |
| `/review`      | 审查当前改动                |
| `/permissions` | 调整权限                  |
| `/skills`      | 管理或调用 Skills          |
| `/plugins`     | 管理 Plugins            |
| `/mcp`         | 查看 MCP 连接             |
| **`/compact`** | 压缩上下文                 |
| `/status`      | 查看状态                  |
| `/usage`       | 查看使用情况                |
| `/model`       | 切换模型                  |
| `/new`         | 新建会话                  |
| `/resume`      | 恢复历史会话                |
| `/delete`      | 删除当前会话，谨慎使用           |
| **`/side`**    | 开启临时side conversation |

## 3. 常用快捷键

| 快捷键                           | 作用            |
| ----------------------------- | ------------- |
| `Cmd + K` / `Cmd + Shift + P` | 打开命令菜单        |
| `Cmd + ,`                     | 打开设置          |
| `Cmd + /`                     | 查看快捷键         |
| `Cmd + O`                     | 打开文件夹         |
| `Cmd + B`                     | 显示或隐藏侧边栏      |
| `Cmd + Option + B`            | 显示或隐藏 diff 面板 |
| **`Cmd + J`**                 | 显示或隐藏终端       |
| `Cmd + N`                     | 新建 thread     |
| `Cmd + F`                     | 当前 thread 内搜索 |
| `Cmd + Shift + B`             | 打开内置浏览器       |
| **`Cmd + G`**                 | 搜索thread      |

## 相关笔记

- **对比**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/AI 辅助编程生态工具(Claude Code 篇)|Claude Code 生态工具]] — Codex 指南提供另一种编码 Agent 使用基线。
- **对比**：[[30-Agent-Engineering-Agent工程/02-Coding-Agents-编程智能体/CC使用笔记|Claude Code 使用笔记]] — Codex 使用方式可用 Claude Code 实践作为参照。
- **应用**：[[90-Personal-个人/01-Knowledge-Tools-知识工具/Ob核心功能与MD语法|Obsidian 与 Markdown]] — Codex 使用流程可作用于 Obsidian 知识库。
