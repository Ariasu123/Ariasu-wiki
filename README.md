# LLM Wiki

个人技术知识库，基于 [Karpathy 的 LLM Wiki 模式](https://gist.github.com/karpathy) 构建：用 Obsidian 管理 Markdown 笔记，由 AI agent 持续维护（ingest / query / lint），知识不断复利积累。

## 内容导航

全站目录见 [index.md](index.md)，操作日志见 [log.md](log.md)。

| 目录 | 内容 |
|---|---|
| `wiki/llm/` | 大语言模型：架构、预训练、后训练、工程实践 |
| `wiki/ai-infra/` | AI 基础设施：推理加速、算子优化、学习路线 |
| `wiki/agent-engineering/` | Agent 工程：基础概念、Coding Agents、框架与工作流 |
| `wiki/rag/` | RAG 与知识库 |

## 维护方式

本仓库由 AI agent 按 [CLAUDE.md](CLAUDE.md) 中的规则维护：

- **ingest**：新资料 → 提炼关键点、创建/更新页面、建立 `[[wikilink]]` 交叉引用
- **query**：回答问题并引用 wiki 页面，回答反哺知识库
- **lint**：定期检查孤儿页、断链、矛盾与过期内容

人机分工：人负责提问、筛选信源、判断价值；AI 负责交叉引用、格式维护、索引更新。

## License

[CC BY 4.0](LICENSE)
