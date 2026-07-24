# Ariasu Wiki

[English](README.md) | **中文**

个人技术知识库，基于 [Karpathy 的 LLM Wiki 模式](https://gist.github.com/karpathy) 构建：用 Markdown 管理笔记，由 AI agent 持续维护（ingest / query / lint），知识不断复利积累。

## 内容导航

全站目录见 [index.md](index.md)，操作日志见 [log.md](log.md)。

| 目录 | 内容 |
|---|---|
| `wiki/llm/` | 大语言模型：模型架构、预训练、后训练、工程实践、复习总结 |
| `wiki/ai-infra/` | AI 基础设施：推理加速、算子优化、学习路线 |
| `wiki/agent-engineering/` | Agent 工程：基础概念、Coding Agents、运行框架与工作流、项目参考 |
| `wiki/rag/` | RAG 与知识库 |

每个主题目录下按 `01-`、`02-` 等编号子目录进一步分类，图片等附件放在各主题的 `_assets/` 子目录。

## 维护方式

本仓库由 AI agent 按 [CLAUDE.md](CLAUDE.md) 中的规则维护：

- **ingest**：新资料 → 提炼关键点、创建/更新页面、建立 `[[wikilink]]` 交叉引用
- **query**：回答问题并引用 wiki 页面，回答反哺知识库
- **lint**：定期检查孤儿页、断链、矛盾与过期内容

人机分工：人负责提问、筛选信源、判断价值；AI 负责交叉引用、格式维护、索引更新。

## License

[CC BY 4.0](LICENSE)
