# Ariasu Wiki

[English](README.md) | **中文**

个人技术知识库，基于 [Karpathy 的 LLM Wiki 模式](https://gist.github.com/karpathy) 构建：用 Markdown 管理笔记，由 AI agent 持续维护（ingest / query / lint），知识不断复利积累。

## 内容导航

全站目录见 [index.md](index.md)。**`index.md` 为自动生成**——新增或重命名页面后运行 `node publish-wiki.js` 即可重建。

| 目录 | 内容 |
|---|---|
| `wiki/llm/` | 大语言模型：模型架构、预训练、后训练、工程实践、复习总结 |
| `wiki/ai-infra/` | AI 基础设施：推理加速、算子优化、学习路线 |
| `wiki/agent-engineering/` | Agent 工程：基础概念、Coding Agents、运行框架与工作流、项目参考 |
| `wiki/IC/` | 集成电路：AI for 硬件设计（RTL） |
| `wiki/rag/` | RAG 与知识库 |

每个主题目录下按 `01-`、`02-` 等编号子目录进一步分类，图片等附件放在各主题的 `_assets/` 子目录。

## 工作流

这是一个由 AI agent 驱动的 wiki，日常循环三步：

1. **确定想了解的知识** —— 选定要学习/收录的主题。
2. **把文件或链接输入给 agent** —— agent 调用 **firecrawl skill**（能力）获取内容：抓取网页链接，或解析本地文件（PDF、DOCX 等）。*（前提：本机需安装 [Firecrawl CLI](https://github.com/firecrawl/firecrawl) 并配置 skill，即 `firecrawl` 命令可用。）*
3. **agent 提炼入库** —— 创建互相链接的知识页面、建立交叉引用，用 `node publish-wiki.js` 重建索引，并在 `log.md` 追加操作记录（本地操作日志，默认已被 git 忽略，是否入库可自行决定）。

## 维护方式

本仓库由 AI agent 按 [CLAUDE.md](CLAUDE.md) 中的规则维护：

- **ingest**：新资料 → 提炼关键点、创建/更新页面、建立标准 Markdown 交叉引用
- **query**：回答问题并引用 wiki 页面，回答反哺知识库
- **lint**：定期检查孤儿页、断链、矛盾与过期内容

人机分工：人负责提问、筛选信源、判断价值；AI 负责交叉引用、格式维护、索引更新。

## 快速开始

1. Clone 本仓库，在 `wiki/` 下添加自己的主题目录。
2. （可选）安装 Firecrawl CLI 并配置 skill，用于收录网页链接或本地文件。
3. 运行 `node publish-wiki.js`（重新）生成 `index.md`。
4. 让你的 AI agent 阅读 `CLAUDE.md`，开始 ingest / query / lint。

## License

[CC BY 4.0](LICENSE)
