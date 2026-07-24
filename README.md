# Ariasu Wiki

**English** | [中文](README_zh.md)

A personal technical knowledge base built on [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy): notes are managed as Markdown files and continuously maintained by an AI agent (ingest / query / lint), so knowledge compounds over time.

## Contents

See [index.md](index.md) for the full table of contents and [log.md](log.md) for the operation log.

| Directory | Contents |
|---|---|
| `wiki/llm/` | Large language models: architecture, pretraining, post-training, engineering practice, review notes |
| `wiki/ai-infra/` | AI infrastructure: inference acceleration, kernel/operator optimization, learning roadmaps |
| `wiki/agent-engineering/` | Agent engineering: foundations, coding agents, harnesses & workflows, project references |
| `wiki/rag/` | RAG and knowledge bases |

Each topic directory is further organized into numbered subdirectories (`01-`, `02-`, ...), with attachments stored in per-topic `_assets/` folders.

## How it's maintained

This repo is maintained by an AI agent following the rules in [CLAUDE.md](CLAUDE.md):

- **ingest**: new material → distill key points, create/update pages, build `[[wikilink]]` cross-references
- **query**: answer questions citing wiki pages; answers feed back into the knowledge base
- **lint**: periodically check for orphan pages, broken links, contradictions, and stale content

Division of labor: humans ask questions, curate sources, and judge value; the AI handles cross-referencing, formatting, and index upkeep.

## License

[CC BY 4.0](LICENSE)
