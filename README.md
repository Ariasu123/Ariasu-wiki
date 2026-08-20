# Ariasu Wiki

**English** | [中文](README_zh.md)

A personal technical knowledge base built on [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy): notes are managed as Markdown files and continuously maintained by an AI agent (ingest / query / lint), so knowledge compounds over time.

## Contents

The full table of contents lives in [index.md](index.md). **`index.md` is auto-generated** — run `node publish-wiki.js` to rebuild it after adding or renaming pages.

| Directory | Contents |
|---|---|
| `wiki/Agent/` | Agent engineering: foundations, coding agents, harnesses & workflows, project references, RAG |
| `wiki/Infra/` | AI infrastructure: inference acceleration, kernel/operator optimization, learning roadmaps |
| `wiki/LLM/` | Large language models: architecture, pretraining, post-training, engineering practice, review notes |

Each topic directory is further organized into numbered subdirectories (`01-`, `02-`, ...), with attachments stored in per-topic `_assets/` folders.

## Workflow

This is an AI-agent-driven wiki. The daily loop:

1. **Pick a topic** you want to learn about.
2. **Feed the agent a source** — a URL or a local file. The agent calls its **Firecrawl skill** to fetch the content: scraping web links, or parsing local files (PDF, DOCX, ...). *(Prerequisite: install the [Firecrawl CLI](https://github.com/firecrawl/firecrawl) and configure the skill, i.e. the `firecrawl` command available on your machine.)*
3. **The agent distills it into the wiki** — creates cross-linked pages, rebuilds the index with `node publish-wiki.js`, and appends an entry to `log.md` (a local operation log; ignored by git by default — track it yourself if you like).

## How it's maintained

This repo is maintained by an AI agent following the rules in [CLAUDE.md](CLAUDE.md):

- **ingest**: new material → distill key points, create/update pages, build standard Markdown cross-references
- **query**: answer questions citing wiki pages; answers feed back into the knowledge base
- **lint**: periodically check for orphan pages, broken links, contradictions, and stale content

Division of labor: humans ask questions, curate sources, and judge value; the AI handles cross-referencing, formatting, and index upkeep.

## Getting started

1. Clone the repo and add your own topics under `wiki/`.
2. (Optional) Install the Firecrawl CLI and configure the skill to ingest URLs or local files.
3. Run `node publish-wiki.js` to (re)build `index.md`.
4. Let your AI agent read `CLAUDE.md` and start ingesting / querying / linting.

## License

[CC BY 4.0](LICENSE)
