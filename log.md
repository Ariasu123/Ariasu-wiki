# 操作日志

- [2026-07-18] init: 仓库初始化 — 建立 LLM Wiki 骨架，导入首批技术笔记
- [2026-07-18] publish: 47 pages, 33 assets — 从私有 vault 同步
- [2026-07-25] fix: 全站 48 页 — 修复 GitHub 渲染：4 处 LaTeX 公式兼容（RoPE×3、MoE×1），212 处 wikilink + 11 处页内锚点批量转为标准 Markdown 相对链接；20 个目标不在本仓库的链接降级为纯文本
- [2026-07-25] fix: MoE学习笔记 — 8 处缩进在列表内的 $$ 公式块改为顶格，修复 GitHub 将其渲染为代码块的问题
- [2026-07-25] ingest: AI Infra 领域概览 — 从知乎文章（锦恢《AI Infra 软核教程一》）提炼五大板块、显存墙、训推难点，建立与推理/训练笔记的交叉引用
- [2026-08-05] ingest: Megatron-LM 论文精读, 张量并行（模型并行） — 从 Megatron-LM 论文（arXiv:1909.08053，本地 PDF 经 firecrawl parse）提炼层内模型并行切分策略与 f/g 通信算子、BERT LayerNorm 重排发现，补充 AI Infra 概览的训练侧交叉引用
