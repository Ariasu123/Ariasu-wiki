# Wiki 维护规则（给 AI Agent）

本仓库是一个 LLM Wiki：由 AI agent 持续维护的个人知识库。你（AI agent）是这个 wiki 的"写手"，负责所有编辑工作；人类负责提问、筛选信源和判断价值。

## 目录结构

- `wiki/` — 所有知识页面，按主题分目录（`Agent/`、`Infra/`、`LLM/`）
- `index.md` — 全站内容目录，按一级主题 + 二级子目录分组（由 `node publish-wiki.js` 自动生成，不要手改）
- `log.md` — 操作日志，每次操作必须追加

## 信源获取（firecrawl，可选）

ingest 时可用 Firecrawl 获取原始资料（需本机已安装 Firecrawl CLI 并配置 skill，即 `firecrawl` 命令可用）：

- **网页链接**：`firecrawl scrape <url> -o .firecrawl/<name>.md`
- **本地文件**（PDF、DOCX、XLSX 等）：`firecrawl parse <file> -o .firecrawl/<name>.md`

解析产物统一落 `.firecrawl/`（已 gitignore，不入库），通读后按 ingest 流程提炼入库。

## 三个核心操作

### ingest（录入新资料）

1. 通读原始资料全文。
2. 检查相关现有页面（先查 `index.md`，必要时全文搜索）。
3. 提炼关键点，创建新页面或更新已有页面——不是摘要，而是把线性资料拆解成网状知识结构：概念、人物、工具、方法各自成页，互相链接。
4. 建立交叉引用；发现与已有内容矛盾时，在页面中显式标记冲突，不要静默覆盖。
5. 运行 `node publish-wiki.js` 重新生成 `index.md`，追加 `log.md`。

### query（查询）

1. 先查 `index.md` 定位相关页面，引用页面路径回答。
2. 如果回答过程中产生了新的综合性结论，征得同意后反哺更新相关页面，并追加 `log.md`。
3. 如果 wiki 中没有相关内容，明确说明，不要编造。

### lint（体检）

定期检查并输出待办清单：

- 孤儿页（没有任何入链的页面）
- 断链（指向不存在页面的链接）
- 矛盾信息（不同页面说法不一致）
- 过期内容（`updated` 久远且相关领域已有新进展）

只报告和提出建议，大规模修改前先列出计划征得同意。

## 页面规范

- **页面不需要 YAML frontmatter**（title/tags/created/updated/status 一律不写），直接从正文开始。
- 标题即文件名，简洁、准确、可检索。
- 页面的创建与更新时间由 `log.md` 的操作记录承担，不再写进页面。

## 命名规范

- **页面以知识主题命名，不以信源命名**：名称里不含具体文章标题、会议、比赛、产品版本或公司名（如「MLSys 2026 FlashInfer 比赛」「Photon」「Hy3 preview」均属信源，应抽象为主题名）；例外是技术名词/系统本身即主题（vLLM、PagedAttention、Megatron-LM、Claude Code）。
- 中文为主、术语保留英文；中英文之间加空格；括号统一**全角**（如 `DPO（直接偏好优化）`）。
- 禁止讲次编号后缀（`_16`、`_10`）和「学习笔记」一类无信息量后缀。
- 图片同样语义化命名（小写连字符，如 `numa-dual-socket-8gpu.png`），禁止 `Pasted image ...`、哈希名。

## 链接规范

- 页面间引用统一用标准 Markdown 相对路径链接：`[显示名](相对路径.md)`，路径需 URL 编码（如 `[MoE学习笔记](MoE%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md)`）。**目标含空格或括号时必须编码**（空格 `%20`、半角括号 `%28`/`%29`、全角括号百分号编码），否则 GitHub 会把链接截断、显示为源码。不使用 Obsidian 的 `[[wikilink]]` 语法——GitHub 无法渲染。
- 页内锚点用 `[显示名](#标题slug)`，slug 规则同 GitHub：小写、去标点、空格转连字符。
- 图片放本主题目录下的 `_assets/` 子目录（如 `wiki/LLM/_assets/`），用标准 markdown 相对路径引用，不用外链。

## LaTeX 公式规范（GitHub MathJax 兼容）

> 根因：GitHub 先解析 Markdown、再渲染数学公式。公式里的 `_`、`*`、`^`、`[` 等字符在第一阶段就可能被 Markdown 吃掉或污染，导致公式不渲染、显示为源码或部分渲染。行内公式 `$...$` 比块级 `$$...$$` 脆弱得多。

1. **下标/上标一律加花括号**（禁止裸写）：
   - 下标：`E_{p}`、`A_{1}`、`X_{1}A_{1}`；禁止 `E_p`、`A_1`、`X_1A_1`
   - 上标：`^{\top}`、`k^{*}`、`x^{2}`；禁止 `^\top`、`k*`、`x^2`
2. **复杂公式一律用块级 `$$...$$`**：独占一行、行首顶格、不在列表项或表格单元格内。行内 `$...$` 只放单个简单符号（`$A$`、`$f$`、`$b$`），不放带下标/分数/矩阵/长表达式的公式。
3. `\text{}` / `\mathrm{}` 内不要出现下划线（`\_` 会被 GitHub 反转义后报错），用连字符代替，如 `\text{rotate-half}`。
4. 公式内不裸写 Markdown 特殊字符：`*`（斜体/粗体）、`_`（斜体）、`#`（标题）、`[`/`]`（链接语法）等必须用 `{}` 包裹或用 `\%`、`\#` 等转义。
5. **禁用 `\operatorname`**：GitHub MathJax 直接报 "macros are not allowed" 并让整块公式显示为源码。一律用 `\mathrm{}` 代替，如 `\mathrm{clip}`。
6. **行内公式前是中文/全角标点时必须加空格**：`：$x$`、`（$x$）`、`中文$x$` 都不会被 GitHub 识别为公式（原样显示 `$`），写成 `： $x$`、`（ $x$）`。行内公式后面跟中文不受影响。
7. **重音命令+下标的行内写法**：行内公式里 `\hat{f}_{e}`、`\bar{s}_{b,e}` 这类 `}_{` 结构会触发 Markdown 斜体规则吃掉公式（`_{...}..._` 被配成一对 `<em>`）。行内写法把重音命令的花括号去掉——`\hat f_{e}`、`\bar s_{b,e}`（下标花括号保留），或直接改用块级 `$$`。
8. 写完公式后自查：`grep -nE '\$[^$]*[A-Za-z0-9]_[A-Za-z0-9]' <file>` 应无输出（裸下标）；`^\字母` 上标同样禁止；块级 `$$` 必须行首。可用 GitHub markdown API（`POST https://api.github.com/markdown`，`mode: gfm`）实测渲染结果。

## Callout 规范（GitHub 兼容）

GitHub 只支持五种 callout：`[!NOTE]`、`[!TIP]`、`[!IMPORTANT]`、`[!WARNING]`、`[!CAUTION]`。Obsidian 的 `[!abstract]`、`[!info]`、`[!example]` 等在 GitHub 上会原样显示为文本，入库时一律映射为五种之一（如 abstract/info → note）。

## 日志格式

`log.md` 每条一行，固定格式，便于解析：

```
- [YYYY-MM-DD] <op>: <pages> — <summary>
```

例：`- [2026-07-18] ingest: MoE学习笔记, Attention 与 GQA — 从知乎文章提炼 MoE 路由机制，补充与 GQA 的关联`

## 人机分工

- **人**：提出好问题、筛选信源、判断内容价值、审批大规模改动。
- **AI**：交叉引用、摘要更新、格式维护、索引与日志维护、图谱结构优化。
