
关联阅读：可结合 [Agent Skill 机制](../01-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/Agent%20Skill%20%E6%9C%BA%E5%88%B6.md) 理解——机制篇讲 Skill 是什么、如何被路由与加载，本篇讲如何写好一个 Skill。

全文核心只有三句话：

1. **结构服务于内容**
2. **激活优于存储**
3. **结构可复用，内容禁止预制**

## 1. 为什么需要 Skill：从 Prompt 到 Skill

- Skill 最重要的作用是**引导**。Agent 默认倾向「找一个改动最少、能让当前现象消失的方案」；正确的做法是「先确认完整语义、状态所有权、调用链和业务不变量，再在所有正确方案中选改动最小的那个」。Skill 就是把这类纠偏固化下来的载体。
- 与 RAG、MCP 的分工：三者都在优化 Agent 执行流程，但各管一段——RAG 提供外挂知识库（有据可依），MCP 提供外部工具接入（探索世界），Skill 是流程的大脑（决定该怎么做）。
- 演进路径：Prompt 规范行为 → 单条提示词适用场景太单薄 → 把提示词 + 规则文档 + 脚本打包成一个「能力」= Skill。
- 相比 Prompt 的核心优势：**按需加载 / 渐进式披露**——Agent 先读 description 命中后再按需加载正文与资源，精简上下文、省钱。

## 2. Skill 的形态：单文件 → 文件夹化

### 2.1 单文件起步：结构服务于内容

参考项目 [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills)：整个项目 6 个文件、859 行，核心是一个 67 行的 `SKILL.md`，只有四条行为准则——没有 `rules/`、`workflows/`、`references/`，因为内容用不上。这不是偷懒，是对架构复杂度的准确判断。

三个可直接借鉴的设计：

1. **原则 + 检验句，而不是原则 + 解释**。`保持简洁`是声明，Agent 只能生成前抽象地「记住」；`问自己：资深工程师会觉得这过度复杂吗？`是检验句，Agent 生成后有钩子触发自我验证、决定要不要回滚。
2. **代码行为层面的 ❌/✅ 对比**。展示的不是明显错误，而是「看起来合理但时机错了」的改动（如修 bug 时顺手加 docstring、加强校验）——这类反模式 Agent 最容易踩，因为「看起来都对」，必须用真实例子提醒。
3. **按需引入复杂度**。设计结构前先回答三问：主题少于 3 个？绑定了任务流程？会持续演进？全「否」→ 单文件就是最优解；一上来就摆全套目录只会制造「项目很完整」的错觉。

### 2.2 文件夹化的信号

出现以下任一信号，单文件就撑不住了：

- **多主题**：SKILL.md 开始出现「### X 相关」的分节
- **任务路由**：不同类型任务需要读不同规则
- **需要沉淀教训**：同样的坑第二次踩，但没地方记录
- **多人协作 / 多项目复用**：规则开始有变体

标准目录结构：

```
skills/<name>/
├── SKILL.md          # 入口：路由表 + 优先级
├── rules/            # 长期约束
├── workflows/        # 步骤流程
├── references/       # 背景资料：架构、坑点、索引
│   └── gotchas.md    # 已知的坑（通常是最高价值内容）
├── docs/             # 可选：提示词、报告
└── scripts/          # 可选：辅助脚本、脚手架工具
```

两条铁律：

- **每个文件必须有独立的「被加载理由」**；几个文件永远一起加载，它们就应该是一个文件。
- 内容归属**按形式不按内容**：「你必须做 X」（指令性）→ `rules/`；「小心 X」（警告性）→ `references/gotchas.md`；「第 1、2、3 步」（流程性）→ `workflows/`。判断窍门：我能做 X 吗 → rules；这个坑怎么避 → references；我现在该做什么 → workflows。
- 行数是信号不是命令：超标触发评估，同一模块的内容即使超标也不应硬拆。

### 2.3 AGENTS.md 的定位：导航和约束，不是知识仓库

几百行的 AGENTS.md 会污染上下文、挤占理解代码的空间、规则互相冲突让 Agent「弱智化」。好的 AGENTS.md 只回答两个问题：你需要知道什么才能安全开始工作？遇到问题时该去哪里找更详细的资料？**分层是关键：AGENTS.md 管方向，其他文档管深度。**

## 3. Skill 三要素：Prompt、Context、Harness

三者缺一，Skill 都只是半成品：

- **缺 Prompt** → 激活率低 / 行为漂移
- **缺 Context** → 规则写了读不到
- **缺 Harness** → 今天好用明天乱来，出错也不知道

### Prompt：定义做什么

- **Description（触发描述）**：写在 `SKILL.md` frontmatter，是模型判断「要不要调用这个 Skill」的最重要依据。模型天然倾向 undertrigger（保守激活），所以 description 要主动覆盖用户可能的各种表达方式。门牌号写模糊了，模型就找不到门。
- **Body（执行指令）**：三个关键原则——用祈使句（「读取文件」优于「你应该先读取文件」）；解释「为什么」而不只是「做什么」，模型才能在边缘情况下合理判断；控制长度（正文建议 500 行以内，超出拆分为引用文件按需加载）。

### Context：决定知道多少

渐进式披露解决的根本矛盾是「信息越多越好，但上下文窗口有限」：只把「始终需要」的信息放顶层，「可能需要」的放引用文件。三个常见问题：Context 太少（行为随意发挥）、太大（后面的指令被静默忽略）、设计混乱（无关信息干扰判断）。rules / workflows / references 严格分目录不是形式主义，是让每个任务只加载最小必要集合。

### Harness：验证好不好用

很多失稳问题的根因不是模型，是 Harness 没给正确的拦截和重试机制。Harness 在 Skill 层做三件事：

1. **结构性拦截（防失控）**：薄壳里的 Red Flags STOP 块、从真实失败抄来的借口表、SessionStart hook，三者叠加才能扛住长会话压力。
2. **自动化验证（防漏项）**：自检脚本抓结构、行数、占位符残留、路由完整性——人类不擅长手动检查几十项，脚本能抓住 80% 的「遗忘型错误」。
3. **真实压力测试（防纸面合规）**：用真实用户可能说的提示词测 description 触发率；模型判断不了「读起来顺不顺」，真实输出必须人眼看。

## 4. SKILL.md：导航中心，不是百科全书

SKILL.md 应该很短（≤ 100 行），Agent 每次任务都要读它，所以它必须只讲「读什么、什么时候读」。核心板块：

```
## Always Read        ← 每次任务都读（2–3 个文件封顶）
## Session Discipline ← 多任务会话的强制再读
## Common Tasks       ← 按任务类型路由
## Known Gotchas      ← 最关键坑点一句话 + 指向 references/gotchas.md 的锚点
## Core Principles    ← 项目特有原则（每条带检验句）
```

### Description = 触发条件

description 不是摘要，是触发条件：

```yaml
# ❌ 错误 —— Agent 无法匹配
description: API development helper

# ✅ 正确 —— 明确触发短语 + 激活条件
description: >
  This skill should be used when the user asks to "add a new API endpoint",
  "write controller logic", "fix a backend bug", or "add a database migration".
```

**一个 description 写不好的 Skill，等同于不存在。** 若有多入口注册（如 `.cursor/skills/<name>/SKILL.md`），各处 description 必须完全一致，否则不同工具用不同判据，激活随机化。

### 两层路由

- **Always Read**：只放「任何任务都必须遵守」的约束（项目通用规则、编码规范）；领域特定规则绝对不放这里。
- **Common Tasks**：按任务类型路由。每条必须列**精确文件路径**；控制在 5–10 条，超出按领域分组；必须有「Other / unlisted task」**兜底条目**（没兜底，不在列表里的任务 Agent 会乱跑）；必须有多子任务路由。

### Known Gotchas：价值密度最高 / 阅读成本最高的内容

全量放 SKILL.md 会把路由中心变成坑点百科，全量放 references 又让 Agent 在任务路径上看不到。**一句话 + 锚点**是最佳平衡：Agent 每次都看得到哪些坑存在，真踩到才 deep read。硬约束：坑点只躺在 `references/` 里不算「捕获」，必须同时出现在任务路径上（workflow 完成检查 / SKILL.md Known Gotchas / rules 摘要）。

## 5. 让 Agent 保持「谦虚」：Session Discipline

症状：同一会话第 1 轮还按规则走，第 3 轮开始凭感觉写代码——新任务匹配的明明是另一条路由、另一份规则文件，Agent 却觉得「我已经知道了」。两个根因：**跨任务没重走路由**（把「Fix bug 路由」误当成「所有任务的路由」）；**上下文已悄悄压缩**（compact 之后 SKILL.md 早就不在 context 里）。

这不是 Skill 内容的问题，是 Harness 没给重读触发。光写「请每次重读 SKILL.md」不管用，必须**结构化地多层冗余**——每一层都可能被压缩器丢掉，留给你的是下一层，最坏情况下只剩薄壳。写法沿用「一条原则，一个检验」：

> 每个新任务——即使是同一会话的第 N 轮——必须重读 SKILL.md、重新匹配路由、重读必读文件。
> 检验：问自己「这次任务我读的文件和路由列的完全一致吗？」有差异立即回头重走。

## 6. 任务级控制：Task Anchor 任务锚点

Workflow 定义某类任务应该怎么做（模板）；Task Anchor 和原生 Plan 管理这一次任务怎么推进（实例）。它补齐的是「任务开始」和「任务执行中」的目标控制，防止 Agent 命中正确 Workflow 却在多轮调查修改中逐渐偏离目标（顺手扩大范围、没满足完成标准、沿用上一任务上下文、证据推翻假设后仍执行过期计划）。

Task Anchor 包含三要素：

- **Goal**：本次任务最终要得到什么结果
- **Boundaries**：本次任务的范围边界
- **Done When**：看到什么证据才能证明完成

## 7. 薄壳（thin shell）：跨工具兼容的基石

问题：「去读 `skills/*/SKILL.md`」这种自然语言指令在上下文压缩时会被当成普通描述丢掉，而**结构化的表格、清单会被保留更多**。薄壳的设计：不写「去读 SKILL.md」，而是把**最小可执行路由表直接内联**进每个 harness 的入口文件（CLAUDE.md / AGENTS.md / CODEX.md / GEMINI.md / `.cursor/` 等）——压缩后表格依然在，Agent 拿到新任务可以当场查表。缺哪个入口，那个工具就完全看不见你的 Skill。

薄壳三块核心内容（≤ 60 行），缺一不可：

1. **Quick Routing**：Task / Required reads / Workflow 三列路由表，必须有 `Other` 兜底行和多子任务行——压缩后这是 Agent 的唯一线索。
2. **Auto-Triggers**：事件 → 动作映射。最关键的一条是「同会话新任务 → 重读 SKILL.md、重新匹配路由」——「我前面读过了」不是理由，上下文会压缩，路由各不同。
3. **Red Flags — STOP**：把「就这一次跳过 AAR」这类借口前置拦截。压缩后只有薄壳会留下，Red Flags 是最后一道防线。

反例：soft-pointer-only 薄壳（只写一句「Please read skills/my-skill/SKILL.md」）短会话能工作，长会话 compact 后失效，且用户察觉不到——输出看起来合理，只是约束全丢了。

## 8. 两道机制级防线：Hook

Hook 不靠 Agent 自觉，是客户端在调用工具前后物理地拦一刀。两道防线不是重复，是接力：

- **SessionStart Hook（防遗忘）**：监听 `startup | clear | compact` 三个事件，自动读取 SKILL.md 并注入 context。分工：Session Discipline 管多任务会话的重读触发；薄壳 Auto-Triggers 管压缩后的路由兜底；SessionStart hook 管清空 / 压缩事件发生后的自动补弹。三者叠加才能扛住长会话 + 多任务 + 多次 compact。
- **PreToolUse Gate（防违规）**：模型的注意力方向是「回答用户的请求」而不是「遵守规则」——当用户说「demo 5 分钟后要用」，模型更倾向于帮用户。PreToolUse hook 拦在 Edit 核心规则文件之前，先拿到调用参数自行判断，非 0 退出码直接取消这次操作，Agent 根本没机会写进去。

局限：对能力较弱的模型，hook 也拦不住「读不到」的情况；且 hook 过多会限制 Agent 发挥。建议复杂任务用 Sonnet 及以上级别的模型。

## 9. 完整的任务闭环：Task Closure Protocol

Agent 常把「主体代码写完 + 测试通过」当作任务完成，但真实的结束还差一步：扫一遍刚才的工作，有没有踩到新坑、发现新规则、暴露已有规则的漏洞。这一步不是可选的 polish，是任务定义的一部分：

```
一个任务在以下条件全部满足前不算完成：
1. 主体工作完成并验证（代码跑通、测试通过、功能交付）
2. 30 秒 AAR 扫描（4 个问题——全部「否」则到此结束）
3. 任何一个「是」→ 过录入标准 → 通过则记录
```

**AAR 四问**：新模式？（用了未记录的模式或约定）新陷阱？（不提前知道就会浪费大量时间）缺失规则？（因缺规则走了弯路）过时规则？（现有规则不再准确）。触发门槛按「非琐碎任务」判断；跳过条件窄且明确：仅格式化、仅注释、仅依赖版本变更、无新教训的重构。

**Rationalizations to Reject（借口表）**：压力下 Agent 会自己编借口绕过协议，维护一张**原话捕获**的借口表。硬约束：只能从真实失败里加行，禁止凭空扩写——虚构借口 Agent 不会真说，混在一起会稀释压力值，稍微变形就能绕过去。

**Red Flags**：发现自己在想「这次 AAR 就算了」、声明完成但没跑扫描、gotcha 写进 reference 但没更新 workflow 完成清单、同一类 bug 修第二次但规则没动——立刻停下，不要自我协商。Red Flags 必须同时出现在薄壳里。

## 10. 多子 Agent 保证主 Agent 纯净（Subagent-Driven Development）

核心思想（借鉴 obra/superpowers）：不是一个大 Agent 从头做到尾，而是每个独立子任务派一个带着**干净上下文**的子 Agent，做完就退出。收益：主 Agent 上下文永远干净；主 Agent 兼 reviewer；可自主跑几小时不偏离原计划——因为每个 worker 只看合约，不看历史。

**启用条件**（满足任意一条）：独立子任务 ≥ 3 个；单任务会吃掉 > 30% 剩余 context；「探索 + 实现 + review」混合形态；即将多小时自动运行。都不满足就直接内联做——派发有开销，小任务不划算。

**四阶段流程**：Plan（每条子任务写成合约）→ Dispatch（合约原文作 prompt，不带主对话历史，无依赖可并行）→ 两阶段 Review（Stage A 查 spec 合规：产物文件、禁区、验收命令、drive-by 改动；Stage B 查质量）→ Merge 或 Reject（Stage A 不过就重派，**不要在主上下文里内联补**——那正好把主上下文污染回去）。

**子任务合约五字段**（任何字段不能空）：Goal（一句话、面向结果）、Inputs（允许读的确切文件）、Outputs（必须产出的确切文件）、Forbidden Zones（不许碰的范围，不确定默认禁）、Acceptance Criteria（可机械验证的命令，不是散文）。

**禁止项**：递归派发；worker 自审；中途往 worker 上下文塞「澄清」（合约错了就取消重写）；只跑一个 review stage；「worker 基本对了，剩下 10% 我在主上下文补」——最常见的借口，也是最污染主上下文的动作。

Harness 兼容性：只有 Claude Code 有原生 Task 工具，Cursor / Codex / Gemini 等只能降级——单上下文按 checklist 模拟，或每个子任务手动开新会话。降级模式仍有价值：两阶段 review + 合约本身就能捕获大部分缺陷。

## 11. 知识库自增强：录入标准与「激活优于存储」

AAR 扫出「新坑 / 新规则」之后，决定要不要记、记到哪、怎么写。随便记会把 Skill 变成冗长的日记本。

- **Recording Threshold（2/3 录入标准）**：可重复吗？不遵守代价高吗？从代码看不出来吗？至少 2/3 通过才录入。典型通过：框架生命周期坑、隐藏时序依赖、跨层交互陷阱。典型不通过：一次性变通方案、看代码就明白的事、风格偏好、官方文档已覆盖的内容。
- **Generalization Rule（泛化规则）**：记录的内容必须脱离当前项目上下文也能看懂。改写公式：`具体发现 → 抽象为通用 pattern → 说明不遵守的后果`。
- **录入格式选最轻的**：一句话 bullet → 一小段加到现有文件 → 新文件（通常不需要）。
- **激活优于存储**：高代价陷阱必须同时**存储**在正确的文件中，并**激活**在会触发它的任务路径上（workflow 检查项、SKILL.md Known Gotchas、rules 摘要）。判断方法：「下次 Agent 走正常任务路径时，会自然读到这条经验吗？」——不会，就只是「记下来了」，还没有「生效」。

## 12. 自我删除与迭代

只增不减的规则文档会变成屎山。Skill 必须学会「忘记」，且这件事本身需要 workflow：

- **Learn from Mistakes**：犯错被纠正后，先搜索规则是否已存在，再分类根因——规则缺失（过录入标准后新增）/ 规则过时（直接更新，过时规则比缺失更有害）/ 规则废弃（走清退流程）/ 规则未被遵循（检查醒目度，可能需要从 references 上浮到 SKILL.md 或薄壳）。
- **Rule Deprecation**：相关技术已移除 → 直接删；迁移中 → 加作用域标注；不确定 → 加 `<!-- DEPRECATED: reason, date -->` 注释，保留一个迭代周期再删。
- **评估式拆分 / 合并**：拆分三问（话题可分离？导航困难？拆后能独立存在？），合并三问（话题相关？合并后更好找？不超标？），全 Yes 才动手。
- **定期 drift 检查**：用两个真实不同类型的项目跑同一套 Quick Start 做 diff——骨架文件应该几乎一样，`rules/`、`gotchas.md`、Common Tasks 应该完全不同；如果一样，说明模板越界、把项目特定内容固化成了默认值。

## 13. 内容编写基本功（Anthropic 建议）

1. **不要陈述显而易见的事**。通用编程知识（「SQL 注入是坏事」）模型已经知道，写了只是浪费 token。判断标准：「资深开发者第一次看你的项目，什么会让他踩坑？」——项目特有的约定、与主流做法不同的地方、Agent 默认行为会出错的场景才值得写。
2. **避免过度指令化**。给约束和上下文，不把每一步写死：硬编码 `bg-blue-500` 在设计系统升级后全部失效，写「使用项目设计系统 token」能多活过几次重构。
3. **利用脚本和代码库**。Agent 调用已有脚本比从头写样板代码可靠得多。判断标准：「这段代码会在多少次任务里被 Agent 重写？」超过 2 次 → 写成脚本。
4. **保持 Skill 聚焦**。拆分信号：description 列了 10+ 个不同领域的触发短语；Common Tasks 15+ 条覆盖不相关工作；Agent 常为单一子领域任务激活整个 Skill。
5. **Skill 也是代码，需要测试和迭代**：写 Skill → 测激活（触发率）→ 测路由（每种任务读对文件了吗）→ 压力测试（时间压力 / 规则冲突 / 模糊 spec 下逐字抓借口）→ 观察失败 → 通过 AAR 更新。

## 14. 多 Skill：隔离与组合

**隔离**（一个 Skill 干一件事）五条硬约束：

1. 独立入口：每个 Skill 有自己的 SKILL.md，不共用
2. 注册：每个 Skill 都需要在各工具的注册入口登记，缺一个该工具就看不见
3. 优先级：frontmatter 用 `primary: true` 标记默认 Skill，全项目有且仅有一个（否则 hook 随机注入）
4. 共享规则：跨 Skill 通用约定放 `skills/shared/`，各 Skill 的 Always Read 指向它
5. 不要强行合并：合并只会让 description 变成「什么都能触发的万金油」

不要从 GitHub 堆叠大量同类 Skill——Skill 越多正确命中率越低，建议少而精 + 自动触发，而非主动引用。拆分信号：两个领域的 Common Tasks 完全不相交、description 要列 10+ 跨领域短语、gotchas 按领域自然分成两半。

**组合**（Skill 主动调用另一个 Skill）三种模式：

- **嵌入调用**：workflow 的某一步去读另一个 Skill 的 SKILL.md 并跟完它的 workflow，然后带产物返回。
- **直接路由**：Common Tasks 里某类任务直接指向另一个 Skill 的 workflow，不写自己的包装。
- **子 Agent 委派**：开子 Agent 把另一个 Skill 的执行整个隔离出去，只要结构化结果回来。

反模式：隐式传递依赖（调用了但下游项目没装这个 Skill → 静默失败，要 vendor 或加 escape 条件）；循环组合（A 调 B 调 A）；匿名调用（不给具体路径，Agent 自己猜，失去复现性）；组合当偷懒借口（项目特定规则积累后还得自己承起来）；跳过自己的 Task Closure。

## 15. Templates 防止结构性遗忘（上游 Skill 编写适用）

问题：让 Agent 实时生成脚手架（heredoc 写 SKILL.md、薄壳、workflows），它会漏段——同一个协议，五次生成出五个版本。解法：上游项目放 `templates/` 目录，下游 `cp -R` + 一次 `sed` 替换占位符。Agent 不再「生成」，只做「填空」。每个 `<!-- FILL: -->` 标记就是一个 TODO，没填完就是 bug（收尾用 `grep -rn 'FILL:'` 检查）。

两条铁律：

1. **结构可以预制，内容禁止预制**。`{{NAME}}` / `{{SUMMARY}}` 机械替换；`<!-- FILL: ... -->` 必须人工 / Agent 判断，留空就是 bug。
2. **两个真实项目测试**：往 `templates/` 加任何东西前问——「一个 Go 后端微服务和一个 React 动画站都会复制这份模板，它们会同意这块内容吗？」会 → 是协议 / 骨架；不会 → 降级为 FILL 标记或写进 ANTI-TEMPLATES.md 明确禁止。

**ANTI-TEMPLATES.md** 是「我们故意不预制」清单（默认 lint 规则、commit message 格式、预填坑点、默认目录结构……），它是反漂移的压力器——清单越长，说明 review 越严肃。Skill 不是创意写作，是工程基础设施：目录结构、frontmatter、FILL 标记、薄壳路由表全是承重构件，漏一个 Agent 就会静默退化。

## 16. 自动化验证脚本兜底

80% 的 Skill 失败来自**遗忘型错误**，不是理解型错误——引用文件不存在、description 和注册入口对不上、占位符漏替换、SKILL.md 悄悄超 100 行。这些用脚本就能抓住：

- **smoke-test（自检测试）**：检查结构、行数、占位符残留、路由完整性、各入口一致性。关键设计：把 SKILL.md 本身当作唯一数据源，Common Tasks 里新增的引用会被自动发现。
- **test-trigger（触发率测试）**：从 Common Tasks 自动生成真实用户可能说的提示词，检查 Agent 能否找到 Skill——单独读一遍觉得没问题，跑一遍才发现一半触发短语命中不了；优化 description 措辞可直接提升命中率。

运行时机：初次迁移完、编辑 SKILL.md 或薄壳之后、模板升级之后、宣布「完成」之前。脚本抓不到「description 不够精准」「路由设计不合理」这类语义问题，不能替代人眼检查。

## 17. 清晰的文件边界：防止 Skill 变成日记本

Agent 会把「记录教训」过度解读成「把整次会话存档」——`references/` 下冒出 `2026-04-14-session-notes.md` 这类文件，违反三条核心设计：泛化规则（会话日志是项目叙事，不是可复用知识）、激活优于存储（没有路由，未来永远命中不到）、自维护（每次会话一个文件，无限膨胀）。

记录位置判断表：

| 内容类型 | 目标位置 |
|---|---|
| 稳定约束 / 通用原则 | `rules/` |
| 陷阱、架构笔记、生命周期坑 | `references/` |
| 有序步骤 / 完成检查清单 | `workflows/` |
| 会话历史 / 调试过程 | **不要写进 Skill**——用 git / CHANGELOG |

## 18. 踩坑清单精选

- 让 Agent 每次实时生成脚手架 → 会漏段；改成 `cp -R templates/` + sed
- 把「具体业务 spec 示例」预制进 templates → 下游照抄例子不写自己的；让 FILL 标记逼它思考
- Rationalizations 表凭空扩写 → 稀释真实借口的压力值；只能从真实失败抄
- Auto-Triggers 只写在 workflow 里不写进薄壳 → 压缩后薄壳是最后防线，薄壳丢了就全丢了
- 薄壳硬压到 ≤ 15 行 → Red Flags + Auto-Triggers 写不下，协议碎片化；≤ 60 行是合理上限
- 多 harness 项目缺 GEMINI.md / Copilot 入口 → 这些工具读不到你的 Skill，等于没有
- Hook 配置用 flat 格式 → Claude Code v2.1+ 的 PreToolUse 只认嵌套格式，看起来注册了但 Edit 时不触发；SessionStart 恰好两种都吃，初期误以为一切正常，实际静默失效
- 用 Agent SDK 子 Agent 测 hook 效果 → 子会话不触发 PreToolUse hook，永远是 false negative；只能开新交互会话手动测
- 子 Agent prompt 里写绝对路径 → 绕过 worktree 隔离直接改到主仓，「感觉隔离了，实际污染了」
- 把 Agent「漏读规则」归因成模型笨 → 先确认文件到底在不在上下文里：能不能拿到入口文件是**客户端行为**（确定性的），能不能遵守是**模型行为**（概率性的）；把客户端路由表写全，比换模型便宜得多

## 相关笔记

- **概念机制**：[Agent Skill 机制](../01-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/Agent%20Skill%20%E6%9C%BA%E5%88%B6.md) — Skill 的三层结构（Metadata / Instruction / Resource）与路由加载机制；本篇是其对应的编写实践。
- **三要素框架**：[从 Prompt 到 Harness：企业级 Agent 工程演进](%E4%BB%8E%20Prompt%20%E5%88%B0%20Harness%EF%BC%9A%E4%BC%81%E4%B8%9A%E7%BA%A7%20Agent%20%E5%B7%A5%E7%A8%8B%E6%BC%94%E8%BF%9B.md) — Prompt / Context / Harness 工程分层的企业级视角。
- **运行环境**：[Harness Engineering](Harness%20Engineering.md) — 薄壳、SessionStart / PreToolUse hook、AAR 任务闭环是 Harness 原则在 Skill 层的具体落地。
- **执行循环**：[Loop Engineering](Loop%20Engineering.md) — Task Closure Protocol 与 AAR 是 Loop 反馈与停止条件机制的 Skill 层实现。
- **总览**：[Agent 核心 16 问](../01-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/Agent%20%E6%A0%B8%E5%BF%83%2016%20%E9%97%AE.md) — Skill 是 Agent 能力组织与渐进加载机制。
