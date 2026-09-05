Comet 是一套 **agent skill harness**——包在 AI 编码 agent 外面的一层工程化框架。模型继续负责思考和写代码，comet 负责管理一次 change（需求/变更）怎么开始、当前走到哪里、中断后怎么恢复，以及满足哪些条件才算真正完成。

它回答的核心问题是：**模型越来越能写代码之后，怎么把一次对话变成可以稳定交付的项目任务。**

> [!NOTE]
> 本文基于 linux.do 社区帖子《强模型还需要工作流吗？从 comet 来看 ai 工程化开发》（作者 luokakale，基于 comet 0.4.0-beta.17/18 的实际体验）提炼整理，并结合了帖子回复区的讨论。

## comet 是什么：三大块能力

拆开来看，comet 由三部分组成：

1. **需求交付工作流**：native 和 classic 两套独立工作流，都能把一次需求从想法推进到实现、验证和归档，只是执行方式不同。
2. **状态恢复与验收**：持续记录 change 走到哪里、还缺什么产物、验证有没有通过。中断后可以继续，验证失败可以回到实现阶段修复，不需要重新猜项目状态。
3. **skill 工程化**：把团队自己的工作流整理成可复用的 skill，并支持评估、审核、发布和分发。这一块已超出单次需求开发，走向团队能力沉淀。

日常使用的主要入口：

| 入口 | 作用 |
|------|------|
| `/comet` | 启动一次 change，根据项目配置进入 native 或 classic |
| `/comet-any` | 创建、优化、组合可复用的 skill |
| `comet eval` | 评估 skill 的实际效果 |
| `comet publish` | 审核和发布 `/comet-any` 生成的产物 |
| `comet status` / `comet doctor` | 查看当前状态、下一步动作和恢复建议 |

安装上，comet 通过全局 npm 包（`npm install -g @rpamis/comet`）+ 项目内 `comet init` 完成，已适配 30+ 个 AI 编码平台——它会按各平台的目录约定安装 skill、rules 和 hooks，让同一套工作流可以放进多个编码工具。但安装成功只代表 comet 被平台识别，具体能用到什么程度取决于平台本身的能力（如是否支持 hooks 和独立 subagent execution）。

## classic 模式：五阶段强约束

classic 是 comet 早期的工作流形态，它把 **openspec（管 change 和 spec）+ superpowers（管设计、计划、测试等工程方法）** 作为外部依赖，comet 在外层负责阶段路由、guard、handoff 和恢复——解决两套 skill 组合使用时"什么时候切换、中断后从哪里继续"的协调问题。

### full 路径：open → design → build → verify → archive

- **open**：调查项目、确认需求边界，生成 `proposal.md`、`design.md`、`tasks.md`、delta spec 和 `.comet.yaml`。其中 delta spec 是需求和验收标准的权威来源，后续设计和实现都围绕它展开。
- **design**：把需求文件整理成 handoff 交给 superpowers 生成 design doc。交接包保存来源和 hash，需求文件变化时 guard 会要求重新生成。可选的上下文压缩（完整保留 delta spec，其余文件换成 hash 引用，约省 25–30% token）也发生在这里。
- **build**：生成实现计划，选择 branch/worktree、执行方式、是否 TDD 和 review mode。
- **verify**：light 级做构建、测试、安全和轻量审查；full 级还会检查设计一致性、验收场景和 spec 漂移。验证失败回到 build。
- **archive**：最终确认后，delta spec 合并进主 spec，change 移入 `docs/openspec/changes/archive/`，归档完成。

### hotfix / tweak：轻量路径

两者都跳过完整 design，走 `open → build → verify → archive`：

- **hotfix**：修已有 bug、无新增能力/接口/架构变化时进入，build 中包含根因消除检查。
- **tweak**：可收敛到单个 openspec change、不需要完整设计的调整。

`/comet` 会根据需求描述、仓库状态和风险信号**自动路由** full/hotfix/tweak；证据不足或信号冲突时会停下来让人确认，不擅自猜路径。过程中若出现跨模块、schema、新 public API 或深层架构变化，会暂停确认是否升级 full，已完成的产物不会丢失。

### 状态文件与恢复

classic 把进度持续写进项目文件（`.comet.yaml` 等），因此长任务可以跨会话继续——新会话里输入 `/comet 继续`，它会依次：读取项目配置确定工作流 → 查找进行中的 change（多个则先选择）→ 对照状态文件与磁盘产物判断停在哪个阶段 → 复用原 branch/worktree 调用对应 skill。状态与磁盘对不上时会按实际产物重新校准；branch/worktree 绑定不一致时暂停确认。

## native 模式：为强模型设计的轻约束工作流

强模型（如 GPT-5.x 这一代）本身已经能完成探索、计划、实现和测试，再让它严格走一遍完整方法链反而会与模型自己的判断冲突——变慢、费 token。native 模式因此把**具体怎么执行交还给模型**，只保留需要长期保存的外置能力。

记住这三层分工：

> [!IMPORTANT]
> - **模型**负责怎么去实现
> - **runtime** 负责判断是否完成
> - **项目文件**负责中断后怎么恢复

native 是**自包含**工作流，不需要 openspec 和 superpowers 等外部 skill。

### 阶段流转：shape → build → verify → archive

**1. 创建 change**：`/comet 给当前项目增加邮箱和密码登录接口`——读取 `.comet/config.yaml` 的 `default_workflow` 确定性地进入 native（不会根据任务大小临时猜工作流）。创建时选择执行工作区：current（当前目录串行）或 worktree（多 change/多 agent 并行，各有独立分支和工作目录）。产物分两处：

```
docs/comet/changes/<change>/        # 跟项目走：brief、spec、状态、最终验证报告
.comet/runtime/native/changes/<change>/  # 本机运行状态：日志、锁、事务
```

**2. shape（需求澄清与验收定义）**：先调查仓库里已存在的接口、数据结构和规范，把能确认的事实查清，只把**真正影响产品行为的问题**交给人决定——左边是影响验收标准的产品决策，右边的实现方法交给模型自己定。澄清模式支持批量（batch，按依赖分轮提问）等方式。产物逐步生成 `brief.md`、`specs/<capability>/spec.md`、`comet-state.yaml`。所有问题处理完后，comet 会把目标、范围、关键决定、非目标和验收标准汇总，**只有人明确确认这份共享理解，runtime 才允许进入 build**。

**3. build（模型自主实现）**：native 不再切换到其他阶段 skill，每次续行重新读取 brief、spec、状态和仓库现状。模型拥有很大的执行空间——用什么方法、要不要 TDD、要不要调子代理，自己判断。完成后提交 builder handoff（改了什么、对应哪些验收项、建议重点检查什么），但 handoff 不等于通过：模型无权自己把 change 推进到 archive。若发现 spec 缺少影响产品行为的决定，退回 shape 重新确认；只是实现不达标则留在 build 修复。

**4. verify（runtime checks + 只读 verifier 独立验收）**：这是 native 最硬的一层。

- **runtime checks**：真实执行测试、构建、lint 等命令，非零退出和超时不能被一段"已通过"的文字覆盖。完整 stdout/stderr 留在本机日志，`comet-state.yaml` 只保存退出码、耗时和摘要。同一代码候选已完成的检查可复用，不重复跑。
- **只读 verifier**：与 builder 分离的新执行上下文，只根据代码、brief、spec 和检查证据对全部验收项（A1…An）重新判断——**自己写完不能给自己通过**。verifier 需要额外检查时也必须通过 request-checks 交给 runtime 执行，不能自己生成检查结果。
- 平台不支持独立 verifier execution 时，comet 会明确记录语义验收已降级，并在 archive 前要求人工确认——不把同一段模型上下文伪装成独立验收。

**5. repair loop（有界修复）**：验证失败时，runtime 把具体缺口写进 continuation、送回 build，而不是让 verifier 顺手改代码。关键规则：

> [!NOTE]
> 新一轮 verifier 仍然要重新检查**完整**的 A1…An，不能只看刚失败的项——一次修复可能破坏原本已通过的行为。

为避免在 build/verify 之间空转，runtime 用计数器判断"真实进展"：未解决验收项减少、失败检查转通过、缺失信息补齐才算进展；重复跑同一条失败命令、只改说明文字不会重置停滞计数。实现失败轮次上限由 `max_verify_failures` 控制（默认 5），超限后流程停下，把失败原因和 resolution_action 交给人处理。

**6. portable state（跨会话与跨设备恢复）**：`comet-state.yaml` 是恢复机制的核心，包含当前 phase/status、iteration 与 attempt、完整验收项和检查摘要、builder handoff 与 verifier 结论、blockers、下一步 continuation。可同步的项目状态与仅本机使用的执行状态（`.comet/runtime/native/`）分开放——换设备时只需同步代码和 portable state，本机 runtime 可重建。会话中断后 `/comet 继续` 即可恢复；verify 中途被断会标记 interrupted 并重做必要检查；代码未同步、产物根变化或 branch/worktree 绑定不一致时返回 await-user/blocked，**不会在错误目录里擅自写代码**。

**7. archive（归档）**：全部验收通过后进入 archive-ready。archive 不重新执行 verify（复用已有结果），生成供人阅读的 `verification.md`，并把 target spec 按 `create`/`modify`/`remove` 声明应用到 canonical spec（`docs/comet/specs/`），change 目录移入 `docs/comet/archive/`。两个 change 同时改同一 capability 时会停下要求确认归档顺序，不允许直接覆盖。归档写 spec、生成报告、移动目录使用**可恢复事务**，中断后重跑同一动作从事务状态继续。

### native 的"轻"体现在哪

native 与 classic 对小任务采用不同的轻量策略：classic 按任务意图**缩短流程**（hotfix/tweak 跳阶段），native 则**保留完整生命周期、把阶段内部执行压轻**——小修改时 shape 可能没有额外产品问题、verify 只跑必要检查，但 brief、spec 和状态产物依然保留。是否值得进 comet 看的不是代码量，而是这次改动需不需要长期管理。

## 两种约束哲学：怎么选

| 维度 | classic | native |
|------|---------|--------|
| 约束方式 | 把工程方法交代清楚（spec、设计、计划、TDD、review 各有阶段和 skill） | 信任强模型判断，守住需求、验收和状态这些不能丢的部分 |
| 外部依赖 | openspec + superpowers | 自包含 |
| 适合模型 | 能力相对弱、需要执行指导的模型 | 强模型（避免与模型自身判断冲突） |
| 适合场景 | 高风险、强协作、强审计的交付场景 | 常规开发任务、个人项目快速交付 |
| 轻量路径 | hotfix/tweak（缩短流程） | 阶段内压缩（流程不变、执行变轻） |

> [!NOTE]
> native 不是 classic 的替代品。模型能力只影响"需要多少执行指导"，项目风险、团队协作和交付治理是另一条维度。未来更可能的格局是：native 覆盖更多常规开发，classic 留在高风险、强协作场景。

## comet vs trellis：两条工程路线

两者在 spec、任务拆分、状态管理、跨会话恢复等主流程上高度相似——这是所有 AI 开发工程化框架的必备能力。区别在于侧重点：

- **trellis**：偏项目**长期维护**——规范沉淀、任务上下文、经验积累、跨工具协作。像给项目建一套可持续积累的知识和工作记忆。
- **comet**：偏**单次需求的可靠交付**——围绕一个 change 管需求确认、阶段推进、验收、失败修复和归档。即使去掉 superpowers 也能保证每次稳定交付。

一句话概括（来自作者在社区讨论中的总结）：**trellis 主规范，comet 主流程**；做需求时 comet 很稳，做项目长期维护时 trellis 更强。中小任务里两者差别不明显，团队和长线任务中差异才会体现。

## 什么任务值得进入工作流

建立 change、确认 spec、完成验收都有成本，判断标准不是改了多少行代码，而是：

1. 这次修改会不会改变项目的**长期行为**；
2. 失败以后的**影响**大不大；
3. 任务是否需要被**其他成员或下一次会话**继续。

一个认证判断、数据迁移或公共 API 的几十行小改动，价值可能很高；一次性实验、可丢弃的调试代码、不影响行为的简单编辑，则没必要为了流程完整专门建 change。

> [!WARNING]
> **个人开发不要过度工程化。** 工程化框架对单人开发经常是额外负担，容易降低开发乐趣和速度。社区讨论中也有用户反馈从 superpowers/comet 退回轻量方案后"乐趣回来了"。是否上工作流框架，取决于它解决的是不是你真实遇到的问题——模型"会"工作流和模型"每次都严格执行"工作流是两回事，状态持久化、阶段边界、验收证据这些事情并不会因为模型变强就自动消失；但如果当前场景 /plan 已够用，就不必为了用而用。

## 相关笔记

- **演进脉络**：[从 OpenSpec、Superpowers 到 TRELLIS 的工程化演进](%E4%BB%8E%20OpenSpec%E3%80%81Superpowers%20%E5%88%B0%20TRELLIS%20%E7%9A%84%E5%B7%A5%E7%A8%8B%E5%8C%96%E6%BC%94%E8%BF%9B.md) — comet 是这条演进线上的下一站：classic 编排 openspec + superpowers，native 则回应了强模型时代的轻约束需求。
- **理论框架**：[Harness Engineering](Harness%20Engineering.md) — comet 是 agent skill harness 的典型实例。
