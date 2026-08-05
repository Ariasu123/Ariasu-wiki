
> 信源：HORIZON：Agentic Hardware Design as Repository-Level Code Evolution，Cunxi Yu 等（NVIDIA Research），2026，arXiv:2606.28279。本页为该论文的网状知识提炼，非全文转载。

**HORIZON** 是一个自进化（self-evolving）agent 框架，把 **RTL 硬件设计当作"仓库级代码演化"（repository-level code evolution）**：一份 Markdown harness 被编译成可执行的"项目包"，一个免手动（hands-free）的 agent 循环在隔离的 git worktree 上演化设计，直到通过可执行的验收条件。在 ChipBench、RTLLM-2.0、Verilog-Eval-v2 和九个 CVDP 类别上，HORIZON 全部达到 100% 完成率——但作者明确强调：**agentic 硬件设计并未被解决**，这些 benchmark 只是更广泛芯片工程问题的受控代理。

---

## 一、核心主张：硬件设计 = 仓库级代码演化

背景：单轮代码生成无法应对可执行设计任务。RTL 的正确性取决于**周期级行为、复位与接口约定、位宽、模拟器反馈**——语法正确的 Verilog 只是起点。已有研究（AlphaEvolve、SATLUTION、ABCEvo）展示了"LLM + 自动评估器 + 版本控制"在算法内核、SAT solver、EDA 软件上的仓库级自进化，但演化对象都是**工程师运行的程序**；HORIZON 把它推进到**工程师设计的产品（硬件本身）**。

关键问题：**硬件设计本身能否被当作仓库级代码演化来管理？**

## 二、框架流程：从 Harness 到可执行任务

```
Markdown harness → bootstrap agent → project pack → 免手动 agent loop 演化 git worktree
```

- **输入**：一份结构化 Markdown harness，含高层意图、仓库上下文、预期产物、评估标准、领域知识（领域感知的 harness 能暴露难以从文件推断的不变量、工具约定和失败模式）。
- **bootstrap agent** 用工具循环 $G_{\phi}$ 把 harness $m$ 编译成 project pack：

$$
p = G_{\phi}(m) = (\pi_{\mathrm{agent}},\; E_{p},\; A_{p},\; \Gamma_{p},\; \Omega_{p})
$$

其中 $\pi_{\mathrm{agent}}$ 是 agent 策略提示与工具契约，$E_{p}$ 是可执行评估器，$A_{p}$ 是接受谓词（acceptance predicate），$\Gamma_{p}$ 是版本控制与产物策略，$\Omega_{p}$ 是领域技能与仓库指令。对 RTL，$E_{p}$ 可包含编译、仿真、覆盖率提取、断言/testbench 检查；对其他领域，同一插槽可换为单元测试、定理证明器、性能分析器、安全扫描器、综合工具或人工评审门。

- **agent loop**：此后完全免手动。每轮循环：生成/编辑候选产物 → 运行评估器 → 打分 → 通过则 commit，否则记录失败日志。

## 三、Git 的三重角色

Git 不是"附带记账"，而是设计的核心：

1. **隔离演化环境**：整个设计问题封装在独立 git worktree 中。
2. **追踪基板**：`git diff` 暴露状态变更、`git commit` 定义被接受的检查点、`git log` 恢复完整轨迹、`git notes` 附加评估证据。
3. **经验缓冲**：成功 commit 成为修复策略的正样本，被拒尝试成为负样本——**仓库历史本身就是经验缓冲，而非独立数据存储**。

## 四、形式化：半马尔可夫决策过程词汇（仅用于记录）

作者借用 SMDP 词汇给 trace 中的对象起精确、可重放的名字（**不做马尔可夫假设，也不训练 RL 策略**）：

- **state** 是仓库的一个版本快照：

$$
s_{t} = (\mathrm{tree}(w_{t}),\; p,\; z_{t},\; \ell_{\le t},\; \mu_{t})
$$

- **option** 是两个检查点之间的一次时间延伸 episode（多次编辑、工具调用与部分修复）。
- **接受谓词**决定轨迹是否前进：

$$
s_{t+1} = \begin{cases} \mathrm{Commit}(w_{t} \oplus \Delta_{t},\; y_{t},\; \Gamma_{p}) & A_{p}(y_{t}) = 1 \\ \mathrm{RejectLog}(s_{t},\; \Delta_{t},\; y_{t}) & A_{p}(y_{t}) = 0 \end{cases}
$$

- **reward** 可为标量或向量：

$$
r_{t} = [\Delta \mathrm{pass},\; \Delta \mathrm{coverage},\; \Delta \mathrm{QoR},\; -\mathrm{tokens},\; -\mathrm{time}]
$$

（本工作报告 $\Delta \mathrm{pass}$、$\Delta \mathrm{coverage}$、$-\mathrm{tokens}$ 三个分量，合成质量 $\Delta \mathrm{QoR}$ 留待未来。）

- **trace** $\tau = \{(s_{t}, a_{t}, r_{t}, s_{t+1}, y_{t})\}_{t=0}^{D-1}$，深度 $D$ 由预算/收敛/停止规则决定，可用于策略分析、奖励建模、课程构建或**离线 agent-RL 训练**——但本工作的 agent backbone 全程固定，不做在线训练。

## 五、Agent Loop 与 Trace Buffer

- 外层每个转移包含一个长度 $K_{t}$ 的内部轨迹：agent 读当前状态、规划目标、编辑 worktree、调用工具、解释失败、修复或提交。内部轨迹**不假设马尔可夫**，每步长度可变。
- trace buffer 完全基于原生 git：staged 编辑用 `git diff --cached` 检查；每个被接受尝试成为一个 commit，其消息与 `git notes` 携带评估结论与 reward；完整版本历史用 `git log` 恢复；提交前还有**独立 review 步骤**对候选做 diff 审查。

## 六、记忆与成本：持久 Session + Prompt Cache

没有真正马尔可夫状态，记忆按务实方式处理：**最大化缓存 token 占比**。HORIZON 跨迭代复用持久模型 session，让 harness、project pack、稳定源码和累积调试上下文从 provider 的 prompt cache 提供服务，新计费 token 主要来自当前 diff、最新评估输出和 agent 响应。这让多轮修复的边际成本很低——实验中 **约 91% 的 token 是缓存输入**。

## 七、实验结果

### 7.1 Benchmark 完成率(Table 1)

| Suite/类别 | 评估焦点 | 首轮通过率 | 收敛迭代 | HORIZON 最终 |
|------------|----------|-----------|---------|--------------|
| ChipBench | 混合 RTL 生成 | 20.0% | 5 | 100% |
| RTLLM-2.0 | 自然语言规格→RTL | 78.0% | 2 | 100% |
| Verilog-Eval-v2 | HDLBits 风格生成 | 86.2% | 2 | 100% |
| CVDP CID 002 | RTL 补全 | 3.2% | 82 | 100% |
| CVDP CID 003 | 规格→RTL | 19.2% | 24 | 100% |
| CVDP CID 004 | RTL 修改 | 10.9% | 36 | 100% |
| CVDP CID 005 | 模块复用 | 9.1% | 14 | 100% |
| CVDP CID 007 | Lint/QoR 改进 | 0.0% | 24 | 100% |
| CVDP CID 012 | 测试计划→激励生成 | 47.8% | 32 | 100% |
| CVDP CID 013 | 测试计划→checker 生成 | 3.8% | 19 | 100% |
| CVDP CID 014 | 测试计划→断言生成 | 79.1% | 1 | 100% |
| CVDP CID 016 | 调试与修 bug | 25.7% | 13 | 100% |
| **总体** | — | **47.8%** | — | **100%** |

- 全部 12 个套件/类别由**单一免手动 agent 循环**驱动到 100%(唯一未通过项是 ChipBench 一个规格-harness 缺陷,属 benchmark 自身问题)。
- 首轮整体通过率 47.8%,最难的 CVDP 类别低至 3.2%(CID 002 补全)与 3.8%(CID 013 checker 生成),最终都收敛到 100%——**差异不在终点,而在路径**。
- 收敛路径差异大:简单的 RTL 生成套件几轮内饱和(Verilog-Eval 2 轮),CVDP 暴露更长的修复轨迹(CID 002 需 82 轮)。

### 7.2 Token 消耗(Table 2)

- 到最早最优迭代累计 209.9M tokens;三个传统套件合计仅 6.0M,九个 CVDP 类别占 **97.1%**(203.9M)。
- 消耗集中点:CID 002(56.0M)、CID 003(38.0M)、CID 012(32.2M)。
- 有趣的对照:CID 013 首轮通过率最低,却只消耗 14.2M tokens 就收敛(轨迹近线性、几乎无平台期);CID 002 才是成本主要驱动——**"少数顽固设计的长尾"决定成本**。
- 实践结论:**benchmark 完成率必须与 token 消耗一起报告**——最难类别最后几个百分点吞噬不成比例的预算,token 效率比最终通过率更值得改进。

### 7.3 测试生成任务的覆盖率(Table 3)

- CID 012(激励生成)平均覆盖率随迭代从 86.5% 升至 97.9%,与通过率 47.8%→100% 同步。
- 关键:**HORIZON 的接受门是 CVDP 的 pass 条件,不是覆盖率目标**——一旦通过,循环停止打磨。覆盖率是**次要的观察信号**(生成的测试有多实质),而非被优化的目标;作者没有尝试把覆盖率推到 100%。

## 八、局限与开放问题

### 8.1 Reward 反馈接口 → over-solving / reward hacking

agent 能查看每轮评估输出(模拟器消息、评估日志、失败 trace),这模拟了真实调试工作流,但也带来风险:**agent 可能定制 RTL 去匹配观察到的失败、确定性测试或评估器怪癖**,而不是稳健实现设计语义。通过的结果可能只是"满足可见 harness",而非"在所有合理测试下满足规格"。

SWE-bench 的做法是**推理期隐藏 fail-to-pass/pass-to-pass 测试**,最终评分时才执行;后续研究还发现 benchmark 设计(解泄露、弱测试套件)会显著虚增成绩。据此建议未来 RTL agent benchmark:
- 分离**诊断反馈**与**最终评分**;
- 引入隐藏随机测试、独立参考模型、形式等价检查、留出模拟器配置;
- 报告对 harness 扰动的鲁棒性、覆盖率闭合与 agent 消费的反馈 trace。

### 8.2 反馈延迟(long-turnaround reward)

本论文的 pass/fail benchmark 评估较快,迭代修复可行;但真实芯片设计自进化中 reward 可能极慢——SATLUTION 评估整个 SAT 竞赛 benchmark 约需 2 小时(800 节点并行),PPA 优化或 EDA 工具自进化的评估延迟可到**数天甚至数周**(综合、布局布线、时序、功耗、大型回归)。延迟奖励从根本上改变问题:朴素"编辑-评估-修复"循环太慢,agent 必须学会在**延迟、稀疏、昂贵的反馈**下推理。这是 agentic 芯片设计的关键研究挑战。

## 九、与 Agent 工程的关联

HORIZON 本质是一个 agent 系统,与 wiki 中 Agent 工程主题直接呼应:

- **Harness Engineering**:Markdown harness → project pack(评估器 + 接受谓词 + 策略)正是 Harness 的"评估与观测/约束校验"层;"独立 review 步骤"呼应"生产验收分离"原则;git worktree 演化是"状态外化到 Git"的极致形式。详见 [Harness Engineering](../agent-engineering/03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Harness%20Engineering.md)。
- **Loop Engineering**:"编辑 → 评估 → commit/reject → 下一轮"的免手动循环、worktree 隔离、仓库历史作持久化记忆,正是 Loop Engineering 的具体实现。详见 [Loop Engineering](../agent-engineering/03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Loop%20Engineering.md)。
- **仓库级自进化谱系**:HORIZON 是 AlphaEvolve → SATLUTION → ABCEvo 谱系的硬件侧延伸,概念详见 [仓库级代码演化](%E4%BB%93%E5%BA%93%E7%BA%A7%E4%BB%A3%E7%A0%81%E6%BC%94%E5%8C%96.md)。

## 相关笔记

- **核心概念**：[仓库级代码演化](%E4%BB%93%E5%BA%93%E7%BA%A7%E4%BB%A3%E7%A0%81%E6%BC%94%E5%8C%96.md) — 自进化 agent 的演化谱系、接受门控原则、Git 即 agent substrate。
- **Agent 侧**：[Harness Engineering](../agent-engineering/03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Harness%20Engineering.md) — harness/评估器/验收分离的工程框架。
- **Agent 侧**：[Loop Engineering](../agent-engineering/03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Loop%20Engineering.md) — 免手动循环、worktree 与持久化记忆的抽象。
