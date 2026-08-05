![](../_assets/Post-Training-%E5%90%8E%E8%AE%AD%E7%BB%83/GRPO%E6%B5%81%E7%A8%8B%E5%9B%BE.png)
## 一、 为什么需要 GRPO？（核心动机）

关联阅读：可结合 [SFT 监督微调](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/SFT%20%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) 理解；GRPO 训练依赖已经具备基础输出能力的模型。


在传统的 RLHF-PPO 算法中，为了训练一个策略模型（Actor），系统必须在显存中同时维护 4 个庞大的模型（Actor、Critic、Reward、Reference）。

- **PPO 的痛点：** 其中 Critic Model（评估模型）负责预测每个状态的期望收益（Value），它的参数量通常与策略模型一样大，这导致了极其可怕的计算和存储消耗 。
    
- **GRPO 的破局：** GRPO 彻底去掉了 Critic 模型 。它不再依赖一个庞大的神经网络去“预测”基准线，而是通过“组内横向比较”来直接计算优势（Advantage），从而极大地释放了显存，让普通算力集群也能训练超大参数模型。
## 二、 GRPO 的核心运行机制 (Step-by-Step)

GRPO 将 PPO 中的“和自我历史预测对比”改成了“和同侪（分身）对比”。

### Step 1: 组采样 (Group Rollout)

对于同一个用户问题 $q$，GRPO 从旧策略模型 $\pi_{\theta_{old}}$ 中采样出一系列的输出（即一个组，大小设为 $G$），记作 $\{o_1, o_2, \dots, o_G\}$ 。

- （注：原论文中通常设置 $G=64$，通过生成大量的不同回答，极大地增加了模型探索到正确答案的概率 ）。

### Step 2: 组内相对优势计算 (Group Relative Advantage)

将这 $G$ 个输出送入奖励模型（Reward Model）或规则引擎中进行评分，得到一组对应的奖励值 $\{r_1, r_2, \dots, r_G\}$ 。 随后，**GRPO 直接使用统计学中的 Z-Score 标准化来计算优势**：

$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}$$

- **直觉理解：** 这道题的“预期得分”就是这 $G$ 个分身的**平均分**。如果你的得分高于平均分（$A_i > 0$），你的生成概率就会被强化；低于平均分（$A_i < 0$）就会被抑制。

### Step 3: Token 级别的优势分配 (ORM vs PRM)

算出了整句话的优势 $A_i$ 后，如何分配给句子里的每一个 Token？

- **ORM (结果监督)：** 最基础的做法。将这个整体的优势值，无差别地赋值给该输出 $o_i$ 中的每一个 Token 。
    
- **PRM (过程监督)：** 为每个推理步骤（Token）分别计算奖励。当前 Token 的优势等于当前步骤的优势加上后续步骤优势的期望之和 。
## 三、 GRPO 的目标函数

GRPO 的最终优化目标是最大化以下函数：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_{\theta} \parallel \pi_{ref}) \right) \right]$$

这里面包含了强化学习稳健训练的两大护法：

### 1. Clip 截断机制 (PPO 的遗产)

- 公式中的 $\frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$ 表示新旧模型输出概率的比值 。
    
- `clip` 函数强行将其限制在 $[1-\epsilon, 1+\epsilon]$ 之间，用于限制策略的更新速度，避免原始模型能力的灾难性丧失 。
    

### 2. 精确的 KL 散度惩罚 (K3 估计量)

为了强制新策略不偏离参考策略 $\pi_{ref}$（防止输出乱码或触发安全风险），GRPO 在 Token 级别引入了 KL 散度惩罚。 它使用的是无偏且方差极低的 **K3 估计量**（公式展开如下）：

$$\mathbb{D}_{KL}(\pi_{\theta} \parallel \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1$$

- （数学推导：设 $r = \frac{\pi_{ref}}{\pi_{\theta}}$，则上式为 $r - \log r - 1$。它不仅是 KL 散度的严格无偏估计，而且通过数学特性保证了该惩罚项恒大于 0，使得训练极度平稳 。）
    

## 四、 优缺点与工程权衡 (Trade-offs)

**优势：**

- **极致的显存释放：** 彻底去掉了 Critic 模型，使得硬件门槛大幅降低，成为普通算力集群微调大模型的首选 。
    
- **实现逻辑简化：** 摆脱了 GAE（广义优势估计）和 Value Loss 的复杂计算，仅需关注 Reward 分数和组内统计即可。
    

**潜在挑战（去掉 Critic 的代价）：**

- **奖励估计不准确：** Critic 的强项在于预估“需要长期规划”的任务价值。失去 Critic 后，如果奖励只有在极长的长期目标完成后才出现，GRPO 的优势评估可能会失准 。
    
- **熵崩溃 (Entropy Collapse)：** 模型为了追求高分，可能会迅速收敛到某一种固定的输出模式，导致行为过于确定化，失去探索新答案的能力 。
    
- **奖励噪音敏感：** 由于完全依赖当前批次的横向对比，如果奖励信号存在随机波动，可能会导致模型接收到不稳定的梯度，进而影响训练稳定性 。

## 相关笔记

- **前置**：[SFT 监督微调](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/SFT%20%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) — GRPO 训练依赖已经具备基础输出能力的模型。
- **对比**：[DPO (直接偏好优化)](DPO%20%28%E7%9B%B4%E6%8E%A5%E5%81%8F%E5%A5%BD%E4%BC%98%E5%8C%96%29.md) — GRPO 的奖励驱动训练与 DPO 的偏好损失形成两条路线。
- **对比**：[GAE 与 Actor-Critic 架构](GAE%20%E4%B8%8E%20Actor-Critic%20%E6%9E%B6%E6%9E%84.md) — GAE 与 Actor-Critic 是理解 GRPO 省略价值模型的基线。
- **前置**：[PPO (近端策略优化)](PPO%20%28%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%29.md) — PPO 的裁剪目标是理解 GRPO 目标函数的基础。
- **复习**：[LLM 综合复习](../05-Review-%E5%A4%8D%E4%B9%A0%E6%80%BB%E7%BB%93/LLM_22.md) — 综合题库可检验 GRPO 的组采样、相对优势和规则奖励。
