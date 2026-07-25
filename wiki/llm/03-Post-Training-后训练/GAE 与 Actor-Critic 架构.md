---
date: 2026-04-28
---

![](../_assets/Post-Training-%E5%90%8E%E8%AE%AD%E7%BB%83/GAE%E5%92%8CActor-Critic%E6%9E%B6%E6%9E%84.png)
## 一、先看整体闭环

关联阅读：可结合 [PPO (近端策略优化)](PPO%20%28%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%29.md) 理解；PPO 的优势估计通常依赖 GAE 和 Actor-Critic 框架。


在 PPO 框架下，这三者的协作流程可以先粗看成一条链：

1. **Actor 前向生成**：根据当前状态 $s_t$ 采样动作 $a_t$。
2. **Reward / Critic 提供评估信号**：Reward 给奖励，Critic 给价值预测 $V_t$。
3. **GAE 做后处理**：根据奖励与价值估计，计算 TD Error、优势 $A_t^{GAE}$ 和目标回报 $R_t$。
4. **Actor / Critic 反向更新**：
   - Actor 使用 $A_t^{GAE}$ 更新策略。
   - Critic 使用 $R_t$ 拟合更准确的价值函数。

## 二、GAE 到底在计算什么

### 1. 第一步：计算单步 TD Error

先计算每个时间步的单步时序差分误差：

$$
\delta_t = r_t + \gamma V_{t+1}^{old} - V_t^{old}
$$

其中：

- $r_t$：当前时间步的即时奖励
- $V_t^{old}$：旧 Critic 对当前状态的价值预测
- $V_{t+1}^{old}$：旧 Critic 对下一状态的价值预测
- $\gamma$：折扣因子

#### 直觉理解

- 如果 $\delta_t > 0$，说明这一步的实际后续收益比 Critic 原本估得更好。
- 如果 $\delta_t < 0$，说明 Critic 原来高估了这一步。

### 2. 第二步：把多步 TD Error 平滑成 Advantage

GAE 通过指数衰减把未来多个 TD Error 汇总起来：

$$
A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

这里的关键超参数是 $\lambda \in [0, 1]$。

#### $\lambda$ 的作用

- **$\lambda = 0$**：退化成单步 TD，方差低，但偏差大。
- **$\lambda = 1$**：接近蒙特卡洛估计，偏差低，但方差大。
- **中间取值**：在偏差与方差之间折中，是 GAE 稳定训练的关键。

> [!note]
> GAE 之所以常用，不是因为它“更复杂”，而是因为它给出的优势估计通常比直接用原始回报更稳。

### 3. 第三步：重构目标回报 $R_t$

在 PPO 里，Critic 往往不用原始累积回报直接训练，而是用 GAE 重构出目标回报：

$$
R_t = A_t^{GAE} + V_t^{old}
$$

这个式子的含义是：

- $V_t^{old}$ 提供一个已有的价值基线
- $A_t^{GAE}$ 则告诉你“实际结果相比这个基线偏了多少”
- 两者相加，就得到更平滑、低方差的训练目标 $R_t$

## 三、GAE 算出来的量分别喂给谁

GAE 计算完成后，输出会分别送往 Actor 与 Critic。

### 1. Actor 如何更新：依赖 $A_t^{GAE}$

Actor 的 PPO-clip 损失通常写成：

$$
\mathcal{L}_{Actor}(\theta) = - \mathbb{E} \left[
\min \left(
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t^{GAE},
\operatorname{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) A_t^{GAE}
\right)
\right]
$$

#### 这在训练里意味着什么

- **$A_t^{GAE} > 0$**：这一步比预期好，Actor 应该提高该动作的概率。
- **$A_t^{GAE} < 0$**：这一步比预期差，Actor 应该压低该动作的概率。
- **Clip 机制**：就算某一步特别好或特别差，也不允许一次性更新过猛。

也可以把它理解成：

- **优势值决定方向**
- **ratio 决定当前新旧策略差了多少**
- **clip 决定这次最多允许改多大**

### 2. Critic 如何更新：依赖 $R_t$

Critic 的目标是拟合更准确的价值函数，因此它使用重构后的 $R_t$ 作为监督目标：

$$
\mathcal{L}_{Critic}(\phi) = \mathbb{E} \left[ (V_t^{new} - R_t)^2 \right]
$$

#### 这在训练里意味着什么

- 当前 Critic 输出 $V_t^{new}$。
- GAE 构造出的 $R_t$ 充当“伪标签”。
- 通过最小化均方误差，让 Critic 在下一轮 rollout 中给出更准的价值预测。

> [!important]
> Actor 学的是“动作概率该怎么调”，Critic 学的是“当前状态大概值多少分”。两者都依赖 GAE，但使用的目标不是同一个量。

## 相关笔记

- **前置**：[PPO (近端策略优化)](PPO%20%28%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%29.md) — PPO 的优势估计通常依赖 GAE 和 Actor-Critic 框架。
- **对比**：[GRPO (组相对策略优化)](GRPO%20%28%E7%BB%84%E7%9B%B8%E5%AF%B9%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%29.md) — GRPO 通过组相对优势弱化对独立 Critic 的依赖。
