---
date: 2026-04-28
---
![](../_assets/Post-Training-%E5%90%8E%E8%AE%AD%E7%BB%83/DPO%E6%B5%81%E7%A8%8B%E5%9B%BE.png)
# DPO（直接偏好优化）

关联阅读：可结合 [LoRA（Low-Rank Adaptation，低秩微调）](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/LoRA%EF%BC%88Low-Rank%20Adaptation%EF%BC%8C%E4%BD%8E%E7%A7%A9%E5%BE%AE%E8%B0%83%EF%BC%89.md) 理解；DPO 的策略模型可采用 LoRA 进行参数高效更新。


> [!abstract]
> DPO 的核心思路是：**不再训练 Reward Model，也不做在线强化学习 rollout，而是直接利用“当前策略相对参考模型更偏好哪条回答”来学习人类偏好。**
>
> 它把 RLHF 中原本复杂的奖励建模问题，改写成一个更轻量的离线偏好优化问题。

## 一、DPO 为什么会出现

在后训练的偏好对齐阶段，PPO 与 GRPO 都属于“显式奖励建模”的强化学习范式：

- 先定义或学习奖励信号
- 再让策略模型朝高奖励方向更新

DPO 提出的范式转换是：**彻底绕开 RL 流程，把偏好对齐问题直接变成监督式偏好学习。**

### 1. RLHF 的两个工程瓶颈

#### 显存墙（Memory Wall）

- PPO 通常需要同时维护 4 个大模型。
- GRPO 虽然去掉了 Critic，但往往仍需加载 2~3 个模型。

#### 吞吐量瓶颈（Throughput Bottleneck）

- RL 训练要求策略模型在训练循环中做 **在线自回归生成（online rollout）**。
- 自回归解码依赖频繁的 KV-Cache 读写，速度远慢于普通前向训练。
- 结果就是：训练过程大量时间花在“生成 token”上，而不是高效矩阵计算上。

### 2. DPO 的工程破局点

DPO 的优势在于：

- **纯离线训练**：不需要在线生成
- **去掉独立 Reward Model**：不再单独训练裁判
- **去掉 Critic**：不再额外学习价值函数
- **训练形态接近 SFT**：整体开销接近标准监督微调

> [!tip]
> 可以把 DPO 理解成：保留“偏好对齐”的目标，但尽量复用 SFT 的训练形态和工程效率。

## 二、DPO 的核心算法逻辑

传统 RLHF 的思路是：

1. 先训练一个 Reward Model 给回答打分
2. 再让 Policy 模型去迎合高分回答

DPO 的关键洞察是：**语言模型自身的相对生成概率，就可以用来表达偏好强弱。**

### 1. 从显式奖励到隐式奖励

对于同一个回答 $y$，如果当前 Policy 模型比 Reference 模型更愿意生成它，那么就说明当前模型更“偏好”这个回答。

可以把它记成一个隐式偏好分数：

$$
s_\theta(x, y) = \log \pi_\theta(y|x) - \log \pi_{ref}(y|x)
$$

这个分数越大，表示：

- 当前训练中的策略模型更偏向这条回答
- 相比参考模型，它更愿意把概率质量分配给这条回答

也正因为如此，**DPO 不再需要一个独立的 Reward Model，语言模型自己就能充当“相对打分器”。**

### 2. 从“打分”变成“比较”

DPO 的训练数据不是单条答案，而是成对偏好样本：

- `chosen / winner`：人类更偏好的回答 $y_w$
- `rejected / loser`：人类不偏好的回答 $y_l$

于是训练目标就变成一句话：

> 在相同 Prompt $x$ 下，让模型对 $y_w$ 的偏好强于对 $y_l$ 的偏好。

## 三、DPO 损失函数怎么理解

DPO 的核心损失函数是：

$$
\mathcal{L}_{DPO} = - \mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

### 1. 先看括号里的核心量

括号里的内容本质上是在计算一个 **偏好差值（preference margin）**：

$$
\bigl(\text{当前模型对好回答的相对偏好}\bigr) - \bigl(\text{当前模型对坏回答的相对偏好}\bigr)
$$

如果这个差值越大，说明：

- 当前模型更明显地偏向人类喜欢的回答
- 当前模型更符合偏好数据的排序关系

### 2. $\beta$ 在控制什么

- $\beta$ 是温度超参数，常见取值大约在 `0.1` 左右。
- 它控制策略偏离参考模型的力度。
- $\beta$ 越大，模型越积极地拉开 chosen 和 rejected 的差距；$\beta$ 越小，更新会更保守。

### 3. 为什么它像一个二分类损失

`log sigmoid` 的结构，本质上就是在做一个“chosen 应该胜过 rejected”的二元判别：

- 如果模型已经明显偏向 `chosen`，Loss 会变小
- 如果模型还在偏向 `rejected`，Loss 会变大

从梯度的角度看，它在做三件事：

- **Push up winner**：提高好回答 $y_w$ 的概率
- **Push down loser**：压低坏回答 $y_l$ 的概率
- **错得越离谱，罚得越重**：偏好排序错得越严重，梯度惩罚越大

## 四、工业级训练流水线

从工程实现上看，DPO 的 Forward / Backward 逻辑很精简。

### 1. 离线数据准备

构造偏好三元组数据：

- `Prompt`
- `Winner_Response`
- `Loser_Response`

也就是典型的：

```text
(x, y_w, y_l)
```

### 2. 前向传播：获取 log-probs

对同一批次样本：

- 用 **冻结的 Reference 模型** 计算 $y_w$ 与 $y_l$ 的基准对数概率
- 用 **训练中的 Policy 模型** 计算 $y_w$ 与 $y_l$ 的当前对数概率

### 3. 计算损失并反向传播

- 把这些对数概率代入 $\mathcal{L}_{DPO}$
- 只更新 Policy 模型参数
- Reference 模型始终冻结

### 4. 为什么它比 PPO 更省算力

DPO 最重要的工程优势是：

- **没有 `model.generate()` 在线 rollout**
- **没有逐 token 等待生成**
- **没有 Reward / Critic 的额外训练链路**

因此，大部分算力都花在高效的稠密矩阵乘法上，而不是耗时的自回归采样上。

## 五、minimind 中的具体实现

这一部分重点不是“DPO 理论是什么”，而是“代码里怎么准确拿到要优化的回答概率”。

### 1. 成对数据的结构化渲染（Paired Rendering）

DPO 训练的前提是拥有成对偏好数据。

- 每条样本必须包含 `chosen` 和 `rejected` 两个对话列表。
- 使用 `apply_chat_template` 把它们渲染成模型可读取的字符串。
- **关键要求**：`chosen` 和 `rejected` 必须共享完全相同的 Prompt，只允许 Assistant 回复部分不同。

### 2. 精准定位 Assistant 回复区间（Response Anchor）

DPO 优化的是 $\pi(y|x)$，也就是“在给定 Prompt 下生成回答”的概率。

因此，必须明确：

- 哪些 token 属于 Prompt
- 哪些 token 属于 Assistant 的回答

常见做法是：

- 预先 tokenize 角色标识符，例如 `bos_token + "assistant\n"` 与 `eos_token`
- 再通过扫描定位回答区间
- 只有这个区间内的 token 才应该计入回答概率

### 3. 构造 Loss Mask

这是 DPO 实现里最关键的工程细节之一。

构造一个与输入序列等长的 `0/1 mask`：

- **Prompt 部分**：mask = 0
- **Assistant 回复部分**：mask = 1
- **Padding 部分**：mask = 0

#### 为什么必须这么做

因为 DPO 要优化的是“回答内容的偏好”，而不是 Prompt 模板本身。

所以：

- Prompt 的概率不能计入 Loss
- Padding 当然也不能计入 Loss
- 只有回答部分才是真正需要奖惩的对象

### 4. 自回归错位对齐（Shift-Right Alignment）

由于大模型做的是“预测下一个 token”，所以输入与目标必须错位：

- **输入**：序列的 `[:-1]`
- **目标**：序列的 `[1:]`
- **Mask**：也对齐到 `[1:]`

这样才能保证：

- 当前 token 的预测，去对齐下一个真实 token
- Loss mask 与真实预测目标处于同一位置

### 5. 批处理一致性（Batch Consistency）

为了做高效批训练，还需要统一处理不同长度的 `chosen` / `rejected`：

- **Padding 到 `max_length`**
- 构造 `attention_mask`
- 避免模型把 padding token 当成有效上下文参与计算

### 6. Mask 是怎么直接参与“整句概率”计算的


DPO 获取一句回答总对数概率的流程通常是：

1. 模型输出所有位置的 logits
2. 提取每个真实目标 token 对应的 log-prob
3. 将这些 log-prob 与 mask 逐元素相乘
4. 最后把结果求和

可以写成：

$$
\log p(y|x) = \sum_i \bigl(\text{LogProb}_i \times \text{Mask}_i\bigr)
$$

这意味着：

- **Prompt 部分（mask = 0）**：对应项被清零，不计入总和
- **回答部分（mask = 1）**：对应项被保留，计入总和
- **Padding 部分（mask = 0）**：同样被忽略

> [!important]
> 这一步决定了你最终拿到的到底是“整段对话的概率”，还是“仅回答部分的概率”。DPO 需要的是后者。

## 六、一页记住 DPO

- DPO 的目标不是做强化学习 rollout，而是**直接学习人类偏好排序**。
- 它不显式训练 Reward Model，而是用 `Policy 相对 Reference 的对数概率差` 充当隐式偏好分数。
- 面对 `(Prompt, chosen, rejected)`，DPO 的核心目标是：**让 chosen 的相对偏好分数高于 rejected**。
- 它的训练形态接近 SFT，因此工程成本明显低于 PPO / GRPO。
- 在具体实现里，最关键的不是公式本身，而是：**只统计 Assistant 回复部分的 log-prob，并正确构造 mask。**

## 相关笔记

- **工程选择**：[LoRA（Low-Rank Adaptation，低秩微调）](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/LoRA%EF%BC%88Low-Rank%20Adaptation%EF%BC%8C%E4%BD%8E%E7%A7%A9%E5%BE%AE%E8%B0%83%EF%BC%89.md) — DPO 的策略模型可采用 LoRA 进行参数高效更新。
- **前置**：[SFT 监督微调](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/SFT%20%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) — DPO 需要一个已有指令能力的参考起点。
- **对比**：[PPO (近端策略优化)](PPO%20%28%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%29.md) — DPO 直接优化偏好对，而 PPO 通过奖励和策略更新训练。
- **对比**：[GRPO (组相对策略优化)](GRPO%20%28%E7%BB%84%E7%9B%B8%E5%AF%B9%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%29.md) — DPO 使用成对偏好，GRPO 使用组内相对奖励进行策略优化。
- **复习**：[LLM 综合复习](../05-Review-%E5%A4%8D%E4%B9%A0%E6%80%BB%E7%BB%93/LLM_22.md) — 综合题库可用于检验 DPO 的损失、参考模型和适用场景。
