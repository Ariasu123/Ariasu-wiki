---
date: 2026-04-18
---
![](../_assets/Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%8B%BC%E6%8E%A5%E6%A8%A1%E5%9E%8B%E4%B8%8E%E5%B0%81%E8%A3%85.png)
# MindModel 与 MindCausalLM

关联阅读：可结合 [[10-LLM-大语言模型/01-Architecture-模型架构/MiniMind架构图解析|MiniMind架构图解析]] 理解；封装步骤依赖对 MiniMind 各模块的整体理解。


## 1. 主线

`input_ids -> MindModel -> hidden_states -> lm_head -> logits -> loss / generate`

- `MindModel` 负责做 Transformer 主干计算。
- `MindCausalLM` 负责在主干外面再包一层语言模型头，并对接 Hugging Face 常用接口。

---

## 2. `MindModel` 做了什么

### 2.1 角色定位

`MindModel` 是整个 Transformer 的主干网络，主要负责把输入的 `input_ids` 逐层加工成高层语义表示 `hidden_states`。

### 2.2 核心组成

1. 词嵌入层 `embed_tokens`
   - 把 token id 映射成向量表示。

2. RoPE 位置编码缓存 `freqs_cos`、`freqs_sin`
   - 预先计算好旋转位置编码，在注意力层里直接使用。

3. 多层 `TransformerBlock`
   - 每一层都会继续加工隐藏状态。
   - 每层内部通常包含：
     - 一个自注意力层 `Attention`
     - 一个前馈网络 `FeedForward`
     - 两次 `RmsNorm`
     - 残差连接

4. 最后一层 `RmsNorm`
   - 对所有 Transformer 层输出再做一次归一化。

### 2.3 前向流程

`MindModel` 的前向过程可以拆成下面几步：

1. 接收 `input_ids`
   - 输入通常是 `[bsz, seq_len]` 形状的 token id。

2. 计算当前位置起点 `start_pos`
   - 如果当前没有缓存，`start_pos = 0`。
   - 如果当前有 `past_key_values`，说明前面已经处理过一段序列，这时 `start_pos` 就等于缓存长度。
   - 这一步的作用是：
     - 让当前 token 使用正确的位置编码。
     - 保证增量生成时位置连续，不会从 0 重新开始。

3. 切出当前序列需要的 RoPE 位置编码
   - 模型初始化时已经预先计算好整张 `freqs_cos` 和 `freqs_sin` 表。
   - 前向时只需要按当前区间切一段：
     - `self.freqs_cos[start_pos : start_pos + seq_len]`
     - `self.freqs_sin[start_pos : start_pos + seq_len]`

4. 将 `input_ids` 送入词嵌入层
   - 得到初始的 token 向量表示。

5. 依次通过每一层 `TransformerBlock`
   - 每层都可以接收和返回 `past_key_values`。
   - 这样模型既能训练整段序列，也能支持推理时的 KV Cache。

6. 最后再经过一层 `RmsNorm`
   - 对最终隐藏状态做归一化。

7. 返回结果
   - `hidden_states`
   - `presents`（各层的 KV Cache）
   - `aux_loss`（当前版本里基本为 0，主要是占位）

---

## 3. `MindCausalLM` 做了什么

### 3.1 为什么要再包一层

`MindCausalLM` 的作用不是重复实现 Transformer，而是：

**在 `MindModel` 这个主干外面，再补上语言模型输出头和 Hugging Face 风格接口。**

这样做以后，模型不仅能输出隐藏状态，还能：

- 直接得到词表维度的 `logits`
- 在训练时直接计算语言模型 loss
- 兼容 Hugging Face 常用接口，如：
  - `generate()`
  - `from_pretrained()`

### 3.2 关键结构

1. 继承 Hugging Face 相关基类
   - 继承 `PreTrainedModel` 和 `GenerationMixin`。
   - 这样模型就能更自然地接入 Hugging Face 的训练和生成逻辑。

2. `self.model = MindModel(config)`
   - 主干网络，负责输出 `hidden_states`。

3. `self.lm_head = nn.Linear(hidden_size, vocab_size)`
   - 把隐藏状态投影到词表维度，得到每个位置的 `logits`。

4. 权重绑定
   - `embed_tokens.weight = lm_head.weight`
   - 输入词嵌入和输出层共享权重，可以减少参数量。

### 3.3 前向流程

`MindCausalLM` 的前向过程可以拆成下面几步：

1. 先调用 `self.model(...)`
   - 这一步本质上就是先执行完整的 `MindModel` 前向。
   - 得到：
     - `hidden_states`
     - `presents`
     - `aux_loss`

2. 根据 `logits_to_keep` 决定保留哪些位置的 `hidden_states`
   - 如果 `logits_to_keep = 0`，表示保留全部位置。
   - 如果 `logits_to_keep = 1`，表示只保留最后一个位置。
   - 如果传入的是张量索引，也可以按指定索引选位置。
   - 这一步的意义是：
     - 训练时通常需要所有位置的 `logits`。
     - 推理时通常只关心最后一个 token 的 `logits`。

3. 经过 `lm_head` 投影到词表维度
   - 即执行：`lm_head(hidden_states)`。
   - 输出 `logits`。
   - 常见形状是：
     - `[bsz, seq_len, vocab_size]`
     - 或 `[bsz, 1, vocab_size]`（只保留最后一个位置时）

4. 判断是否需要计算 loss
   - 如果没有传 `labels`，说明当前只是做前向预测，通常用于推理或生成。
   - 如果传了 `labels`，说明当前在训练，需要继续计算 loss。

5. 做自回归语言模型的错位
   - 自回归语言模型的核心是：
     - 位置 `t` 的输入，用来预测位置 `t + 1` 的词。
   - 所以这里要把 `logits` 和 `labels` 错开一位：
     - `shift_logits = logits[..., :-1, :]`
     - `shift_labels = labels[..., 1:]`

6. 展平后计算交叉熵
   - `F.cross_entropy(...)` 通常要求：
     - 预测值形状为 `[N, vocab_size]`
     - 标签形状为 `[N]`
   - 所以这里会先把：
     - `shift_logits` 展平成二维
     - `shift_labels` 展平成一维
   - 同时设置 `ignore_index=-100`，让 padding 位置不参与损失计算。

7. 用 Hugging Face 标准输出结构打包结果
   - 最后会封装成 `CausalLMOutputWithPast`。
   - 一般包含：
     - `loss`
     - `logits`
     - `past_key_values`
     - `hidden_states`
   - 另外还会额外挂上：
     - `output.aux_loss = aux_loss`

8. 返回最终输出
   - 所以 `MindCausalLM` 返回的不是单个张量，而是一个带属性的输出对象。
   - 训练代码里通常会直接访问：
     - `res.loss`
     - `res.logits`
     - `res.past_key_values`
     - `res.aux_loss`

---

## 相关笔记

- **前置**：[[10-LLM-大语言模型/01-Architecture-模型架构/MiniMind架构图解析|MiniMind架构图解析]] — 封装步骤依赖对 MiniMind 各模块的整体理解。
- **流程衔接**：[[10-LLM-大语言模型/02-Pretraining-预训练/PretrainDataset和pretrain和utils方法|PretrainDataset和pretrain和utils方法]] — 模型封装与数据、训练脚本共同构成可复现训练工程。
- **项目应用**：[[50-Projects-项目/02-MiniMind/MiniMind_interview|MiniMind 项目问答]] — MiniMind 项目交付需要兼容 Hugging Face 的模型格式。
