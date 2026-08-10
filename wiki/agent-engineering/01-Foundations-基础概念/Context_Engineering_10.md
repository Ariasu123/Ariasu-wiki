# Context Engineering 面试经典 10 问

关联阅读：可结合 [Agent 核心问题](Agent_16.md) 理解；Context Engineering 是 Agent 可靠运行的核心子系统。工业界分层防御实践见 [从 Prompt 到 Harness：企业级 Agent 工程演进（千问 AI 平台）](../03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/%E4%BB%8E%20Prompt%20%E5%88%B0%20Harness%EF%BC%9A%E4%BC%81%E4%B8%9A%E7%BA%A7%20Agent%20%E5%B7%A5%E7%A8%8B%E6%BC%94%E8%BF%9B%EF%BC%88%E5%8D%83%E9%97%AE%20AI%20%E5%B9%B3%E5%8F%B0%EF%BC%89.md) 的四层上下文防线。


> 适用于 AI Agent、RAG、LLM 应用开发、AI Coding Agent 相关面试。回答重点：Context Engineering 不是单纯写 Prompt，而是管理模型每一轮调用时能看到什么、看到多少、怎么组织、怎么压缩、怎么召回、怎么保证可信和可追溯。

---

## 1. 什么是 Context Engineering？它和 Prompt Engineering 有什么区别？

**标准回答：**

Prompt Engineering 更关注“提示词怎么写”，比如角色设定、任务描述、输出格式、few-shot 示例等。Context Engineering 更关注“模型每一轮调用时应该看到什么信息”，包括系统规则、用户当前任务、历史消息、检索证据、工具结果、长期记忆、任务状态和权限边界。

一句话说：**Prompt Engineering 解决“话怎么说”，Context Engineering 解决“模型该看什么”。**

**可以展开：**

一个复杂 LLM 应用里，真正影响模型表现的不只是 prompt 文案，而是上下文选择和组织。例如当前任务是修 bug，就应该注入 bugfix 相关规则、最近错误日志、相关代码片段，而不是把全部历史、全部文档和所有工具结果都塞进去。

---

## 2. 为什么需要 Context Engineering？

**标准回答：**

因为大模型本身是无状态的，每次调用只能基于输入窗口里的内容推理。如果上下文设计不好，会出现模型忘记目标、旧信息污染、工具结果过长、检索内容不相关、prompt too long、成本和延迟上升等问题。

Context Engineering 的目标是让模型每轮看到：**足够、相关、结构化、不过量、可追溯**的信息。

**典型问题：**

- RAG 中 top-k 检索结果太多，模型被无关证据干扰。
- Agent 多轮调用工具后，日志和历史消息膨胀，模型忘记当前任务。
- 长代码文件或长测试输出直接塞进 prompt，导致 token 爆炸。
- 长期记忆误召回，把旧任务经验错误套用到当前任务。

---

## 3. 一个完整 LLM 应用的上下文通常包含哪些层？

**标准回答：**

常见上下文可以分为几层：

1. **System Context**：系统规则、安全约束、角色定义。
2. **Runtime Context**：当前环境、时间、权限、工作目录、运行状态。
3. **User Context**：用户当前任务、用户偏好、业务约束。
4. **Conversation Context**：最近对话历史。
5. **Retrieved Context**：RAG 检索到的文档、代码、知识片段。
6. **Tool Context**：工具调用结果、日志、报错、执行状态。
7. **Memory Context**：长期记忆、历史经验、用户或项目偏好。

**核心思想：**

不同上下文优先级不同，进入 prompt 的方式也不同。系统规则和当前用户目标优先级最高，长日志、旧工具结果、低相关历史应该被压缩或按需召回。

---

## 4. 为什么不能把所有内容都塞进长上下文模型？

**标准回答：**

长上下文不等于有效上下文。全部塞进去会带来成本上升、延迟增加、注意力稀释、噪声干扰和过期信息污染。模型可能“看到了很多内容”，但不一定能关注到关键内容。

**举例：**

一个 5000 行测试日志里，真正有价值的可能只有失败用例、错误类型、关键 traceback 和 exit code。一个 2000 行代码文件里，当前任务相关的可能只有几个函数。

**总结句：**

大上下文解决的是容量问题，Context Engineering 解决的是信息质量问题。

---

## 5. RAG 中的 Context Engineering 怎么做？

**标准回答：**

RAG 的上下文工程重点是：如何把检索证据变成模型可用的上下文。不是检索 top-k 后直接塞给模型，而是要做 query rewrite、chunking、metadata filter、hybrid search、rerank、去重和 context packing。

**关键步骤：**

- **Query Rewrite**：改写用户问题，提高召回质量。
- **Chunking**：合理分块，避免语义断裂。
- **Metadata Filter**：按时间、权限、来源、类型过滤。
- **Hybrid Search**：向量检索 + 关键词检索。
- **Rerank**：对候选结果重新排序。
- **Context Packing**：控制证据数量、顺序和格式。
- **Citation**：保留来源，保证可追溯。

**面试总结：**

RAG 里上下文工程的目标是让进入 prompt 的证据高相关、低冗余、结构清晰、来源可追溯。

---

## 6. Agent 中的 Context Engineering 怎么做？

**标准回答：**

Agent 的上下文比普通 RAG 更动态，因为它会多轮调用工具。每一轮模型需要知道当前目标、已有计划、最近工具结果、执行失败原因、已完成步骤、下一步候选动作和权限限制。

**Agent 上下文的难点：**

- 工具调用结果会快速膨胀。
- 旧观察结果可能干扰新决策。
- 模型可能重复调用同一个工具。
- 长任务中模型容易忘记最初目标。

**常见设计：**

- 只保留最近关键工具结果。
- 长工具输出摘要化。
- 旧工具结果落盘，保留 raw_path。
- 定期 compact 会话历史。
- 把任务状态结构化，例如目标、已完成、待完成、风险点。

**总结句：**

Agent 的上下文工程本质上是在管理 action-observation 循环产生的大量中间状态。

---

## 7. 上下文压缩怎么设计？压缩会不会丢信息？

**标准回答：**

上下文压缩是把长内容变短，同时保留当前任务需要的关键信息。常见对象包括长对话历史、长工具日志、长代码文件、长网页内容和测试输出。

压缩一定有信息损失风险，所以不能只做不可逆摘要。更稳妥的做法是：**原文落盘，摘要进上下文，保留 raw_path / source_id / chunk_id，需要时再读取原文。**

**常见压缩策略：**

- **规则截断**：保留头部、尾部、错误行。
- **关键词提取**：提取文件路径、函数名、异常类型、失败用例。
- **模型摘要**：生成结构化总结。
- **分层摘要**：先局部摘要，再全局摘要。
- **落盘引用**：原文保存，只把 summary + raw_path 放入上下文。

**面试总结：**

压缩不是简单缩短文本，而是把长内容变成下一步决策需要的结构化信息，同时保留可追溯路径。

---

## 8. Memory 应该怎么设计？它和普通聊天历史有什么区别？

**标准回答：**

普通聊天历史保存的是对话流水账，可能很长、噪声很大。Memory 应该保存跨任务可复用的结构化经验，例如用户偏好、项目规则、历史任务目标、成功步骤、失败原因和工具使用经验。

**Memory 示例：**

```json
{
  "task": "修复登录鉴权失败",
  "context": "FastAPI JWT 项目",
  "steps": ["复现测试", "定位 validate_token", "修复 None 分支", "补充测试"],
  "tools": ["grep_search", "read_file", "edit_file", "pytest"],
  "result": "测试通过",
  "tags": ["bugfix", "auth", "pytest"]
}
```

**召回原则：**

长期记忆不能全部注入 prompt，需要按当前任务相关性召回 top-k，并以摘要形式注入。否则旧经验可能污染当前任务。

**一句话：**

Session 解决“这次任务做到哪了”，Memory 解决“以前类似任务怎么做”。

[ByteRover](../04-References-%E9%A1%B9%E7%9B%AE%E5%8F%82%E8%80%83/ByteRover.md) 是这一思路的项目级实现案例：它把经过策展的知识保存为分层 Context Tree，再用全文检索、生命周期信号和逐级升级策略选择应进入当前任务的 Memory Context。

---

## 9. Tool Result 应该如何进入上下文？为什么要保留 metadata？

**标准回答：**

工具结果不应该作为普通长文本无脑拼进 prompt，而应该结构化成 Tool Observation。尤其是 Shell 日志、测试输出、网页内容和长文件，需要保留关键字段和 metadata。

**示例：**

```json
{
  "tool": "bash",
  "command": "pytest tests/test_auth.py -q",
  "exit_code": 1,
  "status": "failed",
  "summary": "1 个测试失败，错误发生在 invalid token 分支",
  "key_output": "AttributeError: 'NoneType' object has no attribute 'user_id'",
  "raw_path": ".agent/logs/pytest_001.log"
}
```

**metadata 的作用：**

metadata 可以记录来源、时间、文件路径、行号、工具名、退出码、置信度、权限等级和原文路径。它不一定直接给用户看，但决定上下文如何过滤、排序、去重、压缩和回溯。

**总结句：**

工具结果应该作为结构化 Observation 进入上下文，而不是直接拼接 stdout。

---

## 10. 如何评估 Context Engineering 做得好不好？

**标准回答：**

可以从准确性、稳定性、成本、延迟和可追溯性几个维度评估。

**通用指标：**

- 任务完成率。
- 回答准确率。
- 引用正确率。
- prompt too long 次数。
- 平均输入 token。
- 平均延迟和调用成本。
- 无关上下文比例。
- 用户纠错次数。

**RAG 场景：**

- Context Precision。
- Context Recall。
- Faithfulness。
- Answer Relevance。
- MRR / Hit@K。

**Agent 场景：**

- 工具调用成功率。
- 重复工具调用率。
- 失败恢复率。
- 长任务完成率。
- Memory 召回命中率。
- 会话压缩后任务是否还能继续。

**总结句：**

好的 Context Engineering 应该让模型更稳定、更少跑偏、更少超上下文、更低成本，并且答案或行动可以追溯到明确证据。

---

# 60 秒总回答模板

我理解的 Context Engineering 是对大模型输入上下文的系统化治理。它不只是写 Prompt，而是决定模型每轮调用时应该看到哪些信息、以什么结构看到、看到多少、如何排序、如何压缩以及如何召回历史经验。在 RAG 里，它体现在 query rewrite、chunking、retrieval、rerank 和 context packing；在 Agent 里，它体现在任务状态、工具观察、长期记忆、权限边界和会话压缩。好的上下文工程要让模型看到足够但不过量、相关且可追溯的信息，从而降低 token 成本，提高回答准确性和长任务稳定性。

## 相关笔记

- **系统位置**：[Agent 核心问题](Agent_16.md) — Context Engineering 是 Agent 可靠运行的核心子系统。
- **源码实现**：[Claude Code 上下文压缩](../02-Coding-Agents-%E7%BC%96%E7%A8%8B%E6%99%BA%E8%83%BD%E4%BD%93/Claude-Code-Internals-%E6%BA%90%E7%A0%81%E5%8E%9F%E7%90%86/%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86%E4%B8%8E%E4%BA%94%E5%B1%82%E5%8E%8B%E7%BC%A9.md) — Claude Code 五层压缩是上下文预算治理的具体案例。
- **边界**：[Claude Code 长期记忆](../02-Coding-Agents-%E7%BC%96%E7%A8%8B%E6%99%BA%E8%83%BD%E4%BD%93/Claude-Code-Internals-%E6%BA%90%E7%A0%81%E5%8E%9F%E7%90%86/%E9%95%BF%E6%9C%9F%E8%AE%B0%E5%BF%86%E6%9C%BA%E5%88%B6.md) — 上下文处理当前会话，长期记忆负责跨会话保留稳定信息。
- **项目应用**：MiniCode 压缩策略 — MiniCode 用分层压缩解决 token 预算与信息保真问题。
- **面试验证**：一面复盘 — 一面复盘集中追问 Prompt 注入、上下文压缩和异常恢复。
- **工程实现**：[ByteRover](../04-References-%E9%A1%B9%E7%9B%AE%E5%8F%82%E8%80%83/ByteRover.md) — 用分层知识、生命周期评分和渐进检索实现跨任务 Memory Context。
