# Agent Skill 机制整理

关联阅读：可结合 [Agent 核心问题](Agent_16.md) 理解；Skill 是 Agent 能力组织与渐进加载机制。


## 1. Skill 是什么

Skill 可以理解为 Agent 的一组“专业能力包”。

它不是单纯的一个函数，也不是单纯的一段 Prompt，而是把某类任务所需的说明文档、触发条件、执行流程、脚本、资源文件、权限约束和输入输出规范打包在一起，让 Agent 在遇到相关任务时按需加载并执行。

一句话总结：

> Skill 是 Agent 的可复用能力插件，用来告诉 Agent 面对某类任务时应该怎么做、用哪些工具、遵守哪些规则。

## 2. 核心区别

| 概念   | 作用                                              |
|--------|---------------------------------------------------|
| Tool   | Agent 的原子动作，比如读文件、写文件、执行 Python |
| Skill  | 围绕某类任务封装的能力包                          |
| Script | Skill 内部的确定性执行脚本                        |
| MCP    | 标准化、安全、可控地暴露工具给 Agent 的协议层     |

可以理解为：

    Tool = 一个动作
    Skill = 一套解决方案
    Script = Skill里的具体实现
    MCP = Tool的标准化接入协议

## 3. Skill 的三层结构（Metadata、Instruction、Resource）

现代 Agent Framework 中的 Skill 通常采用渐进式加载（Progressive Disclosure）机制。

原因很简单：

如果把所有 Skill 的全部内容一次性塞进上下文：

    100个Skill
    每个Skill几千Token

上下文会迅速爆炸。

因此 Skill 通常被拆成三层：

    Metadata
    Instruction
    Resource

### 第一层：Metadata（元数据层）

Metadata 是最轻量的一层。

Metadata 通常包含：

- **Skill 名称**
- **描述**
- 关键词
- 版本信息

特点：

    体积小
    始终加载
    用于路由
    不包含复杂执行逻辑

可以理解为：

> Metadata 相当于 Skill 的“名片”。

### 第二层：Instruction（指令层）

Instruction 是 Skill 的核心说明书。

通常放在：

    SKILL.md

里面描述：

    什么时候使用这个Skill
    如何执行任务
    执行步骤是什么
    有哪些注意事项
    输出格式是什么

特点：

    只有命中Skill后才加载
    体积中等
    主要给LLM阅读
    指导推理和规划

可以理解为：

> Instruction 相当于 Skill 的“操作手册”。

### 第三层：Resource（资源层）

Resource 是真正执行任务时需要的内容。

    skill/
    ├── SKILL.md
    ├── references/
    ├── scripts/
    └── assets/

特点：

    默认不加载
    执行时按需读取
    体积最大

可以理解为：

> Resource 相当于 Skill 的“工具箱”。

## 4. Agent 怎么选择 Skill

Agent 通常先进行 Skill Routing（技能路由）。

流程如下：

    用户请求
    ↓
    读取所有Skill Metadata
    ↓
    匹配候选Skill
    ↓
    选择最相关Skill
    ↓
    加载Instruction
    ↓
    执行任务

### Skill 选择的常见方式

#### 方式1：关键词匹配

路由使用关键词匹配：

- 关键词是 query 子串：`+4`。
- 关键词命中分词结果：`+2`。
- 名称或描述 token 命中：`+1`。
- 最后取 Top 3。

中文处理采用单字和连续双字符切分，不是 Embedding 语义检索。

    用户问题
    ↓
    匹配tags和keywords
    ↓
    选择Skill

#### 方式2：Embedding 检索

流程：

    用户请求
    ↓
    Embedding
    ↓
    Skill Metadata Embedding
    ↓
    向量检索
    ↓
    Top-K Skill

#### 方式3：LLM Router

高级 Agent 常用。

流程：

    用户请求
    ↓
    读取Skill列表
    ↓
    LLM判断
    ↓
    返回候选Skill

## 5. Agent 怎么加载 Skill 中的资源

Skill 不会一次性把所有资源读进上下文。

否则：

    几十个PDF
    几百个模板
    大量脚本

上下文会直接爆掉。

## 按需加载

### 第一阶段：只加载 Metadata

    用户请求
    ↓
    Metadata

此时只知道：

    有哪些Skill
    Skill做什么

### 第二阶段：加载 Instruction（按需）

Skill 被选中后：

    加载 SKILL.md

Agent 开始理解：

    执行步骤
    规则
    输出格式

### 第三阶段：按需加载 Resource（按需中的按需）

执行过程中：

    Planner发现需要参考文档
    ↓
    读取references（读）

或者：

    需要执行脚本
    ↓
    调用scripts（只执行不读）

或者：

    需要模板
    ↓
    读取assets

整个过程是动态的。

## 6. Manifest 是什么

Manifest 是 Skill 的结构化配置清单。

通常是：

    manifest.json
    manifest.yaml
    skill.yaml

主要给系统使用。

Manifest 负责：

- Skill 注册
- Skill 发现
- 资源声明
- 权限声明
- 入口配置
- 输入输出 Schema

简单理解：

    Manifest = 给系统看的
    SKILL.md = 给模型看的

## 7. Assets 是什么

Assets 是 Skill 中的静态资源。

例如：

    assets/
    ├── report_template.md
    ├── logo.png
    └── style.css

通常不会直接进入模型上下文。

而是：

    脚本读取
    工具读取
    运行时引用

## 8. Script 和 MCP 的区别

Skill Script 是某个 Skill 内部的具体执行逻辑。

MCP（Model Context Protocol）是一套标准化工具接入协议。

核心区别：

> Script 是能力实现；MCP 是能力暴露和调用的标准化边界。

例如：

### Script 方案

    Agent
     ↓
    Skill
     ↓
    Python Script
     ↓
    数据库

### MCP 方案

    Agent
     ↓
    MCP Client
     ↓
    MCP Server
     ↓
    Database Tool

MCP 提供：

- Tool Discovery
- 参数 Schema
- **权限控制**
- **日志审计**
- **错误处理**
- **多 Agent 复用**

安全性和稳定性不如MCP

## 9. 什么时候用 Script，什么时候用 MCP

### 适合直接写 Script

- 小任务
- 单个 Skill 使用
- 本地文件处理
- 快速验证
- 不需要**复杂权限控制**

### 适合 MCP

- 多个 Agent 共用
- 多个 Skill 共用
- 数据库访问
- GitHub 操作
- 浏览器自动化
- 企业知识库
- 外部 API

## 10. 面试总结

可以这样回答：

> Skill 是 Agent 的能力封装单元，通常采用 Metadata、Instruction、Resource 三层结构。Metadata 用于 Skill 路由和发现；Instruction 用于指导模型如何完成任务；Resource 用于提供执行阶段需要的脚本、模板和参考资料。

进一步说明：

> Agent 不会一次性加载所有 Skill 内容，而是先读取 Metadata，通过关键词匹配、Embedding 检索或 LLM Router 选择候选 Skill；命中后再加载 SKILL.md 等 Instruction；执行过程中根据需要动态读取 references、scripts 和 assets 等 Resource。

最后总结：

> Skill 解决的是“什么时候做、怎么编排”；Script 解决的是“具体怎么执行”；MCP 解决的是“如何标准化、安全、可复用地调用工具”。小任务适合直接写 Script，通用工具和外部系统能力更适合封装成 MCP。
![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/Pasted%20image%2020260621222658.png)

## 相关笔记

- **系统位置**：[Agent 核心问题](Agent_16.md) — Skill 是 Agent 能力组织与渐进加载机制。
- **边界**：[Tools、MCP 与 Function Calling](Tools_16%EF%BC%88MCP%E3%80%81Function%20Calling%E4%B8%8E%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE%EF%BC%89.md) — Skill 负责能力说明与流程，Tool 负责实际执行接口。
- **项目应用**：MiniCode 工具系统 — MiniCode 将 Skill 路由和 Tool／MCP 执行组合成能力系统。
- **案例**：[Hermes Agent 项目功能与通用使用场景分析](../04-References-%E9%A1%B9%E7%9B%AE%E5%8F%82%E8%80%83/Hermes%20Agent%20%E9%A1%B9%E7%9B%AE%E5%8A%9F%E8%83%BD%E4%B8%8E%E9%80%9A%E7%94%A8%E4%BD%BF%E7%94%A8%E5%9C%BA%E6%99%AF%E5%88%86%E6%9E%90.md) — Hermes 用 Skills、MCP 和插件扩展通用 Agent 能力。
- **概念落点**：业务会议 — 业务流程展示 Skill 如何承载行业规则。
