---
title: Hermes Agent 项目功能与通用使用场景分析
source: https://www.youtube.com/watch?v=IHan9Pje_z4
author:
  - "henrylin的量化策略工坊"
published: 2026-04-09
created: 2026-05-24
tags:
  - clippings
---
关联阅读：可结合 [[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Skill_10|Skill_10]] 理解；Hermes 平台是 Skill 生态化使用的实例。

![](https://www.youtube.com/watch?v=IHan9Pje_z4)

#hermes #hermesagent  
https://dev.to/henry\_lin\_3ac6363747f45b4/hermes-agent-xiang-mu-gong-neng-yu-tong-yong-shi-yong-chang-jing-fen-xi-2bi5  
1\. 项目定位  
Hermes Agent 是一个通用型 AI Agent 平台，不是单纯的聊天界面，也不是单一的代码助手。它把大模型推理、工具调用、终端执行、文件操作、网页检索、浏览器自动化、长期记忆、定时任务、多平台消息接入和外部系统扩展整合到一个统一框架里。  
  
从仓库结构和主干代码来看，它的目标不是“回答问题”本身，而是让模型具备持续执行任务的能力，并且能在不同入口、不同环境和不同工具集之间稳定运行。  
  
可以把它理解为三层系统：  
  
Agent 执行层：负责会话循环、模型调用、工具调度、上下文压缩。  
平台能力层：负责 CLI、消息网关、状态存储、调度、配置、权限边界。  
扩展生态层：负责 skills、MCP、插件、自定义工具和多环境运行。  
所以，Hermes Agent 更接近一个“可运行的 Agent Operating Layer”，而不是一个普通的 LLM 应用。

相比之下，[[30-Agent-Engineering-Agent工程/04-References-项目参考/AgentScope|AgentScope 2.0]] 更偏向 Python Agent 框架和多租户服务骨架：Hermes 面向开箱运行的个人自动化，AgentScope 面向开发者构建自己的 Agent 产品。

## 相关笔记

- **概念基础**：[[30-Agent-Engineering-Agent工程/01-Foundations-基础概念/Skill_10|Skill_10]] — Hermes 平台是 Skill 生态化使用的实例。
- **对比案例**：[[30-Agent-Engineering-Agent工程/03-Harness-and-Workflows-运行框架与工作流/OpenClaw架构与常见问题|OpenClaw架构与常见问题]] — Hermes 的平台分层可与 OpenClaw 架构比较。
- **业务应用**：[[80-Work-工作/01-YOFC-Internship-长飞实习/业务会议|业务会议]] — 长飞会议明确讨论用 Hermes 解析知识、选择参数和生成 BOM。
- **应用场景**：[[90-Personal-个人/02-Finance-财商/富爸爸穷爸爸_财商九课笔记|富爸爸穷爸爸_财商九课笔记]] — Hermes 的持续任务与工具能力可服务财务信息工作流。
- **对比案例**：[[30-Agent-Engineering-Agent工程/04-References-项目参考/AgentScope|AgentScope 2.0]] — Hermes 更接近开箱运行的 Agent 平台，AgentScope 更偏构建产品的框架与服务骨架。
