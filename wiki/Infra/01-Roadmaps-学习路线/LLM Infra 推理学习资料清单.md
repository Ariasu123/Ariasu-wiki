
## 目录

- [模块一：GPU/CUDA 基础与 Kernel 优化](#模块一gpucuda-基础与-kernel-优化)
- [模块二：主流推理训练框架的核心技术](#模块二主流推理训练框架的核心技术)
- [模块三：瓶颈分析与性能优化](#模块三瓶颈分析与性能优化)
- [模块四：强化学习](#模块四强化学习)
- [模块五：算法](#模块五算法)
- [模块六：C++ / Python / 工程](#模块六c--python--工程)
- [模块七：学术界和业界团队](#模块七学术界和业界团队)
- [模块八：AI Infra 课程](#模块八ai-infra-课程)
- [相关页面](#相关页面)

---

## 模块一：GPU/CUDA 基础与 Kernel 优化

**入门与编程模型**

1. [CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)（小小将）
2. [理解CUDA中的thread, block, grid和warp](https://zhuanlan.zhihu.com/p/123170285)（三七和酒）——全局内存、共享内存与寄存器
3. [NVIDIA 官方 CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/)
4. [NVIDIA 官方入门博客：Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
5. [CUDA-Programming-Guide-in-Chinese](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese)（GitHub 中文翻译）
6. [你敲一行 CUDA 编译命令，GPU 后台偷偷干了 8 件大事！](https://mp.weixin.qq.com/s/JQzpbScrswk_gGsu6Czb2A)
7. [CUDA编程系列2：CUDA 编程模型](https://mp.weixin.qq.com/s/6LxI65-YAsMIyY8sRuMSkA)
8. [一条 CUDA 命令的硬件之旅](https://mp.weixin.qq.com/s/bj6aYEif3TPLfEjrokZv5A)
9. CUDA编程系列(3)：CUDA 内存模型（草帽路飞；原文为限时签名链接）；另有同作者《【CUDA】存储模型和内存管理》、《GPU 硬件基础及内存模型》

**矩阵乘优化**

10. [CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)
11. [如何高效实现矩阵乘？万文长字带你从CUDA初学者的角度入门](https://mp.weixin.qq.com/s/rWWx0Uf4oin0kmtEjVXBqw)（机器之心）
12. [英伟达Blackwell上的矩阵乘优化：第一部分－介绍](https://mp.weixin.qq.com/s/nABsPXfgBKzdEUDumZjRhw)（朱金鹏，Modular 系列）
13. [CUDA编程：矩阵乘运算从CPU到GPU](https://mp.weixin.qq.com/s/QNqGsXhgxIzwbf1wBEoUTA)（InfraTech）

**练习与教程**

14. [BasicCUDA](https://github.com/CalvinXKY/BasicCUDA)（GitHub 练习仓库）
15. [LeetGPU Challenges](https://leetgpu.com/challenges/reduction)（GPU 版 LeetCode）
16. [Modern GPU Programming For MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/index.html)
17. [ashraf-bhuiyan 的 GPU 博客](https://ashraf-bhuiyan.com/blog/)

**GPU 架构与内存**

18. [【万字长文】GPU架构全攻略-架构设计与高性能计算启发](https://zhuanlan.zhihu.com/p/1984203517506917754)
19. [如何系统学习GPU架构？](https://www.zhihu.com/question/319355296/answer/1931398398445060845)
20. [GPU 架构详解：以 NVIDIA H100 为例](https://github.com/iosmers/Cuda_Tutorial/blob/main/H100-Streaming-Multiprocessor-SM.md)
21. [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU)（Branch Education）
22. [用最好的动画为你讲解——HBM的原理](https://www.youtube.com/watch?v=yF2BY8kQfyo)
23. [Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)（NVIDIA 官方）
24. [从CUDA内存管理的角度上，有什么常用的优化技巧？](https://www.zhihu.com/question/372915215/answer/2038007243115869383)

**进阶机制（TMA / Bank Conflict / Pin Memory / DMA）**

25. [NVIDIA TMA 全面分析](https://zhuanlan.zhihu.com/p/1945136522455122713)（请问肥嘟嘟）
26. [CUTLASS Tutorial: Mastering the NVIDIA Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)（Colfax）
27. [What is Bank Conflict?](https://modal.com/gpu-glossary/perf/bank-conflict)（Modal GPU Glossary）
28. [CUDA Shared Memory and Bank Conflict Optimization](https://www.youtube.com/watch?v=rDoNTB-VtkM)
29. [CUDA 如何调度 kernel 到指定的 SM？](https://www.zhihu.com/question/652642080/answer/1985070382152184624)（ZxZhao）
30. [What is Pin Memory](https://giahuy04.medium.com/pinned-memory-5d408b72241d)（CisMine Ng）
31. [Pinned Memory and DMA Transfers in PyTorch](https://www.abhik.ai/concepts/language-internals/pin-memory)
32. [CPU传输 & GPU计算的并行（pin_memory, non_blocking）](https://www.bilibili.com/video/BV15Xxve1EtZ/)
33. [DMA Controller: How Peripheral Devices Transfer Data to RAM](https://www.youtube.com/watch?v=s8RGHggL7ws)

**FlashAttention 与算子实例**

34. [FlashAttention 内部机制解析](https://www.bilibili.com/video/BV19DPJe5EJQ/)（小小升）
35. [FlashAttention 动画演示](https://www.bilibili.com/video/BV1HJWZeSEF4/)
36. [Tri Dao 官方讲解 FlashAttention](https://www.youtube.com/watch?v=FThvfkXWqtE)
37. [FlashAttention-2 in CuTe, from Scratch](https://blog.echen.io/p/flashattention-2-in-cute-from-scratch/)
38. [Blackwell架构的Linear Attention算子优化](https://www.bilibili.com/video/BV1gpG46xEN2)（NVIDIA）

**Roofline 与面试**

39. [Roofline Model与深度学习模型的性能分析](https://zhuanlan.zhihu.com/p/34204282)（Michael Yuan）
40. [【CUDA调优指南】Roofline Model详解](https://www.bilibili.com/video/BV1p1rkYqESW/)
41. [如果你是一位cuda面试官，你会问哪些问题？](https://www.zhihu.com/question/10951382954/answer/2041954576190985755)（kaiyuan）

## 模块二：主流推理训练框架的核心技术

**vLLM**

1. [vLLM框架快速入门引导](https://zhuanlan.zhihu.com/p/1984742841528902530)（kaiyuan）
2. [vLLM Scheduler逻辑难啃？先手搓一个基础调度器](https://zhuanlan.zhihu.com/p/1988193790129902960)（kaiyuan）
3. [vLLM显存管理详解](https://zhuanlan.zhihu.com/p/1916529253169734444)
4. [vLLM V1 KV cache 管理机制剖析](https://zhuanlan.zhihu.com/p/1954128446398633139)
5. [vLLM V1 Scheduler的调度逻辑&优先级分析](https://zhuanlan.zhihu.com/p/1900957007575511876)
6. [vLLM框架V1演进分析](https://zhuanlan.zhihu.com/p/1894423873145004335)
7. [vLLM的prefix cache为何零开销](https://zhuanlan.zhihu.com/p/1896927732027335111)
8. [vLLM DP特性与演进方案分析](https://zhuanlan.zhihu.com/p/1909265969823580330)
9. [LLM推理数据并行负载均衡(DPLB)浅析](https://zhuanlan.zhihu.com/p/1927317160889386326)
10. [AI Infra面试常考——vLLM大模型推理框架](https://zhuanlan.zhihu.com/p/2011083570035319972)（砂川同学）
11. [vLLM 架构详解（一）](https://zhuanlan.zhihu.com/p/693279132)（CalebDu）
12. [vLLM源码解析1：整体架构](https://zhuanlan.zhihu.com/p/691045737)（猛猿）
13. [LLM / vLLM：Sampling 采样介绍](https://zhuanlan.zhihu.com/p/1911546380444496008)（hcxx）
14. [vLLM 中 FP8 KV-Cache 与 Attention 量化的现状](https://zhuanlan.zhihu.com/p/2037813435979805457)
15. [vLLM Attention后端整体结构实现详解](https://zhuanlan.zhihu.com/p/2048677881111392915)
16. [vLLM Triton Attention 后端深度解析](https://zhuanlan.zhihu.com/p/2017515128577349209)
17. [vLLM 在 Prefill/Decode 阶段对 MLA 的不同实现对比](https://zhuanlan.zhihu.com/p/1897225385751585767)
18. [vLLM 官方论坛](https://discuss.vllm.ai/latest)
19. [AISystem 课程内容大纲](https://infrasys-ai.github.io/aisystem-docs/)

**SGLang**

20. [SGLang 源码学习笔记（一）：Cache、Req与Scheduler](https://zhuanlan.zhihu.com/p/17186885141)
21. [看不懂SGLang?先试试miniSGLang！](https://zhuanlan.zhihu.com/p/1986026310913528033)（kaiyuan）
22. [手撕SGLang KV Cache核心逻辑：快速理解RadixAttention](https://zhuanlan.zhihu.com/p/1994495318197305400)（kaiyuan）

**PD 分离**

23. [PD分离还是不分离，看这篇文章就够了](https://arxiv.org/pdf/2508.01989)
24. [Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](https://arxiv.org/pdf/2603.13358)——多轮对话第 2 轮起直接在 Decode 节点做增量计算（本站笔记：[ICML 2026 AI Infra 论文趋势解读](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/AI%20Infra%20%E5%89%8D%E6%B2%BF%E8%AE%BA%E6%96%87%E8%B6%8B%E5%8A%BF%EF%BC%88ICML%202026%EF%BC%89.md) 有详解）
25. [vLLM PD分离方案浅析](https://zhuanlan.zhihu.com/p/1889243870430201414)（kaiyuan）
26. [vLLM PD分离KV cache传递机制详解与演进分析](https://zhuanlan.zhihu.com/p/1906741007606878764)（kaiyuan）
27. [PD分离-XpYd系统服务化](https://zhuanlan.zhihu.com/p/30619735151)
28. [PD 分离推理架构详解](https://zhuanlan.zhihu.com/p/1953242569489257274)
29. [大模型推理夯实：SGLang PD分离流程](https://zhuanlan.zhihu.com/p/2049550468813264222)
30. [vLLM v1 PD分离与mooncake引擎](https://www.bilibili.com/video/BV1UwDjBXEU1/)

**Mooncake / KV Cache 中心化**

31. [Mooncake: Trading More Storage for Less Computation（论文演讲）](https://www.youtube.com/watch?v=-Lpx9QuCEsw)
32. [Mooncake 讲解](https://www.bilibili.com/video/BV1KmJ8zaEkj/)
33. [共建大模型推理生态：Mooncake、KTransformers 与 SGLang](https://www.bilibili.com/video/BV17vwezLEFV/)
34. [Mooncake: A KVCache-Centric Disaggregated Architecture 解读](https://mp.weixin.qq.com/s/Rp3wTRrE9Lt_djOdE6UTYg)（三只大黄）
35. [Mooncake：Kimi 的推理引擎，长上下文吞吐暴涨525%](https://mp.weixin.qq.com/s/rynv_OEQdrFNVdankxg6ag)（AI 大排档）
36. [Mooncake：以 KVCache 为中心的云上 LLM 推理软件栈](https://www.bilibili.com/video/BV1SwDjBXEHY/)

**MTP / 投机采样**

37. [DeepSeek MTP 优化](https://www.bilibili.com/video/BV1jA9HYfEAC/)
38. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/pdf/2401.15077)
39. [EAGLE-2: Faster Inference with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
40. [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840)
41. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/html/2602.06036v2)（另有[讲解视频](https://www.youtube.com/watch?v=NXIiKnatDmA)）
42. [Speculative Speculative Decoding](https://arxiv.org/pdf/2603.03251)
43. [近三年来的MTP技术演变综述](https://mp.weixin.qq.com/s/5bluDyAcLdHkmwg4eRnYGQ)
44. [JetSpec: Parallel Tree Drafting](https://github.com/hao-ai-lab/JetSpec)

**其他**

45. [MoE 模型中的通信优化](https://www.bilibili.com/video/BV1NQGx6mELa/)（NVIDIA）
46. [InfraTech 开源仓库](https://github.com/CalvinXKY/InfraTech)
47. [深入解析LLM推理Decode Batch内部负载不均问题](https://zhuanlan.zhihu.com/p/2044873485688861958)
48. [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)——百度 Ring-AllReduce 实现（作者 [Andrew Gibiansky](https://andrew.gibiansky.com/)）
49. [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595)
50. [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/pdf/2404.14294)（2024 综述）
51. [手撕代码：softmax / safe softmax / online softmax（FlashAttention 核心）](https://www.bilibili.com/video/BV1ETKW67EXG/)

## 模块三：瓶颈分析与性能优化

1. [如何系统地分析和定位大模型推理框架（如 SGLang, vLLM）的性能瓶颈？](https://www.zhihu.com/question/1993781500349539243/answer/2030665166598174331)
2. [Nsight Systems工具原理与GPU性能优化实战详解](https://zhuanlan.zhihu.com/p/1955933603209917837)（鹅厂架构师）
3. [GPU性能瓶颈定位：Nsight Systems（nsys）实操技巧](https://zhuanlan.zhihu.com/p/1952331097288479733)（白蘋渡口）
4. [AI Systems Performance Engineering：PyTorch性能分析、调优与扩展](https://zhuanlan.zhihu.com/p/2046914224316798131)（想飞的石头）
5. [如何系统性定位并分析 PyTorch 模型训练/推理中的性能瓶颈？](https://www.zhihu.com/question/1927112862976972744/answer/1987626122918257208)（BBuf）
6. [大模型推理 & memory bandwidth bound：性能瓶颈与优化概述](https://zhuanlan.zhihu.com/p/6228754823)（haoguang.dai）
7. [CUDA 性能分析实战：Nsight Systems和Nsight Compute入门](https://zhuanlan.zhihu.com/p/2025619489903900661)（鬼马行天Mark）
8. [NVIDIA性能分析工具 Nsight Systems/Compute 使用介绍](https://www.bilibili.com/video/BV15P4y1R7VG/)（百度技术培训中心）
9. [LLM推理加速方法-2025年终总结](https://zhuanlan.zhihu.com/p/1987290155812423513)
10. [The Art of Debugging](https://github.com/stas00/the-art-of-debugging)（Stas Bekman）

## 模块四：强化学习

1. [slime 源码走读：SGLang-Native 推理架构解析上篇](https://zhuanlan.zhihu.com/p/2039496436262363196)
2. [图解RLHF](https://www.bilibili.com/video/BV1XgLp6ZEtE/)
3. [Reinforcement Learning from Human Feedback 综述](https://arxiv.org/pdf/2504.12501)
4. [A Survey of Reinforcement Learning from Human Feedback](https://arxiv.org/pdf/2312.14925)
5. [RLHF, Clearly Explained!!!](https://www.youtube.com/watch?v=qPN_XZcJf_s)（YouTube）
6. [Kimi K2 如何实现高效 RL 参数更新（checkpoint-engine）](https://moonshotai.github.io/checkpoint-engine/?lang=zh)
7. [Kimi checkpoint-engine GitHub](https://github.com/MoonshotAI/checkpoint-engine)
8. [RL共卡权重同步：vLLM与训练框架之间的IPC实践](https://zhuanlan.zhihu.com/p/2044116178860298908)

## 模块五：算法

1. [深入解读DeepSeek V1-V4！](https://www.bilibili.com/video/BV1rpovBCEGH)（飞天闪客）
2. [GLM-5.2 为什么在 long-horizon Agentic RL 上从 GRPO 回到 PPO？](https://zhuanlan.zhihu.com/p/2050537350888002209)
3. [1M 上下文很难吗？深入解读 GLM5.2 上下文背后的技术](https://www.bilibili.com/video/BV1uVLX6uEYC/)（飞天闪客）
4. [GLM-5技术报告解读](https://www.bilibili.com/video/BV1J5A3zWEdS/)（李小羊学AI）
5. [MiMo-V2-Flash技术报告解读](https://www.bilibili.com/video/BV1ENqkB6E3f)（李小羊学AI）
6. [Step3.5 Tech Report](https://arxiv.org/pdf/2602.10604)
7. [DeepSeek V3 Tech Report](https://arxiv.org/pdf/2412.19437)
8. [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/pdf/2602.15763)
9. [DeepSeek V4 Tech Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/discussions/129)
10. [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（Sebastian Raschka）
11. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://www.bilibili.com/video/BV1UhTu6tEKs/)
12. [ViT 简介](https://www.bilibili.com/video/BV1gnWdzSEzY/)（多模态相关）
13. [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://github.com/openai/whisper)（语音模型）
14. [Kimi K3深度解读](https://www.bilibili.com/video/BV1rJKa63Eic/)（Zomi）
15. [从 RNN 到 Transformer](https://www.bilibili.com/video/BV1MNoRYEEVM/)（飞天闪客）
16. [RNN 讲解](https://www.bilibili.com/video/BV1FJ8UzfEJX/)（RethinkFun）

## 模块六：C++ / Python / 工程

1. [AI Infra面试常考—C++八股](https://zhuanlan.zhihu.com/p/2017623155192115538)
2. [AI Infra 面试问题QA 总结——C++](https://zhuanlan.zhihu.com/p/2032225037470671229)
3. [什么是僵尸进程和孤儿进程](https://www.cnblogs.com/Anker/p/3271773.html)
4. [花了两天，终于把 Python 的 setup.py 给整明白了](https://zhuanlan.zhihu.com/p/276461821)
5. [Python异步编程 asyncio](https://www.bilibili.com/video/BV157mFYEEkH/)
6. [A Conceptual Overview of asyncio](https://docs.python.org/3/howto/a-conceptual-overview-of-asyncio.html)（Python 官方）
7. [英伟达的护城河究竟是什么？](https://www.bilibili.com/video/BV1MjN16oE84/)
8. [PCIe到底是个什么东西？](https://www.bilibili.com/video/BV1n4411m7HX/)

## 模块七：学术界和业界团队

- [Sky Computing Lab](https://sky.cs.berkeley.edu/)（UC Berkeley；前身 AMPLab/RISELab，Ray、Spark、vLLM、SGLang 生态来源）
- [MAST Lab](http://mast.stanford.edu/)、[CSL](http://csl.stanford.edu/)、[DAWN Lab](http://dawn.cs.stanford.edu/)（Stanford）
- [CSAIL Systems](http://csail.mit.edu/)、[PDOS](https://pdos.csail.mit.edu/)、[Computer Systems Security Group](https://css.csail.mit.edu/)（MIT）
- [SysLab](http://syslab.cs.washington.edu/)（University of Washington）
- [CSL](https://csl.illinois.edu/)（UIUC）
- [Parallel Data Laboratory (PDL)](https://www.pdl.cmu.edu/)（CMU）
- [CSL](https://www.csl.cornell.edu/)（Cornell）
- [Systems Group](https://systems.ethz.ch/)（ETH Zurich）
- [Systems Laboratory](https://systems.engin.umich.edu/)、[Computer Engineering Laboratory](https://ce.engin.umich.edu/)（University of Michigan）
- [CERCS](https://cercs.gatech.edu/)、[CASL](https://casl.gatech.edu/)（Georgia Tech）
- [Systems & Networking Group](https://www.cs.princeton.edu/research/areas/systems)（Princeton）
- [Laboratory for Parallel and Distributed Systems](https://ece.utexas.edu/research/groups/laboratory-parallel-and-distributed-systems)（UT Austin）
- [CSL](https://csl.yale.edu/)（Yale）
- [Computer Systems Group](https://systems.cs.colorado.edu/)（CU Boulder）

## 模块八：AI Infra 课程

**LLM / AI 系统课程**

1. [Stanford CS349D: AI Inference Infrastructure](https://web.stanford.edu/class/cs349d/)
2. [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
3. [Stanford CS224G: Building and Scaling LLM Applications](https://web.stanford.edu/class/cs224g/)
4. [UC Berkeley CS294/194-196: LLM Agents / Agentic AI](https://rdi.berkeley.edu/agentic-ai/f25)
5. [UC Berkeley CS294/194-280: Advanced LLM Agents](https://rdi.berkeley.edu/adv-llm-agents)
6. [MIT AI System Architecture and LLM Applications](https://professional.mit.edu/llm)
7. [MIT 6.S191: Introduction to Deep Learning](https://introtodeeplearning.com/)
8. [CMU Large Language Model Systems](https://www.cmu.edu/online/generative-ai-llms)
9. [CMU 17-445/645: Machine Learning in Production / AI Engineering](https://mlip-cmu.github.io/s2025/)

**计算机体系结构**

10. [MIT 6.5930: Hardware Architecture for Deep Learning](https://csg.csail.mit.edu/6.5930/)
11. [CMU 18-447: Introduction to Computer Architecture](https://courses.ece.cmu.edu/18447)
12. [Stanford CS 217: Hardware Accelerators for Machine Learning](https://cs217.stanford.edu/)
13. [UC Berkeley CS267: Parallel Computing Applications](https://www2.eecs.berkeley.edu/Courses/CSC267/)
14. [University of Michigan EECS 570: Parallel Computer Architecture](https://www.eecs.umich.edu/courses/eecs570/)
15. [Cornell ECE 4750/CS 4420: Computer Architecture](https://www.csl.cornell.edu/courses/ece4750/)
16. [Cambridge: Introduction to Computer Architecture](https://www.cl.cam.ac.uk/teaching/2526/IntComArch/)
17. [Oxford: Computer Architecture](https://www.cs.ox.ac.uk/teaching/courses/2022-2023/ca/)
18. Coursera – Computer Architecture（David Wentzlaff，Princeton）

**GPU 体系结构**

19. NVIDIA DLI: Fundamentals of Accelerated Computing with CUDA C/C++（官方课程）
20. [Northwestern COMP_SCI 368/468: Programming Massively Parallel Processors with CUDA](https://www.mccormick.northwestern.edu/computer-science/academics/courses/descriptions/368.html)
21. [University of Virginia CS 6501: GPU Architectures](https://adwaitjog.github.io/teach/uva_6501_s25.html)
22. [Georgia Tech OMSCS CS7295: GPU HW/SW](https://omscs.gatech.edu/cs-7295-gpu-hardware-and-software)
23. [KTH DD2360: Applied GPU Programming](https://www.kth.se/student/kurser/kurs/DD2360?l=en)
24. University of Virginia CS 6501: CPU/GPU Memory Systems and Near-Data Processing
25. SMU ECE/CS 7385: GPU Interfacing: Parallel Architectures for ML
26. [KAUST CS 380: GPU and GPGPU Programming](https://vccvisualization.org/CS380_GPU_and_GPGPU_Programming/)
27. University of Toronto ECE 559: GPU Architecture

## 相关页面

- [LLM 推理加速与算子优化学习路线](LLM%20%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md) — 本站自建的推理优化学习路线，可与本清单互为补充
- [AI Infra 领域概览](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md) — AI Infra 五大板块与入行建议
