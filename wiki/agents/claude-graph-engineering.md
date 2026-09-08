---
id: claude-graph-engineering
title: Claude 图工程（Graph Engineering）：模式精要与与 DAG 的真实区别
type: reference
summary: Claude Code dynamic workflows 的图编排模式（节点契约/假边测试/菱形/检查器/loop-until-dry/模型分层/pipeline vs parallel），及批判分析——图论层面就是 DAG，真创新是编排从对话层下沉到代码层（节点烧 token，边免费）
tags: [agents, claude, graph-engineering, dag, orchestration, workflow, subagent]
status: draft
created: 2026-09-08
updated: 2026-09-08
source: session:oc_039419815ca8fa8923d371b60250cc38
confidence: high
review_after: 2026-10-08
links: []
superseded_by:
refuted_because:
---

## 背景

来源：两篇 X 长文 + 会话内批判性分析：
- Codez（@0xCodez）《Graph Engineering with Claude: 14-Step roadmap from 0 to graph architect》，2026-07-20，750 万阅读
- Mahax（@Mahaximus_）《Graph Engineering with Claude. What It Is and How to Actually Use It》，2026-07-29，220 万阅读（入门版：概念 + 假边测试 + 菱形 + 检查器 + 静态/动态图 + workflow 语法）

主题：用 Claude Code 的 dynamic workflows 把多步 agent 从线性链重画为图——节点=工作单元，边=数据依赖。

## 核心概念

- **节点** = 工作单元：一个 agent、一个有边界任务、一个输入一个输出。节点要有**契约**：有界输入（显式传入，不从共享窗口假设）、有界输出（JSON schema 在工具调用层强制校验，不匹配自动重试而非"解析并祈祷"）、恰好一个职责。
- **边** = 数据依赖：只有当 A 的输出真正喂给 B 的输入时才存在。"然后"不是边。边是**数据契约**：A 产出形状、B 消费形状；按数据（不按顺序）命名边，就能随时换掉任一端节点而不破坏图。
- **关键经济学**：节点烧 token，边免费。编排（fan-out/reduce/routing）活在纯 JS 代码里而非模型对话——协调零 token、确定性、可版本控制。
- Claude Code 入口：prompt 里说 "workflow" 关键字 → Claude 写 JS 编排脚本并派 subagent 舰队执行；`depends_on` 是唯一需要理解的语法（**无依赖=并行，有依赖=等待**）；`/deep-research` 是内置生产级图（scope → 并行搜索 → fetch → 对抗验证 → 合成）；ultracode 为每个实质任务自动规划工作流；按 `s` 把脚本存进 `.claude/workflows/`（版本控制、按名重跑）。

## 关键模式（Codez 14 步精要）

1. **假边测试**：对每个"然后"问"下一步读上一步的输出吗？"不读就没有边，等待是浪费。几乎任何工作流都有 2-3 条假边。五分钟画框-画箭头-逐箭头检查即可完成。
2. 线性脚本 = 退化图（单链无冗余：C 卡死则 D 永不跑，A 的工作困在上游）。
3. 节点契约（见核心概念）。
4. 边 = 数据契约；fan-out 与合成之间的 reduce（flatten/dedupe/filter）是纯代码，不需要 agent。
5. **parallel() 扇出**：屏障语义（等全部完成才返回）；抛异常的 thunk 解析为 null 而非整批失败；永远 `.filter(Boolean)`；并发按核心数封顶、超出排队（传一百个也能全部完成）。
6. **只在阶段真正需要全部前置结果时用屏障**（跨源去重/按影响排序/总量为空提前退出）；只是展平列表 = 边，内联做掉。parallel → transform → parallel 且中间无跨项依赖 = 本该用 pipeline。
7. **菱形**（主力拓扑）：**fan out → reduce → synthesize**。两条规则：并行节点必须真独立（无伪装成真边的假边）；汇聚节点必须真需要全部输入（只需要一个则其余是浪费）。
8. **条件路由**：对节点校验输出做 JS if/switch。节点得到 Claude 的判断，边得到脚本的可靠性——不会涌现"Claude 自己决定跳过审计"（跳过必须写进图里，而它没有）。
9. **边上放验证器**：唯一工作是试图杀死发现，活下来才放行。三模式：**对抗验证**（N 个独立怀疑者反驳，多数存活才保留）；**视角多样验证**（正确性/安全/可复现不同透镜——多样性抓 N 个相同检查抓不到的失败）；**评审团**（N 个尝试并行打分，从赢家合成并嫁接落败者最佳）。Bun 运行时移植即内建此模式。
10. **隔离**：失败控制在节点内（null + filter 即隔离层）；扇入设计成容忍缺失输入；**只在节点真的并行写文件时**才用 git worktree 隔离（安全带，不是默认税）。
11. **环（loop-until-dry）**：持续派 finder 直到连续 K 轮无新发现。**去重对象 = 见过的所有东西，不只是确认的结果**——否则被否决的发现每轮重现、循环永不枯竭，造出永远付费重新发现同样死胡同的机器（几乎所有人第一次都犯这个错）。
12. **模型分层**：subagent 默认继承会话模型（大运行全按会话档位计费）；单个 agent() 调用的 model 选项只路由该节点；重复节点（提取字段/分类工单）降便宜档、判断节点（合成/裁决）保持高档。不动图的形状就把吞 token 的图变经济。
13. **拓扑 = 成本和延迟**：默认 **pipeline()**（每个条目独立流过所有阶段、无屏障——A 在阶段 3 时 B 可能在阶段 1）；屏障让所有东西等最慢节点。"代码更干净"/"阶段感觉分开"不是理由；**分开 ≠ 同步**。
14. **自路由**：Claude 自己画图——描述目标，它现场生成编排脚本。"图不是规划出来的，是长出来的"。

## 与 DAG 的区别：什么真新、什么不新

**不新的部分（图论层面就是 DAG）**：菱形 = fork-join/scatter-gather（MapReduce 就是这个形状）；parallel() 屏障 = Spark stage 边界 / Airflow task 依赖；pipeline() = 数据流流水线；条件路由 = 分支；假边测试 = make -j 并行编译时代的依赖分析。唯一超出严格 DAG 的是受控环（LangGraph 同样允许 cycle）。

**真的差异（4 点）**：
1. **节点是概率性 LLM agent**：会幻觉、会跑题、会自信地返回垃圾（vs 确定性任务失败=抛异常一眼可见）。因此节点契约 + 检查器/验证器成为一等公民。
2. **编排下沉到代码层**：传统 orchestrator agent 用对话调度 subagent，每轮协调烧 token、占上下文；图工程把编排下沉到纯 JS——边免费。成本模型从 CPU 时间变成 token。
3. **上下文窗口是第一公民**：fan-out 的主要目的之一是上下文隔离（每个 subagent 带自己的上下文，只有最终答案回流，主会话不被 N 个源的原文淹没）——这个约束在传统调度理论里没有对应物。
4. **图由 LLM 运行时生成**：Airflow DAG 是工程师预先手写的 Python；这里 Claude 现场写编排脚本。

**核心创新一句话**：把 40 年 DAG/数据流调度智慧搬到 LLM agent 舰队，同时把协调从模型对话层下沉到代码层——**节点烧 token，边免费**。

**先例与反驳**：LangGraph（2024 初）几乎同模型（LLM 节点+条件边+允许环）；Claude 的差异 = 纯 JS 编排 + subagent 上下文隔离 + 图可现场生成。Alexis Santos 的反驳值得记住：**大多数人需要的是 loop 不是图**——只有存在真正的扇出-合并时，图才配得上它的复杂度（"一个 agent、一个定时、一份启动前读的日志"就够，无聊但每天都在跑）。

## 坑

- 把"然后"当边（假边）→ 白白等待。
- 无检查器节点 → 坏输出自信地流进综合节点；并行结构移除了天然检查点；汇聚处错误被好输出稀释、不可追踪。
- loop-until-dry 只对确认结果去重 → 被否决发现每轮重现，永不枯竭。
- 滥用 parallel() 屏障 → 全部等最慢节点；默认 pipeline()。
- 静态图够用却上动态图 → 更难调试、无法审计"到底跑了什么、为什么"。大多数"感觉需要动态图"的工作流只需要设计更好的静态图。
- 忘了 subagent 默认继承会话模型 → 整个大运行按会话档位计费。
- 并行写文件冲突 → 才需要 worktree 隔离。
- Mahax 文章自己的可疑边（评论区 Ryan Johnson 指出）："Research + Check sources 并行"只在检查器已有预定义源集合时成立——通常事实核查需要先知道来源和声明。

## 适用边界

- 适用于 Claude Code dynamic workflows（2026-07 时点能力：workflow 关键字、depends_on、parallel()/pipeline()、agent() 的 model 选项、/deep-research、ultracode、.claude/workflows/）。
- 值得用图：任务可重复（跑 >1 次）、时间节省会复利、或中间错误昂贵到检查器节点能回本、存在真扇出-合并形状。
- 不值得：一次性任务（线性版更快搭建和运行）、大多数自动化（loop 足够）。
- 对自建 agent 基础设施最值得抄的一条：**编排层放代码不放对话**——同时解决成本和上下文两个问题。

相关条目（EvolveKB store）：wiki/agents/knowledge-centric-self-improvement.md（知识中心自改进）、wiki/agents/uber-software-factory.md（Uber 的 subagent 默认弱模型与本条"模型分层"同源）。
