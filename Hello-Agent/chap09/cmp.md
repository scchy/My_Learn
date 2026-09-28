# 上下文工程四种做法对比：HelloAgents / Claude Code 泄露复盘 / Pi / DSH

> 口径说明：这里的 “Claude 泄露” 指 Claude Code 源映射/源码泄露后被公开复盘的上下文组装机制；“Pi” 指 pi.dev / earendil-works 的 coding agent harness；“DSH” 指 DeepSeek Harness。文档日期：2026-09-22。

## 0. 一页结论

| 维度 | HelloAgents ContextBuilder | Claude Code 泄露复盘 | Pi | DSH / DeepSeek Harness |
|---|---|---|---|---|
| 核心抽象 | 每次调用前用 GSSC 组装一个上下文串 | 完整 agent runtime：组装、工具循环、回合后压缩/记忆、缓存边界、后台整合 | 极薄 harness，把上下文策略交还给 extension/用户 | 事件溯源 Session + Surface 投影 + 可替换 compaction seam |
| 上下文载体 | `ContextPacket` + 固定分区模板 | system prompt 多层组装 + messages/history + cache-aware sections | AGENTS.md / SYSTEM.md / skills / prompt templates / extension 动态上下文 | append-only session log 是唯一真源；surface 是模型可见投影 |
| 选择机制 | relevance + recency 加权，贪心填入 token 预算 | 10+ 来源组装；工具结果、记忆、CLAUDE.md、git/OS 等分层进入 | extension 可在每轮前注入消息、过滤 history、做 RAG/长期记忆 | 默认不是“选 packet”，而是“投影 + 压力触发 + 区域替换”；社区插件再叠加选择器/编译/模型驱动剪枝 |
| 压缩触发 | 构建后超预算才兜底截断 | 回合后 post-sampling hooks：threshold check、memory extraction、可选 dream consolidation | 接近 context limit 自动 summarize，可由 extension 定制 | `agent/pre-step` 压力触发；先 tool-result pruner，再 tokenMeter 复测，必要时摘要替换 |
| 信息是否丢失 | 教学示例偏截断；长期状态依赖 Memory/Note/RAG | 生产做法会分层：microcompact/snip/collapse/reactive compact/dream；仍可能损失细节，所以强调 memory extraction | 由用户控制；默认少做，减少黑盒 | 日志不丢，投影丢：raw log 全量保留，surface replace 只隐藏旧节点；但已知 in-memory log 若无持久化后端会增长 |
| 缓存意识 | 基本未展开 | 很强：memoized sections、stable prefix、volatile section 会 break cache | 有 cache-aware 社区后端/思路，但非默认核心 | 官方事务关心 log/surface/lock；cache-aware 多为社区后端，如 Reasonix-style 后端 |
| 可审计/回放 | 模板可调试，但不等同事件溯源 | 泄露机制显示大量内部事件/flags，但外部不可完全验证 | 依赖会话文件与扩展质量 | 最强：compaction/start、summary、end 均 log-only，summary 以 user/message + surfaceOp replace 进入 surface |
| 适合借鉴 | 快速搭教学/原型：如何选料、预算守护 | 生产闭环：调用前组装 + 回合后沉淀 + 多层压缩 | 想自己掌控策略、避免 MCP/子代理膨胀 | 想要可回放、可审计、可替换压缩引擎的长会话系统 |

## 1. HelloAgents：调用前的 GSSC “选料器”

Datawhale HelloAgents 第九章把上下文工程定义为“每次模型调用前，如何拼装并优化输入上下文”。ContextBuilder 实现 GSSC：Gather 汇集系统指令、记忆检索、RAG 检索、最近少量对话历史与自定义 packet；Select 用相关性 + 新近性加权打分并按 token 预算贪心选择；Structure 落成 `[Role & Policies] / [Task] / [State] / [Evidence] / [Context] / [Output]` 等稳定分区；Compress 在超预算时做兜底压缩。

价值：
- 模板稳定，便于调试、A/B、评估。
- packet 化使系统指令、证据、历史、笔记可以统一打分。
- 明确 token budget 与 reserve ratio，避免系统指令被挤占。

边界：
- 更像 per-turn assembler，不负责长会话运行时：消息角色、tool message、prompt cache breakpoint、跨 session resume、压缩后代际失效等都要自己补。
- 教学实现的相关性是 Jaccard、token 是估算、压缩是截断；生产化要换成向量相关性、真实 tokenizer、结构化摘要与校验。

## 2. Claude Code 泄露复盘：不是“写一个 system prompt”，而是上下文生命周期

公开复盘显示，Claude Code 的关键不在单点提示词，而在循环：`user input -> append history -> build system prompt -> stream -> tool executor -> post-sampling hooks`。系统提示来自 role、tools、CLAUDE.md、memory、git、OS 等 10+ 来源；post-sampling hooks 在每个 assistant turn 后做 context compaction threshold check、memory extraction，以及可选的 “dream” consolidation。

值得借鉴的控制面：
- **回合后沉淀**：压缩和记忆提取不只发生在“快爆窗”时，而是每轮后的例行治理。
- **多层压缩**：泄露 flags 里可见 REACTIVE_COMPACT、CONTEXT_COLLAPSE、HISTORY_SNIP、CACHED_MICROCOMPACT、TOKEN_BUDGET、EXTRACT_MEMORIES 等，说明压缩策略是分层组合，而不是单一 summary。
- **缓存边界**：system prompt 被组织为多层 memoized sections；volatile section 会 break cache，稳定前缀尽量保 cache。
- **上下文分层**：不是所有信息都进 history；高价值事实进入 memory，工具细节被 microcompact，探索性上下文被子代理隔离。

风险/边界：
- 泄露资料不能等同官方设计文档；工程上应借鉴控制面，不复刻专有实现。
- 自动化越强，越要防“压缩丢证据”：错误栈、文件路径、命令、决策理由在压缩后必须可恢复。

## 3. Pi：把上下文工程做成可编程钩子，而不是内置全家桶

Pi 的取向是薄 harness：小 system prompt，默认不做 MCP、不做专用 sub-agent/plan mode；上下文控制点开放给用户。它支持 AGENTS.md 分层加载、SYSTEM.md replace/append、skills 按需加载、prompt templates；extension 可以在每轮前注入消息、过滤 message history、实现 RAG 或长期记忆。压缩上，Pi 接近 context limit 时自动 summarize older messages，并允许 extension 定制压缩策略。

价值：
- 避免 MCP/工具描述常驻上下文带来的膨胀；用 CLI tool + README 做 progressive disclosure。
- 让“什么进历史、什么只留引用、什么 JIT 取”变成显式策略。

边界：
- 自由度高意味着默认保护少；压缩 prompt、切点、保留尾部、证据回链都要自己设计。
- 社区 issue 显示其 compaction summarization prompt 曾硬编码 “AI coding assistant” 且不可覆盖，说明薄 harness 也会在产品化细节上翻车。

## 4. DSH / DeepSeek Harness：事件溯源 + 可替换 compaction seam

DSH 的差异最大：它不是先问“本轮选哪些 packet”，而是把会话建成 **append-only event log**，再从 log 派生模型可见的 **surface**。官方 `dsh-session` README 的表述是：session log 记录每个 model-visible fact，模型历史由该记录派生；compaction 会隐藏被取代条目，但不删除它们。

### 4.1 核心机制

- **事件溯源**：原始事件全量保留；surface 是当前活跃节点列表；`deriveMessages()` 只投影 surface 上的节点。好处是压缩、回放、fork、审计都基于同一事件流。
- **compaction 事务**：官方文档定义 `compaction/start`、`compaction/summary`、`compaction/end` 三个事件，均为 log-only；摘要本身作为一条 `user/message`，携带 `surfaceOp: { op: 'replace', start, end }` 进入 surface。也就是说：**日志记录事实，surface 只是视图**。
- **触发与回退**：pressure compaction 在串行 `agent/pre-step` 运行；达到压力或 overflow 后，`compaction-basic` 可先调用可选 `ctx.toolResultPruner`，再通过 `ctx.tokenMeter` 复测；若修剪已释放足够空间，可只做 surface advance 而不调用 LLM summary。失败请求恢复走 `agent/request-error`。
- **安全切点**：区域边界保留 tool-call/result 配对，但不保证保留完整 turn；超大 turn 中较早闭合的步骤也可被压缩。导出 `toolPairingBalancedBefore/After` 做配对校验。
- **可替换后端**：compaction 是 `ctx.compaction` seam；社区已有 near-lossless compilation、model-driven Active Context Pruning、cache-aware compaction、Fresh+Aggregate / Micro Compact / TailTrim selector 等后端或 profile。

### 4.2 与另外三家的本质差别

- HelloAgents 是“调用前选料”；DSH 是“全量事实 + 可替换视图”。它默认不追求把最优上下文一次性选出来，而是保证任何时候都能回答：模型到底看到了什么、隐藏了什么、为什么、原始证据在哪。
- Claude 泄露复盘强调回合后 hooks 与记忆整合；DSH 官方核心更强调事务、锁、日志、surface replace、tool pairing。记忆能力在 DSH 更像插件生态补足，而不是官方内置主线。
- Pi 把策略交给用户；DSH 把策略做成 seam，但仍然提供官方 compaction-basic 作为默认兜底。换言之：Pi 是“你自己搭”，DSH 是“我给你事务和替换点，你换引擎”。

### 4.3 已知坑

- 官方 compaction 只减少模型可见 surface，不修剪 `Session.log`。DSH 讨论区明确指出：in-memory session log 是 append-only 且无长度/字节上限；没有 persistence backend 时，长重会话会让 web 进程 heap 持续增长。也就是说：**上下文 bounded 不等于内存 bounded**。
- “near-lossless” 社区引擎通常指原文仍在 append-only log、可由 recall/search 找回；并不等于模型当前上下文无损失。
- 只加载一个 `ctx.compaction` 实现；Active Context Pruning 类插件与官方 basic 同时启用会冲突，需要按 profile/preset 明确替换。

## 5. 共同原则与分歧点

共同原则：
1. 上下文是有限预算，且边际收益递减；长上下文会 context rot。
2. 稳定系统区、易变历史区、外部证据区应分开管理。
3. 工具结果先修剪/外置/引用化，LLM summary 是更贵的后手。
4. 压缩必须保留下一步行动所需的“硬事实”：目标、决策、文件路径、命令、错误、未解决问题。
5. 长程任务需要上下文外持久化：memory、notes、handoff、artifact、子代理摘要。

主要分歧：
- **选料 vs 投影**：HelloAgents 偏 packet selection；DSH 偏 event-sourced projection；Claude 泄露是二者叠加 plus runtime hooks；Pi 把选择权外部化。
- **默认自动化程度**：Claude/DSH-basic 更自动；Pi 更 manual；HelloAgents 教学版居中但实现较简。
- **审计优先级**：DSH 最高；HelloAgents 中等；Pi 取决于用户；Claude 泄露无法外部验证但设计上重视 memory/extract/reconstruct。

## 6. 落地建议：取交集，不取宗教

建议组合：
- **稳定层**：system prompt 分区，稳定前缀保 cache；敏感配置不要塞进可回显系统提示。
- **选料层**：每轮 Gather 后做 packet selection；系统指令与 hard constraints 不参与淘汰。
- **历史层**：保留 role/tool 结构；工具结果超阈值先 pruner/offload；摘要只在仍超压时触发。
- **压缩层**：两级策略。轻触式清理深历史工具输出；重压缩用结构化摘要，字段至少含 goal、decisions、files/commands、errors/fixes、pending next step。
- **沉淀层**：回合后异步 memory extraction；长任务用 notes/handoff；子代理只回传凝练摘要与证据指针。
- **观测层**：记录每次上下文构成、压缩事件、shadowed range/token、recall 命中率；DSH 的 log-only compaction 事件是可借鉴范式。
- **恢复层**：压缩必须可回放/可检索；至少保留 raw log 或 session archive，否则“省 token”会变成“丢证据”。

## 7. 参考来源

- HelloAgents Datawhale 第九章：上下文工程 / ContextBuilder / NoteTool / TerminalTool。https://hello-agents.datawhale.cc/#/./chapter9/第九章%20上下文工程
- DeepSeek Harness `dsh-session` README：append-only session log、derive model history、compaction hides without deleting。https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/README.md
- DeepSeek Harness Compaction reference：`compaction/*` log-only events；summary 经 `user/message + surfaceOp replace`；`agent/pre-step` pressure compaction；tool pairing。https://deepseek-harness.github.io/deepseek-harness/en/reference/subsystems/compaction
- DeepSeek Harness discussion #4722：in-memory session log unbounded；compaction trims surface not log。https://github.com/deepseek-ai/deepseek-harness/discussions/4722
- Pi docs：dynamic context extensions、compaction、AGENTS.md/SYSTEM.md/skills。https://pi.dev/docs/context/
- Claude Code 泄露复盘公开分析：query loop、system prompt 10+ sources、post-sampling hooks、feature flags。https://github.com/davccavalcante/claude-code-source-code-analysis
