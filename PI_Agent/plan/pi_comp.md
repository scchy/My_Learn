# Pi Agent 上下文压缩机制（Context Compaction）源码解析

> 基于 `earendil-works/pi` 仓库 `packages/coding-agent/src/core/compaction/compaction.ts`

---

## 一、概述

Pi Agent 的上下文压缩机制是一个**双摘要 + 增量更新 + Turn 边界保护**的精密管道。其核心设计哲学是 **"最小破坏"**：在压缩历史消息的同时，保护当前进行中的 Turn 不被打断，并通过文件操作追踪确保工程上下文不丢失。

---

## 二、整体架构：5 阶段流程

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: prepareCompaction()                               │
│  ├── 定位上一次 CompactionEntry 边界                        │
│  ├── 计算当前上下文 Token 总量                              │
│  └── 调用 findCutPoint() 确定切割点                         │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: findCutPoint()                                    │
│  ├── 从最新消息倒推，累积估算 Token                         │
│  ├── 超过 keepRecentTokens 时向前对齐合法切割点             │
│  └── 标记 isSplitTurn（是否切割在 Turn 中间）               │
├─────────────────────────────────────────────────────────────┤
│  Stage 3: generateSummary()                                 │
│  ├── 首次压缩 → SUMMARIZATION_PROMPT                        │
│  ├── 增量压缩 → UPDATE_SUMMARIZATION_PROMPT                 │
│  └── 输出结构化摘要（Goal / Progress / Next Steps 等）      │
├─────────────────────────────────────────────────────────────┤
│  Stage 4: generateTurnPrefixSummary() [可选]                │
│  └── 当 isSplitTurn=true 时，并行生成当前 Turn 前缀摘要     │
├─────────────────────────────────────────────────────────────┤
│  Stage 5: compact()                                         │
│  ├── 合并双摘要（历史摘要 + Turn 前缀摘要）                 │
│  ├── 追加文件操作记录（readFiles / modifiedFiles）          │
│  └── 返回 CompactionResult → SessionManager 持久化          │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、核心：切割点算法（`findCutPoint`）

### 3.1 从后往前累积

```typescript
for (let i = endIndex - 1; i >= startIndex; i--) {
    const messageTokens = estimateTokens(entry.message);
    accumulatedTokens += messageTokens;
    if (accumulatedTokens >= keepRecentTokens) {
        // 找到最近的有效切割点
        cutIndex = 第一个 ≥ i 的有效切割点;
        break;
    }
}
```

**关键设计**：从最新消息倒推，累积估算 token 数，直到超过 `keepRecentTokens`（默认 20,000），然后**向前对齐到最近的合法切割点**。

### 3.2 合法切割点（`findValidCutPoints`）

| 可切割的消息类型 | 不可切割的消息类型 |
|------------------|--------------------|
| `user` | `toolResult`（孤儿问题） |
| `assistant`（含其后续 tool results） | `thinking_level_change` |
| `bashExecution` | `model_change` |
| `custom` / `branchSummary` / `compactionSummary` | `label` / `session_info` |

**Turn 边界规则**：
- 若切割点落在 `assistant` 消息上，其后续的 `toolResult` 会自动跟随保留
- 若切割点落在 Turn 中间（非 user 消息），标记 `isSplitTurn = true`，触发**双摘要**

### 3.3 Turn 起始点查找（`findTurnStartIndex`）

```typescript
export function findTurnStartIndex(entries, entryIndex, startIndex): number {
    for (let i = entryIndex; i >= startIndex; i--) {
        if (entry.type === "branch_summary" || entry.type === "custom_message") return i;
        if (entry.type === "message" && (role === "user" || role === "bashExecution")) return i;
    }
    return -1;
}
```

---

## 四、双摘要机制（Split Turn 处理）

当 `isSplitTurn = true` 时，**并行**生成两个摘要：

### 4.1 历史摘要（`generateSummary`）

- **覆盖范围**：`boundaryStart` → `turnStartIndex`（当前 Turn 开始之前）
- **Prompt 结构**：

```
<conversation>
<previous_summary>  ← 如果有前一次压缩，增量更新
SUMMARIZATION_PROMPT / UPDATE_SUMMARIZATION_PROMPT
```

- **输出格式强制结构化**：

```markdown
## Goal
[用户目标]

## Constraints & Preferences
- [约束与偏好]

## Progress
### Done
- [x] [已完成任务]

### In Progress
- [ ] [进行中任务]

### Blocked
- [阻塞项]

## Key Decisions
- **[决策]**: [理由]

## Next Steps
1. [下一步]

## Critical Context
- [关键数据、文件路径、错误信息]
```

### 4.2 Turn 前缀摘要（`generateTurnPrefixSummary`）

- **覆盖范围**：`turnStartIndex` → `cutPoint`（当前被切割的 Turn 的前半段）
- **Prompt**：`TURN_PREFIX_SUMMARIZATION_PROMPT`
- **输出格式**：

```markdown
## Original Request
[用户本轮请求]

## Early Progress
- [前缀中的关键决策和工作]

## Context for Suffix
- [理解保留的后缀所需的信息]
```

### 4.3 合并输出

```typescript
summary = `${historyResult}

---

**Turn Context (split turn):**

${turnPrefixResult}`;
```

**设计意图**：保留 Turn 的完整性——历史压缩成摘要，当前进行中的 Turn 保留原始消息，但其前缀也压缩，让模型既能理解上下文又能继续当前工作。

---

## 五、Token 估算策略

### 5.1 上下文 Token 计算

```typescript
export function calculateContextTokens(usage: Usage): number {
    return usage.totalTokens || usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
}
```

**触发条件**：

```typescript
contextTokens > model.contextWindow - settings.reserveTokens
```

### 5.2 消息 Token 估算（`estimateTokens`）

采用 **chars/4 启发式**（保守高估）：

| 消息类型 | 估算方式 |
|----------|----------|
| `user` / `assistant` text | `chars / 4` |
| `assistant` toolCall | `name.length + JSON.stringify(args).length` |
| `toolResult` image | **固定 1,200 tokens**（`4800 chars / 4`） |
| `bashExecution` | `command.length + output.length` |
| `compactionSummary` | `summary.length / 4` |

> 图片按 1,200 tokens 估算，这是 Pi Agent 处理图片附件时的关键假设。

### 5.3 上下文 Token 估算（`estimateContextTokens`）

```typescript
export function estimateContextTokens(messages: AgentMessage[]): ContextUsageEstimate {
    const usageInfo = getLastAssistantUsageInfo(messages);
    if (!usageInfo) {
        // 无 usage 记录时，全部用启发式估算
        let estimated = 0;
        for (const message of messages) estimated += estimateTokens(message);
        return { tokens: estimated, usageTokens: 0, trailingTokens: estimated, lastUsageIndex: null };
    }
    const usageTokens = calculateContextTokens(usageInfo.usage);
    let trailingTokens = 0;
    for (let i = usageInfo.index + 1; i < messages.length; i++) {
        trailingTokens += estimateTokens(messages[i]);
    }
    return { tokens: usageTokens + trailingTokens, usageTokens, trailingTokens, lastUsageIndex: usageInfo.index };
}
```

**为什么用最后一条非中断 assistant 的 usage？** 因为 LLM provider 返回的 usage 是最准确的，后续新增消息用启发式补全。

---

## 六、增量摘要（Iterative Compaction）

```typescript
let previousSummary: string | undefined;
if (prevCompactionIndex >= 0) {
    previousSummary = prevCompaction.summary;
}
```

### 6.1 边界处理

1. 找到上一次 `compaction` entry 的位置
2. 新的摘要范围从 `firstKeptEntryId`（上次保留的第一条消息）开始
3. **绝不跨越压缩边界**——每次压缩只处理两个 compaction 事件之间的消息

### 6.2 Prompt 切换

| 场景 | 使用的 Prompt |
|------|---------------|
| 首次压缩 | `SUMMARIZATION_PROMPT` |
| 增量压缩 | `UPDATE_SUMMARIZATION_PROMPT` |

**增量更新规则**（`UPDATE_SUMMARIZATION_PROMPT` 要求）：
- PRESERVE 所有已有信息
- ADD 新进度、决策和上下文
- UPDATE Progress 部分：将已完成项从 "In Progress" 移至 "Done"
- UPDATE "Next Steps" 基于当前状态
- PRESERVE 精确文件路径、函数名、错误信息

---

## 七、文件操作追踪（FileOps）

```typescript
export interface CompactionDetails {
    readFiles: string[];
    modifiedFiles: string[];
}
```

### 7.1 提取来源

1. **前一次压缩的 details**（如果是由 Pi 生成的，非 hook 注入）
2. **当前待摘要消息中的 tool calls**（通过 `extractFileOpsFromMessage` 解析）

### 7.2 输出格式

```markdown
## Files Read
- /path/to/file1
- /path/to/file2

## Files Modified
- /path/to/file3
```

这确保了即使历史被压缩，文件操作轨迹也不会丢失。

---

## 八、配置与默认值

```typescript
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
    enabled: true,
    reserveTokens: 16384,    // 摘要输出空间 (~13k) + 安全余量 (~3k)
    keepRecentTokens: 20000, // 保留最近原始消息的 token 预算
};
```

### 8.1 摘要输出限制

```typescript
const maxTokens = Math.min(
    Math.floor(0.8 * reserveTokens),  // ~13,107
    model.maxTokens > 0 ? model.maxTokens : Infinity
);
```

Turn 前缀摘要更省：

```typescript
const maxTokens = Math.min(Math.floor(0.5 * reserveTokens), model.maxTokens);
```

### 8.2 配置存储位置

```
~/.pi/agent/settings.json
```

---

## 九、与 Claude Code 的核心差异

| 维度 | Pi Agent (`compaction.ts`) | Claude Code |
|------|---------------------------|-------------|
| **压缩粒度** | 单次摘要（+ 可选 Turn 前缀） | 5 层渐进管道（Budget→Snip→Microcompact→Collapse→Auto-compact） |
| **Turn 处理** | 双摘要保护进行中的 Turn | 懒降级，可能中断 tool 执行 |
| **增量更新** | 显式 `UPDATE_SUMMARIZATION_PROMPT` | 隐式在模型层处理 |
| **文件追踪** | 结构化 `CompactionDetails` | 无显式文件列表 |
| **Token 估算** | chars/4 启发式 | 可能使用 tiktoken |
| **系统提示** | 极简（< 1k tokens） | ~19.6k tokens |
| **Session 结构** | JSONL 树结构，支持分支/恢复 | 线性历史 |

---

## 十、源码关键函数索引

| 函数 | 作用 | 所在阶段 |
|------|------|----------|
| `prepareCompaction()` | 准备压缩所需数据（切割点、待摘要消息、文件操作） | Stage 1 |
| `findCutPoint()` | 从后往前累积 Token，找到合法切割点 | Stage 2 |
| `findValidCutPoints()` | 筛选合法切割点索引 | Stage 2 |
| `findTurnStartIndex()` | 查找当前 Turn 的起始 user 消息 | Stage 2 |
| `estimateTokens()` | 消息级 Token 估算（chars/4 启发式） | Stage 2 |
| `estimateContextTokens()` | 上下文级 Token 估算 | Stage 2 |
| `generateSummary()` | 生成历史摘要（首次或增量） | Stage 3 |
| `generateTurnPrefixSummary()` | 生成当前 Turn 前缀摘要 | Stage 4 |
| `compact()` | 合并双摘要 + 文件操作记录 | Stage 5 |
| `extractFileOperations()` | 提取文件读写操作 | Stage 1/5 |
| `calculateContextTokens()` | 从 usage 计算总 Token | 全局 |
| `shouldCompact()` | 判断是否触发压缩 | 全局 |

---

## 十一、设计哲学总结

1. **最小破坏**：用双摘要机制在压缩历史的同时，保护当前进行中的 Turn 不被打断
2. **增量更新**：避免重复生成完整摘要，只更新变化部分
3. **文件追踪**：确保工程上下文（文件操作轨迹）不因压缩而丢失
4. **保守估算**：chars/4 启发式高估 Token 数，宁可多压缩也不溢出
5. **Turn 边界神圣**：绝不切割 toolResult 与 assistant 的配对关系