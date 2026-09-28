# python3
# Create Date: 2026-09-16
# Author: Scc_hy
# Func: 习题 3.1 智能遗忘策略 + 3.2 记忆归档机制（同一生命周期框架）
# Run : eval "$(grep '^export ALI_API_KEY' ~/.bashrc)" && \
#       /home/scc/anaconda3/envs/LLM/bin/python3 p_smart_forget_archive.py
# 说明: 配置见同目录 .env（嵌入式 Qdrant + DashScope embedding）
# ===========================================================================================
"""智能遗忘 + 记忆归档。

核心思想（3.1）：遗忘不是按单一维度，而是「保持率」R = 0.5^(Δt_idle / 半衰期)——
重要性、访问频率、情感标记、关联度共同决定记忆的稳定性 S，半衰期 = S·T0；
闲置时间 Δt_idle 按指数规律侵蚀它。遗忘分 F = 1 - R，越高越该被清理。

生命周期（3.2）：遗忘分同时驱动三档决策，归档是"删除"与"保留"之间的中间态：
    F < 0.5          保留（ACTIVE）
    0.5 <= F < 0.7   归档（ARCHIVED：写 JSONL 档案 + 从各存储移除，可恢复）
    F >= 0.7         直接删除（DELETED）
    importance >= 0.9 硬保护：永不删除，至多归档
"""

import json
import math
import os
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

CHAP_DIR = Path(__file__).resolve().parent
os.chdir(CHAP_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAP_DIR / ".env")

import _qdrant_embedded  # noqa: E402,F401  # 0.2.8 补丁：QDRANT_PATH 设置时走嵌入式本地模式
from hello_agents.tools import MemoryTool  # noqa: E402

# ================================================================ 3.1 智能遗忘
# 权重与超参（一次性集中定义，调参只动这里）
W = {                       # 稳定性 S 的构成权重：行为证据 > 情感 > 结构
    "importance": 0.40,     # 显式重要性（LLM 打分 / 用户指定）
    "frequency":   0.25,    # 访问频率（复述效应）
    "emotion":     0.20,    # 情感标记（高唤醒事件优先固化——杏仁核-海马机制）
    "link":        0.15,    # 与既有记忆的关联数（能挂到知识网的记忆更稳）
}
T0_DAYS = 60.0              # 半衰期基准：S=1 的记忆半衰期 60 天
FORGET_T = 0.70             # F >= T   -> 删除
ARCHIVE_T = 0.50            # T_A <= F < T -> 归档
FREQ_FULL = 3               # 被访问 3 次记满频率分
LINK_FULL = 3               # 挂 3 条关联记满关联分
IMPORTANCE_FLOOR = 0.9      # 高重要性硬保护：永不删除，至多归档


def intelligent_forgetting_score(
    importance: float,
    idle_days: float,           # 距「最后一次访问」的天数（不是距写入！）
    access_count: int = 0,
    emotion: float = 0.0,       # 0~1，0=中性；写入时由词典+LLM双路标定
    link_count: int = 0,        # 与其他记忆的关联数（图连接/被引用次数）
    pinned: bool = False,       # 显式记忆指令（"请记住…"）
) -> Dict[str, Any]:
    """记忆遗忘评分（习题 3.1 核心函数）。

        稳定性 S = 0.40·Importance + 0.25·Freq + 0.20·Emotion + 0.15·Link
        半衰期 H = S · T0                                # T0=60 天
        保持率 R = 0.5^(idle_days / H)                   # 艾宾浩斯式指数遗忘
        遗忘分 F = 1 - R

    F 越高越应该被清理。返回中间量便于解释"为什么忘了它"。
    硬规则兜底：pinned 永不遗忘；importance>=0.9 的 F 压到 0.5 以下（至多归档）。
    """
    if pinned:
        return {"S": 1.0, "H": T0_DAYS, "R": 1.0, "F": 0.0,
                "decision": "keep", "why": "pinned（显式记忆指令）"}

    freq = min(1.0, access_count / FREQ_FULL)
    link = min(1.0, link_count / LINK_FULL)
    s = (W["importance"] * importance + W["frequency"] * freq
         + W["emotion"] * emotion + W["link"] * link)
    s = max(s, 0.05)                       # 防除零：完全不稳定的记忆半衰期 3 天
    half_life = s * T0_DAYS
    r = 0.5 ** (max(idle_days, 0.0) / half_life)
    f = 1.0 - r

    decision, why = "keep", f"F={f:.2f} < 归档阈值{ARCHIVE_T}"
    if importance >= IMPORTANCE_FLOOR and f >= ARCHIVE_T:
        f = min(f, 0.49)                   # 硬保护：重要信息只允许保留/归档
        decision, why = "archive", f"importance={importance} 触发硬保护，F 压至 {f:.2f}（永不删除）"
    elif f >= FORGET_T:
        decision, why = "delete", f"F={f:.2f} >= 删除阈值{FORGET_T}（久未访问、零情感、低重要）"
    elif f >= ARCHIVE_T:
        decision, why = "archive", f"归档阈值{ARCHIVE_T} <= F={f:.2f} < 删除阈值{FORGET_T}"
    return {"S": round(s, 3), "H": round(half_life, 1), "R": round(r, 3), "F": round(f, 3),
            "decision": decision, "why": why}


def demo_scoring() -> None:
    """构造 7 条对照记忆，展示智能遗忘 vs 纯时间/纯重要性策略的差异。"""
    print("=" * 76)
    print("3.1 智能遗忘评分演示（T0=60天，半衰期=S·T0；对照 7 条典型记忆）")
    cases = [  # (描述, kwargs)
        ("A 过敏信息：青霉素重度过敏（90天未访问，重要0.95）",
         dict(importance=0.95, idle_days=90)),
        ("B 闲聊：今天天气不错（60天未访问，重要0.2）",
         dict(importance=0.20, idle_days=60)),
        ("C 目标：备考12月软考（闲置3天，重要0.8，访问过3次）",
         dict(importance=0.80, idle_days=3, access_count=3)),
        ("D 确诊事件（闲置45天，重要0.5，情感0.9，访问1次）",
         dict(importance=0.50, idle_days=45, access_count=1, emotion=0.9)),
        ("D' 同上但【无】情感标记（情感信号缺失的对照）",
         dict(importance=0.50, idle_days=45, access_count=1, emotion=0.0)),
        ("E 临时数值：某次计算的中间结果（闲置5天，重要0.4）",
         dict(importance=0.40, idle_days=5)),
        ("F 知识：RAG=检索增强生成（闲置45天，重要0.6，访问3次，挂3条关联）",
         dict(importance=0.60, idle_days=45, access_count=3, link_count=3)),
    ]
    print(f"{'记忆':<40}{'S':>6}{'半衰期':>7}{'R':>7}{'F':>7}  决策")
    for desc, kw in cases:
        sc = intelligent_forgetting_score(**kw)
        print(f"{desc:<40}{sc['S']:>6.2f}{sc['H']:>6.0f}天{sc['R']:>7.2f}{sc['F']:>7.2f}  {sc['decision']}")
    print("-" * 76)
    print("对照实验（体现'智能'在哪，与上表逐行对应）：")
    print("  · 纯时间策略会删掉 A——重要性硬保护：至多归档，永不删除")
    print("  · 纯重要性策略会保留 B——零访问+低重要，60 天即达删除线")
    print("  · C 写入 30 天仍活跃——3 次访问抬高稳定性（复述效应）")
    print("  · D vs D'：情感标记把「确诊」从删除线拉回归档区（情绪增强记忆）")
    print("  · E 年轻记忆不急删（闲置仅 5 天，F 未过线）——避免'写入即遗忘'")


# ================================================================ 3.2 记忆归档
class MemoryArchiver:
    """episodic 记忆的归档/恢复（生命周期：ACTIVE -> ARCHIVED -> ACTIVE/DELETED）。

    档案格式：JSONL（一行一条），目录 memory_data/archive/{user_id}/YYYYMM.jsonl。
    档案只存 content + 元数据，不存向量——恢复时由当前 embedder 重算（省 ~76% 空间
    [1024 维 float64 ≈ 8KB/条 vs 文本 ~200B]，且换 embedding 模型后恢复的向量维度
    自动与库一致）。
    """

    def __init__(self, memory_tool: MemoryTool, archive_root: str = "./memory_data/archive"):
        self.mt = memory_tool
        self.mm = memory_tool.memory_manager
        self.root = Path(archive_root) / self.mm.user_id
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    # ---- 扫描：只处理 episodic（SQLite 是权威库，跨进程可读） ----
    def scan_episodic(self, now: datetime) -> List[Dict[str, Any]]:
        epi = self.mm.memory_types["episodic"]
        rows = epi.doc_store.search_memories(
            user_id=self.mm.user_id, memory_type="episodic", limit=10000)
        enriched = []
        for r in rows:
            write_age = (now - datetime.fromtimestamp(r["timestamp"])).total_seconds() / 86400.0
            props = r.get("properties") or {}
            last_access = props.get("last_accessed")
            idle_days = ((now - (datetime.fromisoformat(last_access) if isinstance(last_access, str)
                                 else datetime.fromtimestamp(r["timestamp"]))).total_seconds()) / 86400.0
            sc = intelligent_forgetting_score(
                importance=r.get("importance", 0.5),
                idle_days=idle_days,
                access_count=int(props.get("access_count", 0)),
                emotion=float(props.get("emotion", 0.0)),
                link_count=int(props.get("link_count", 0)),
                pinned=bool(props.get("pinned", False)),
            )
            # 保留判定用「闲置年龄」，归档前置过滤用「写入年龄」——
            # 刚写入的记忆即使从未访问也要给访问频率留出积累窗口
            enriched.append({**r, "write_age_days": round(write_age, 1),
                             "idle_days": round(idle_days, 1), **sc})
        return enriched

    def archive_old_memories(
        self,
        retention_days: float = 30.0,
        dry_run: bool = False,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """主入口：写入超 retention_days 的记忆按遗忘分三档处理。

            F < 0.5          保留
            0.5 <= F < 0.7   归档（写 JSONL + 从 SQLite/Qdrant/内存移除）
            F >= 0.7         直接删除
        """
        now = now or datetime.now()
        epi = self.mm.memory_types["episodic"]
        report: Dict[str, Any] = {"kept": 0, "archived": 0, "deleted": 0,
                                  "not_due": 0, "archive_file": None, "items": []}
        for r in self.scan_episodic(now):
            if r["write_age_days"] < retention_days:
                report["not_due"] += 1
                continue
            decision = r["decision"]
            if decision == "keep":
                report["kept"] += 1
            elif decision == "archive":
                if not dry_run:
                    report["archive_file"] = str(self._write_archive(r))
                    epi.remove(r["memory_id"])     # SQLite + Qdrant + 内存三处一起清
                report["archived"] += 1
            else:
                if not dry_run:
                    epi.remove(r["memory_id"])
                report["deleted"] += 1
            report["items"].append({
                "content": (r["content"] or "")[:36],
                "write_age_days": r["write_age_days"], "idle_days": r["idle_days"],
                "F": r["F"], "decision": decision})
        return report

    def _write_archive(self, row: Dict[str, Any]) -> Path:
        file = self.root / f"{datetime.now():%Y%m}.jsonl"
        record = {
            "memory_id": row["memory_id"],
            "content": row["content"],
            "memory_type": "episodic",
            "importance": row.get("importance", 0.5),
            "timestamp": row.get("timestamp"),
            "properties": row.get("properties") or {},
            "forget_score": row.get("F"),
            "archived_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return file

    # ---- 恢复：档案命中查询则重新入库（当前 embedder 重算向量） ----
    def restore(self, query: str, limit: int = 3) -> List[str]:
        """按关键词/模糊匹配找回档案，重新写回 episodic（打 restored 标记）。"""
        candidates: List[tuple] = []
        for file in sorted(self.root.glob("*.jsonl")):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    text = rec["content"]
                    hit = max((self._ratio(text, kw) for kw in query.split()), default=0.0)
                    if query in text or hit > 0.4:
                        candidates.append((hit, file, rec))
        candidates.sort(key=lambda x: -x[0])
        restored = []
        for _, file, rec in candidates[:limit]:
            props = dict(rec.get("properties") or {})
            props["restored_from"] = file.name
            self.mm.add_memory(
                content=rec["content"], memory_type="episodic",
                importance=rec.get("importance", 0.5),
                metadata={"context": props}, auto_classify=False)
            restored.append(rec["content"][:50])
        return restored


# ================================================================ 端到端验证
def main() -> None:
    demo_scoring()

    user = f"arch_demo{int(time.time())}"
    mt = MemoryTool(user_id=user)
    mt.current_session_id = f"arch_session_{int(time.time())}"
    mm = mt.memory_manager
    archiver = MemoryArchiver(mt)
    now = datetime.now()

    # 播种 5 条情景记忆（auto_classify=False 确保都进 episodic），覆盖全部三档决策：
    seeds = [  # (内容, importance, write_age_days, idle_days, visits)
        ("项目复盘：RAG 问答助手上线，首周回答准确率 87%", 0.70, 90, 3, 2),   # 最近还复习过 -> keep
        ("闲聊记录：中午吃了食堂的麻辣香锅", 0.20, 90, 90, 0),                # 久置+琐碎   -> delete
        ("重要结论：用户的论文选题定为多智能体协作", 0.92, 90, 90, 0),          # 久置+imp>=0.9 -> 硬保护归档
        ("实验记录：第二批样本的清洗结果存疑", 0.60, 90, 20, 0),               # 中间档      -> 归档
        ("刚记的：明天上午十点组会", 0.50, 0, 0, 0),                           # 新鲜        -> 不进扫描
    ]
    epi = mm.memory_types["episodic"]
    for content, imp, write_age, idle, visits in seeds:
        item_id = mm.add_memory(
            content=content, memory_type="episodic", importance=imp,
            metadata={"context": {"access_count": visits, "emotion": 0.0, "link_count": 0}},
            auto_classify=False)
        _backdate(epi, item_id, now - timedelta(days=write_age),
                  idle_days=idle, visits=visits)

    print("\n" + "=" * 76)
    print(f"3.2 归档生命周期验证（user={user}，retention=30天）")
    dry = archiver.archive_old_memories(retention_days=30, dry_run=True, now=now)
    for it in dry["items"]:
        print(f"  [dry] 写入{it['write_age_days']:>5}天前 闲置{it['idle_days']:>5}天 "
              f"F={it['F']:.2f} -> {it['decision']:<8} {it['content']}")
    print(f"  [dry] 未到期（不进扫描）: {dry['not_due']} 条")

    result = archiver.archive_old_memories(retention_days=30, dry_run=False, now=now)
    print(f"  实际执行: 保留{result['kept']} 归档{result['archived']} 删除{result['deleted']} "
          f"未到期{result['not_due']}  档案: {result['archive_file']}")

    # 验证1：归档文件落盘且含两条被归档记忆（硬保护路径 + 中间档路径）
    files = list(archiver.root.glob("*.jsonl"))
    archived_contents = "".join(f.read_text(encoding="utf-8") for f in files)
    checks = []
    checks.append(("归档文件落盘且含'论文选题'（硬保护路径）+ '实验记录'（中间档路径）",
                   bool(files) and "论文选题" in archived_contents and "实验记录" in archived_contents))

    # 验证2：被删除/归档的记忆已从权威库消失；保留的还在
    rows_left = epi.doc_store.search_memories(user_id=user, memory_type="episodic", limit=100)
    left_texts = " | ".join(r["content"] for r in rows_left)
    checks.append(("删除档已删、归档档已离库",
                   all(k not in left_texts for k in ("麻辣香锅", "论文选题", "实验记录"))))
    checks.append(("保留项仍在（项目复盘+组会）",
                   "项目复盘" in left_texts and "组会" in left_texts))
    print(f"  剩余: {left_texts}")

    # 验证3：恢复——用关键词从档案找回"论文选题"
    got = archiver.restore("论文选题")
    rows_after = epi.doc_store.search_memories(user_id=user, memory_type="episodic", limit=100)
    checks.append(("恢复：'论文选题'从档案重新入库",
                   any("论文选题" in r["content"] for r in rows_after)))
    print(f"  恢复命中: {got}")

    print("\n" + "=" * 76)
    all_ok = True
    for name, ok in checks:
        all_ok &= ok
        print(f"  {'✅' if ok else '❌'} {name}")
    print("\n全部通过" if all_ok else "\n存在失败项！")


def _backdate(epi, memory_id: str, ts: datetime, idle_days: float, visits: int) -> None:
    """把 SQLite 里的时间戳改成'很久以前写入'，并写入访问统计（验证用）。"""
    import sqlite3
    db_path = epi.doc_store.db_path
    props = {"access_count": visits, "emotion": 0.0, "link_count": 0,
             "last_accessed": (datetime.now() - timedelta(days=idle_days)).isoformat(timespec="seconds")}
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE memories SET timestamp=?, properties=? WHERE id=?",
            (int(ts.timestamp()), json.dumps(props, ensure_ascii=False), memory_id))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
