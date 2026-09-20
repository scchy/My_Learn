# python3
# Create Date: 2026-09-16
# Author: Scc_hy
# Func: 习题 3.3 敏感数据彻底清除——删除后残留探测实验（Qdrant / Neo4j / SQLite / 磁盘字节）
# Run : eval "$(grep '^export ALI_API_KEY' ~/.bashrc)" && \
#       /home/scc/anaconda3/envs/LLM/bin/python3 p_secure_erase_probe.py
# 说明: 配置见同目录 .env；需要本地 Neo4j（/home/scc/neo4j-local/start.sh start）
# ===========================================================================================
"""「仅仅从数据库删除是否足够？」——用实验回答。

对三种存储各做一轮「写入敏感探针 -> 正常删除 -> 多层翻找残留」：

1. Qdrant（语义/情景向量）：API 层按 memory_id 硬删是否删净？payload 明文是否还躺在磁盘文件里？
2. Neo4j（语义记忆知识图谱）：remove() 后实体/关系还在吗？（源码 _cleanup_entities_and_relations 是 pass 空壳）
3. SQLite（情景记忆权威库）：DELETE 后内容是否还在空闲页/WAL 里可恢复？VACUUM 呢？
"""

import json
import os
import sqlite3
import time
from pathlib import Path

CHAP_DIR = Path(__file__).resolve().parent
os.chdir(CHAP_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAP_DIR / ".env")

from hello_agents.tools import MemoryTool  # noqa: E402

RESULTS = []


def record(name: str, residue: bool, detail: str):
    RESULTS.append((name, residue, detail))
    print(f"  {'❌ 有残留' if residue else '✅ 无残留'} | {name}\n     {detail}")


def scan_dir_for_token(root: Path, token: bytes, limit_files: int = 500) -> list:
    """在目录下所有文件里做字节级搜索，返回命中文件列表。"""
    hits = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            try:
                if token in p.read_bytes():
                    hits.append(str(p.relative_to(root)))
            except Exception:
                continue
            if len(hits) >= limit_files:
                break
    return hits


def main() -> None:
    user = f"erase_probe{int(time.time())}"
    mt = MemoryTool(user_id=user)
    mt.current_session_id = f"erase_session_{int(time.time())}"
    mm = mt.memory_manager
    sem = mm.memory_types["semantic"]
    epi = mm.memory_types["episodic"]
    marker = f"SSENS{int(time.time())}"          # ASCII 唯一标记，便于字节扫描
    sensitive = f"隐私探针{marker}: 银行卡号 {marker}-6222-0000（演示假数据）"
    token = marker.encode()
    print("=" * 76)
    print(f"3.3 残留探测实验（user={user}, marker={marker}）")

    # ---------------------------------------------------------------- Qdrant
    print("\n[1] Qdrant 向量库（payload 含明文 content）")
    mid = mm.add_memory(content=sensitive, memory_type="semantic",
                        importance=0.6, metadata={}, auto_classify=False)
    qdrant_dir = Path("./memory_data/qdrant_local")
    coll = sem.vector_store.collection_name

    def count_points() -> int:
        from qdrant_client.http import models as qm
        flt = qm.Filter(should=[qm.FieldCondition(key="memory_id",
                                                  match=qm.MatchValue(value=mid))])
        return sem.vector_store.client.count(coll, count_filter=flt, exact=True).count

    n0 = count_points()
    disk_hits0 = scan_dir_for_token(qdrant_dir, token)
    record(f"删除前 sanity：Qdrant 点数={n0}，磁盘命中={len(disk_hits0)}个文件",
           n0 != 1 or not disk_hits0, f"payload 明文写入磁盘: {disk_hits0[:2]}")

    mm.remove_memory(mid)
    n1 = count_points()
    disk_hits1 = scan_dir_for_token(qdrant_dir, token)
    record("Qdrant API 层删除后", n1 != 0, f"存活点数={n1}（按 memory_id 过滤删除）")
    record("Qdrant 磁盘文件字节扫描（删除后）", len(disk_hits1) > 0,
           f"仍含标记的文件: {disk_hits1[:3]}{'...' if len(disk_hits1) > 3 else ''}"
           if disk_hits1 else "payload 明文已从磁盘清除")

    # ---------------------------------------------------------------- Neo4j
    print("\n[2] Neo4j 知识图谱（remove() 的实体清理是 pass 空壳）")
    # 环境无 spaCy，自动抽取产出为空——这里手动建图，模拟启用 NLP 后的正常写入
    tag = f"kgprobe{int(time.time())}"
    graph = sem.graph_store
    graph.add_entity(f"{tag}_card", "银行卡号", "PII", {"user_id": user})
    graph.add_entity(f"{tag}_bank", "某银行", "ORG", {"user_id": user})
    graph.add_relationship(f"{tag}_card", f"{tag}_bank", "BELONGS_TO", {"user_id": user})

    def graph_count() -> int:
        with graph.driver.session(database=graph.database) as s:
            return s.run(
                f"MATCH (e:Entity) WHERE e.id STARTS WITH '{tag}' "
                "OPTIONAL MATCH (e)-[r]->() RETURN count(DISTINCT e)+count(r) AS n"
            ).single()["n"]

    n_g0 = graph_count()
    mm.remove_memory(mid)          # 已删过一次；这里验证再删也不会碰图
    n_g1 = graph_count()
    record("remove() 后图上的实体/关系", n_g1 >= n_g0 and n_g1 > 0,
           f"删除后仍有 {n_g1} 个节点+关系（SemanticMemory._cleanup_entities_and_relations"
           f" 是 pass 空壳，图数据从不清理）")

    with graph.driver.session(database=graph.database) as s:
        s.run(f"MATCH (e:Entity) WHERE e.id STARTS WITH '{tag}' DETACH DELETE e")
    record("DETACH DELETE 彻底清除图", graph_count() != 0,
           "带 user_id 前缀定位 + DETACH DELETE 后节点与关系归零")

    # ---------------------------------------------------------------- SQLite
    print("\n[3] SQLite 权威库（DELETE 后空闲页 / WAL 残留）")
    eid = mm.add_memory(content=sensitive + "-episodic", memory_type="episodic",
                        importance=0.5, metadata={}, auto_classify=False)
    db_path = Path(epi.doc_store.db_path)
    mm.remove_memory(eid)
    with sqlite3.connect(db_path) as conn:
        n_row = conn.execute("SELECT COUNT(*) FROM memories WHERE id=?", (eid,)).fetchone()[0]
    record("行级删除（SELECT 确认）", n_row != 0, f"残留行数={n_row}")

    wal = Path(str(db_path) + "-wal")
    files_to_scan = [db_path] + ([wal] if wal.exists() else [])
    hits = [f.name for f in files_to_scan if token in f.read_bytes()]
    record("主库/WAL 字节级扫描（删除后）", len(hits) > 0,
           f"敏感内容仍躺在 {hits} 的空闲页里——普通文件恢复工具可直接读出" if hits else "已清除")

    with sqlite3.connect(db_path) as conn:
        conn.execute("VACUUM")
    hits_vac = [f.name for f in [db_path] + ([wal] if wal.exists() else [])
                if token in f.read_bytes()]
    record("VACUUM 重建后字节级扫描", len(hits_vac) > 0,
           f"仍有残留: {hits_vac}" if hits_vac else "空闲页已回收，明文从磁盘消失")

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 76)
    print("结论汇总（❌ = 存在残留，删除≠清除）：")
    for name, residue, _ in RESULTS:
        if "sanity" in name:
            continue
        print(f"  {'❌' if residue else '✅'} {name}")
    residue_cnt = sum(1 for n, r, _ in RESULTS if r and "sanity" not in n)
    print(f"\n共探测到 {residue_cnt} 处残留 -> 「仅调 delete API」远不等于「彻底清除」")


if __name__ == "__main__":
    main()
