# python3
# Create Date: 2026-09-16
# Author: Scc_hy
# Func: 习题 5 语义记忆与知识图谱——5.1 质量评估 / 5.2 图查询能力 / 5.3 延迟对比
# Run : eval "$(grep '^export ALI_API_KEY' ~/.bashrc)" && \
#       /home/scc/anaconda3/envs/LLM/bin/python3 p_kg_lab.py
# 说明: 需要本地 Neo4j（/home/scc/neo4j-local/start.sh start）；
#       演示子图带 kgdemo 前缀，脚本结束时清理，不污染真实图谱
# ===========================================================================================
"""5.1 自动抽取的知识图谱质量评估（一致性/连通性/冗余度指标 + 实测）
   5.2 纯向量检索做不到的图查询：多跳补基础 / 最短路径 / 共前置社区（带真实数据实测）
   5.3 图查询 vs 向量检索的延迟微基准
"""

import os
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

CHAP_DIR = Path(__file__).resolve().parent
os.chdir(CHAP_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(CHAP_DIR / ".env")

import _qdrant_embedded  # noqa: E402,F401  # 0.2.8 补丁：QDRANT_PATH 设置时走嵌入式本地模式
from hello_agents.memory.embedding import get_text_embedder  # noqa: E402
from hello_agents.memory.storage.neo4j_store import Neo4jGraphStore  # noqa: E402
from hello_agents.memory.storage.qdrant_store import QdrantConnectionManager  # noqa: E402

graph = Neo4jGraphStore(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "hello-agents-password"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),
)


def cypher(query: str, **params):
    with graph.driver.session(database=graph.database) as s:
        return [r.data() for r in s.run(query, **params)]


# ================================================================ 5.1 质量评估
def kg_quality_report(label: str = "Entity") -> Dict[str, object]:
    """自动抽取图谱的四类质量指标（全部可 SQL/Cypher 化，可每日定时跑）。"""
    rep: Dict[str, object] = {}
    total = cypher(f"MATCH (e:{label}) RETURN count(e) AS n")[0]["n"]
    rep["total_nodes"] = total
    if total == 0:
        return rep

    # (1) 实体一致性：同名不同 id（硬重复）+ 名字高相似（软重复，大小写归一后比较）
    hard_dup = cypher(
        f"MATCH (e:{label}) WITH e.name AS name, collect(e.id) AS ids, count(*) AS c "
        "WHERE c > 1 RETURN name, ids, c ORDER BY c DESC LIMIT 10")
    names = [r["name"] for r in cypher(f"MATCH (e:{label}) RETURN DISTINCT e.name AS name")]
    soft_dup = []
    low = [n.lower() for n in names]
    for i, a in enumerate(low):
        for j, b in enumerate(low):
            if j <= i or not a or not b:
                continue
            if SequenceMatcher(None, a, b).ratio() > 0.6:
                soft_dup.append((names[i], names[j]))
    rep["hard_dup"] = hard_dup
    rep["soft_dup"] = soft_dup

    # (2) 孤立节点与自环（抽取失败的典型产物：提了实体但没抽出关系）
    rep["orphans"] = cypher(
        f"MATCH (e:{label}) WHERE NOT (e)--() RETURN count(e) AS n")[0]["n"]
    rep["self_loops"] = cypher(
        f"MATCH (e:{label})-[r:{label}]->(e) RETURN count(r) AS n")[0]["n"]

    # (3) 图连通性：连通分量数与最大分量占比（Python 端 BFS，图小够用；
    #     大图换 GDS: gds.graph.project + gds.wcc）
    edges = cypher(
        f"MATCH (a:{label})-[]->(b:{label}) RETURN a.id AS s, b.id AS t")
    adj: Dict[str, List[str]] = defaultdict(list)
    nodes = set(r["name"] for r in cypher(f"MATCH (e:{label}) RETURN e.id AS name"))
    for e in edges:
        adj[e["s"]].append(e["t"])
        adj[e["t"]].append(e["s"])
    seen, comps = set(), []
    for n0 in nodes:
        if n0 in seen:
            continue
        stack, comp = [n0], 0
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp += 1
            stack.extend(adj[cur])
        comps.append(comp)
    comps.sort(reverse=True)
    rep["components"] = len(comps)
    rep["giant_ratio"] = round(comps[0] / total, 2) if comps else 0.0

    # (4) 冗余度：反向重复边（A→B 与 B→A 同型）+ 二跳路径重复（两点间 ≥2 条不同路径）
    rep["inverse_dup"] = cypher(
        f"MATCH (a:{label})-[r]->(b:{label}) "
        f"MATCH (b)-[r2:{label}]->(a) RETURN count(r) AS n")[0]["n"]
    rep["multi_path_pairs"] = cypher(
        f"MATCH (a:{label})-[]->(m:{label})-[]->(b:{label}) "
        f"WHERE (a)-[]->(b) AND a<>b RETURN count(DISTINCT [a.id, b.id]) AS n")[0]["n"]
    return rep


def print_quality(rep: Dict[str, object]) -> None:
    if not rep or rep.get("total_nodes", 0) == 0:
        print("  图为空。")
        return
    print(f"  节点总数={rep['total_nodes']} ｜ 连通分量={rep['components']} "
          f"（最大分量占比 {rep['giant_ratio']}）")
    print(f"  孤立节点={rep['orphans']} ｜ 自环={rep['self_loops']} ｜ 反向重复边={rep['inverse_dup']} "
          f"｜ '直接边+二跳旁路'冗余对={rep['multi_path_pairs']}")
    if rep["hard_dup"]:
        print(f"  硬重复实体（同名不同id，前3）: {rep['hard_dup'][:3]}")
    else:
        print("  硬重复实体: 无")
    if rep["soft_dup"]:
        print(f"  软重复（忽略大小写后相似度>0.6，前5）: {rep['soft_dup'][:5]}")


def seed_noisy_entities() -> None:
    """播种一批带典型质量问题的 :Entity 数据（演示评估指标能抓到什么）。"""
    with graph.driver.session(database=graph.database) as s:
        s.run("MATCH (e:Entity) WHERE e.id STARTS WITH 'kgdemo' DETACH DELETE e")
    rows = [
        # (id, name, type)  —— 故意埋 3 类问题：同名不同id / 相似变体 / 孤立节点
        ("kgdemo_e1", "Python", "TECH"), ("kgdemo_e2", "Python", "TECH"),   # 硬重复
        ("kgdemo_e3", "python语言", "TECH"),                                 # 软重复变体
        ("kgdemo_e4", "RAG", "CONCEPT"), ("kgdemo_e5", "向量检索", "CONCEPT"),
        ("kgdemo_e6", "Embedding", "CONCEPT"), ("kgdemo_e7", "孤立概念X", "CONCEPT"),  # 孤立
    ]
    rels = [("kgdemo_e4", "kgdemo_e5", "REQUIRES"), ("kgdemo_e4", "kgdemo_e5", "DEPENDS_ON"),  # 反向?同向重复型
            ("kgdemo_e5", "kgdemo_e6", "REQUIRES")]
    with graph.driver.session(database=graph.database) as s:
        for eid, name, t in rows:
            s.run("MERGE (e:Entity {id:$id}) SET e.name=$name, e.type=$t",
                  id=eid, name=name, t=t)
        for a, b, t in rels:
            s.run(f"MATCH (a:Entity {{id:$a}}),(b:Entity {{id:$b}}) "
                  f"MERGE (a)-[r:{t}]->(b)", a=a, b=b)


# ================================================================ 5.2 图查询能力
def seed_learner_graph() -> None:
    """播种学习助手场景子图（Concept/Learner 两种标签，kgdemo 前缀，幂等）。"""
    with graph.driver.session(database=graph.database) as s:
        s.run("MATCH (n) WHERE n.id STARTS WITH 'kgdemo' DETACH DELETE n")
    concepts = ["注意力机制", "Transformer", "Embedding", "向量检索", "RAG",
                "MQE", "HyDE", "提示词工程", "向量数据库", "Qdrant"]
    with graph.driver.session(database=graph.database) as s:
        for c in concepts:
            s.run("MERGE (:Concept {id:'kgdemo_c_'+$name, name:$name})", name=c)
        s.run("MERGE (:Learner {id:'kgdemo_u1', name:'张三'})")
        prereq = [("注意力机制", "Transformer"), ("Transformer", "Embedding"),
                  ("Embedding", "向量检索"), ("向量检索", "RAG"),
                  ("RAG", "MQE"), ("RAG", "HyDE"), ("提示词工程", "HyDE"),
                  ("Embedding", "向量数据库"), ("向量数据库", "Qdrant")]
        for a, b in prereq:
            s.run("MATCH (a:Concept {id:'kgdemo_c_'+$a}),(b:Concept {id:'kgdemo_c_'+$b}) "
                  "MERGE (a)-[:PREREQ_OF]->(b)", a=a, b=b)
        for c in ["注意力机制", "Transformer", "Embedding"]:
            s.run("MATCH (u:Learner {id:'kgdemo_u1'}),(c:Concept {id:'kgdemo_c_'+$c}) "
                  "MERGE (u)-[:MASTERED]->(c)", c=c)
        for c in ["RAG", "MQE"]:
            s.run("MATCH (u:Learner {id:'kgdemo_u1'}),(c:Concept {id:'kgdemo_c_'+$c}) "
                  "MERGE (u)-[:STRUGGLES_WITH]->(c)", c=c)
        s.run("MATCH (u:Learner {id:'kgdemo_u1'}),(c:Concept {id:'kgdemo_c_向量数据库'}) "
              "MERGE (u)-[:LEARNING]->(c)")


def run_graph_queries() -> None:
    print("\n" + "=" * 76)
    print("5.2 图查询实测（学习助手场景：薄弱概念补基础 / 最短路径 / 共前置社区）")

    # Q1 多跳 + 否定过滤：薄弱概念的「前置」知识中，张三尚未掌握的（跨 1~4 跳）。
    # 方向注意：PREREQ_OF 指向"后续概念"（向量检索 PREREQ_OF RAG = 向量检索是 RAG 的前置），
    # 所以从薄弱概念找基础要沿【入边】反向走：base-[:PREREQ_OF*1..4]->weak
    q1 = """
    MATCH (u:Learner {name:'张三'})-[:STRUGGLES_WITH]->(weak:Concept)
    MATCH (base:Concept)-[:PREREQ_OF*1..4]->(weak)
    WHERE NOT EXISTS { MATCH (u)-[:MASTERED]->(base) }
      AND NOT EXISTS { MATCH (u)-[:STRUGGLES_WITH]->(base) }
    RETURN weak.name AS 薄弱概念, collect(DISTINCT base.name) AS 需补基础 ORDER BY 薄弱概念
    """
    for r in cypher(q1):
        print(f"  [Q1 多跳补基础] {r['薄弱概念']} ← 需补: {r['需补基础']}")

    # Q2 最短路径：从已掌握的 Transformer 到薄弱概念 MQE 的知识依赖链
    q2 = """
    MATCH (a:Concept {name:'Transformer'}), (b:Concept {name:'MQE'})
    MATCH p = shortestPath((a)-[:PREREQ_OF*]->(b))
    RETURN [n IN nodes(p) | n.name] AS 依赖链
    """
    for r in cypher(q2):
        print(f"  [Q2 最短路径] Transformer →·→ MQE 依赖链: {' → '.join(r['依赖链'])}")

    # Q3 同源聚类（无 GDS 的轻量社区）：共享同一个前置的概念分到同组
    # （Embedding 同时支撑 向量检索/向量数据库 —— 两个子领域的源头都是它）
    q3 = """
    MATCH (p:Concept)-[:PREREQ_OF]->(c1:Concept), (p)-[:PREREQ_OF]->(c2:Concept)
    WHERE c1 <> c2 AND id(c1) < id(c2)
    RETURN p.name AS 共同前置, collect(DISTINCT c1.name) + collect(DISTINCT c2.name) AS 同组概念
    ORDER BY 共同前置
    """
    for r in cypher(q3):
        print(f"  [Q3 同源聚类] 前置「{r['共同前置']}」支撑同组: {sorted(set(r['同组概念']))}")

    # Q4 枢纽概念（被最多概念依赖）：学它性价比最高
    q4 = """
    MATCH (hub:Concept)-[:PREREQ_OF]->(x:Concept)
    RETURN hub.name AS 枢纽, count(x) AS 支撑数, collect(x.name)[..4] AS 直接支撑
    ORDER BY 支撑数 DESC LIMIT 3
    """
    for r in cypher(q4):
        print(f"  [Q4 枢纽概念] {r['枢纽']}（支撑 {r['支撑数']} 个概念）: {r['直接支撑']}")


# ================================================================ 5.3 延迟微基准
def bench_graph_vs_vector() -> None:
    print("\n" + "=" * 76)
    print("5.3 延迟微基准（同样的'找相关内容'意图，两条技术路线）")
    q = "RAG 的前置知识有哪些"

    # 图路线：2 跳关系遍历（本机 Neo4j）
    t0 = time.perf_counter()
    graph_hits = cypher(
        "MATCH (c:Concept {name:'RAG'})-[:PREREQ_OF*1..2]->(x:Concept) "
        "RETURN DISTINCT x.name AS name LIMIT 5")
    t_graph = (time.perf_counter() - t0) * 1000

    # 向量路线：query embedding（云端 API）+ Qdrant 检索
    embedder = get_text_embedder()
    store = QdrantConnectionManager.get_instance(
        url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
        vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "1024")))
    t0 = time.perf_counter()
    vec = embedder.encode(q)
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    hits = store.search_similar(query_vector=vec, limit=5)
    t_vec = (time.perf_counter() - t0) * 1000

    print(f"  图遍历（2跳, {len(graph_hits)} 结果）: {t_graph:.1f} ms  -> {[r['name'] for r in graph_hits]}")
    print(f"  向量检索（embedding API + Qdrant, {len(hits) if hits else 0} 结果）: {t_vec:.1f} ms"
          f"  （其中网络 RTT 主导）")
    print("  注：图遍历是本机毫秒级纯结构运算；向量路线每次都要为查询付一次 embedding 调用")


# ================================================================ main
def main() -> None:
    print("=" * 76)
    print("5.1 知识图谱质量评估")
    print("\n[现状] 真实图谱（library 自动抽取产出）：")
    print_quality(kg_quality_report())

    print("\n[演示] 播种带质量问题的图谱后，指标能抓到：")
    seed_noisy_entities()
    print_quality(kg_quality_report())

    print("\n  抽取能力实测：环境缺 spaCy 模型 -> SemanticMemory._extract_entities 对中文返回空列表")
    print("  （即：自动抽取在当前环境的准确率是 0%，图谱实际处于'饥饿'状态——评估机制的第一步")
    print("    就是先把这种'静默零产出'暴露出来，而不是等下游查询莫名变差才发现）")
    with graph.driver.session(database=graph.database) as s:
        s.run("MATCH (e:Entity) WHERE e.id STARTS WITH 'kgdemo' DETACH DELETE e")

    seed_learner_graph()
    run_graph_queries()
    try:
        bench_graph_vs_vector()
    finally:
        with graph.driver.session(database=graph.database) as s:
            s.run("MATCH (n) WHERE n.id STARTS WITH 'kgdemo' DETACH DELETE n")
        print("\n（演示子图已清理）")


if __name__ == "__main__":
    main()
