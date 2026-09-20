# python3
# Create Date: 2026-09-15
# Author: Scc_hy
# Func: 习题 4.1/4.2/4.3 端到端验证（检索智能路由 / 智能学习报告 / 多用户隔离）
# Run : /home/scc/anaconda3/envs/LLM/bin/python3 test_exercise4.py
# 注意: 需要 chap08/.env（LLM + embedding）；会产生少量 DashScope/DeepSeek 调用
# ===========================================================================================
import json
import time

from bot import PDFLearningAssistant

# 每次运行用全新用户：路由探测的记忆命中数会被历史 QA 抬高（旧问题与
# 新问题语义相近 -> 双侧强证据 -> hybrid），fresh user 才能得到干净的路由判定
USER = f"ex4_demo{int(time.time())}"
PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'✅' if ok else '❌'} {name}" + (f"  [{detail}]" if detail else ""))


print("=" * 70)
print("0) 初始化（嵌入式 Qdrant + DashScope embedding）")
bot = PDFLearningAssistant(user_id=USER)
print("  rag ready:", bot.ready, "| init_error:", bot.init_error or "-")
check("RAG 初始化", bot.ready, bot.init_error)

# 给知识库放一点资料（演示用小文本，避免大 PDF 的入库耗时）
kb = (
    "检索增强生成（RAG）通过先检索知识库再生成答案来缓解大模型幻觉。"
    "多查询扩展（MQE）把一个问题改写成多个视角的子查询以提升召回。"
    "HyDE 先让 LLM 生成假设文档，再用它检索，缓解词汇不匹配。"
)
print("\n" + "=" * 70)
print("1) 习题 4.1 检索智能路由")
bot.add_text(kb, document_id="ex4_rag_basics")

cases = [
    # (问题, 期望路由)
    ("RAG 是什么技术？它解决什么问题？", "rag"),          # 意图规则：明确指向文档内容？无“文档”字样 -> 探测：知识库强命中
    ("我之前都问了哪些问题？", "memory"),                   # 意图规则：指向个人历史
    ("MQE 和 HyDE 有什么区别？", None),                     # 探测仲裁（预期 rag：知识库里有）
    ("上次学习报告里我掌握得怎么样？", "memory"),            # 意图规则
]
for q, expect in cases:
    d = bot.route_question(q)
    tag = f"路由[{q[:16]}...] -> {d['route']}"
    if expect:
        check(tag, d["route"] == expect, d["reason"])
    else:
        print(f"  ℹ️ {tag}  [{d['reason']}]")

# 三条路径真实走一遍（各一次 LLM 调用）
for q, route in [
    ("文档里说 RAG 解决什么问题？", "rag"),
    ("我之前问了什么问题？", "memory"),
    ("结合知识库和我之前的学习记录，MQE 是什么？", "hybrid"),
]:
    r = bot.ask(q, route=route, limit=3)
    check(f"ask({route})",
          r["success"] and r.get("route") == route,
          f"{r.get('reason', r.get('message', ''))[:40]}")
    if r["success"]:
        print(f"     回答摘要: {r['answer'][:60]}...")

print("\n" + "=" * 70)
print("2) 习题 4.2 智能学习报告")
# 制造盲点信号：同一主题反复问 + 疑难词 + 不记笔记
for q in ["RAG 到底怎么工作的，还是不太懂",
          "RAG 的原理没明白，再讲一遍",
          "MQE 和 HyDE 有什么区别？没懂"]:
    bot.ask(q, route="rag", limit=3)
bot.add_note("RAG = 检索增强生成，先检索再生成", concept="RAG")   # 只沉淀 RAG，不沉淀 MQE/HyDE

report = bot.generate_report(with_llm=True)
check("报告结构完整",
      all(k in report for k in ("trajectory", "topics", "blind_spots", "recommendations", "llm_summary")))
print(f"  轨迹段数: {len(report['trajectory'])} | 主题数: {len(report['topics'])} | "
      f"盲点数: {len(report['blind_spots'])}")
check("轨迹还原出本会话问答", any("MQE" in str(seg) or "RAG" in str(seg) for seg in report["trajectory"]))
check("盲点识别到未沉淀主题", any(not s["noted"] for s in report["blind_spots"]),
      "盲点主题: " + ", ".join(s["topic"][:12] for s in report["blind_spots"]))
check("LLM 叙事生成", bool(report["llm_summary"]))
print("  推荐建议:")
for rec in report["recommendations"]:
    print("   -", rec[:70])
print("  LLM 叙事前 200 字:", (report["llm_summary"] or "")[:200].replace("\n", " "))

print("\n" + "=" * 70)
print("3) 习题 4.3 多用户隔离")
iso = bot.verify_isolation()
for k in ("rag_isolated", "memory_isolated"):
    if iso[k] is None:
        print(f"  ℹ️ {k} 检查被跳过（RAG 不可用或探针创建失败）")
check("RAG 隔离（探针用户检索不到我的知识）", iso["rag_isolated"] is not False)
check("记忆隔离（探针用户检索不到我的记忆）", iso["memory_isolated"] is True)
check("自检总判定", iso["passed"] is not False, iso["message"])

# collection 模式再验一次（每用户独立 collection）
bot_c = PDFLearningAssistant(user_id=f"{USER}_c", isolation_mode="collection")
if bot_c.ready:
    bot_c.add_text("collection 模式私有知识：独一无二的紫斑牡丹。", document_id="ex4_c")
    iso_c = bot_c.verify_isolation()
    check("collection 模式 RAG 隔离", iso_c["rag_isolated"] is True)
else:
    print("  ℹ️ collection 模式初始化失败，跳过:", bot_c.init_error)

print("\n" + "=" * 70)
print(f"结果: {len(PASS)} 通过 / {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    raise SystemExit(1)
