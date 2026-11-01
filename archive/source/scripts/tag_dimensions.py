# -*- coding: utf-8 -*-
"""为冻结题库补标 16 个检索维度，并生成抽样题单。

维度定义见 research/retrieval_evaluation_20261101/完整检测方案.md 第 4 节。
标签来源：题型 → 维度 的确定性映射 + 依赖可回答性/跳数/来源格式的规则。
只读题目标签，不读模型答案或分数；留种子与题单 SHA。
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "eval_bench" / "versions" / "20261030_v2"
OUT = ROOT / "research" / "retrieval_repair_20261101" / "proposed_bank"

DIMENSIONS = [
    "01_精确定位", "02_同义与别名", "03_模糊需求定位", "04_两跳检索",
    "05_三跳以上", "06_汇总去重", "07_数值与单位", "08_时间推断",
    "09_历史与生效版本", "10_冲突判断", "11_多条件筛选", "12_表格与跨表",
    "13_跨格式取证", "14_长文与干扰", "15_多轮检索更新", "16_无答案与证据不足",
]

# 题型 → 主维度。题干自带的补充维度在 assign_dimensions 里叠加。
TYPE_DIM = {
    "fact_lookup": "01_精确定位",
    "numeric_lookup": "07_数值与单位",
    "list_enumeration": "06_汇总去重",
    "multi_hop": "05_三跳以上",
    "cross_doc": "04_两跳检索",
    "person_relation": "04_两跳检索",
    "temporal": "08_时间推断",
    "version_compare": "09_历史与生效版本",
    "constraint_check": "11_多条件筛选",
    "aggregation": "06_汇总去重",
    "conflict_detect": "10_冲突判断",
    "cross_doc_composition": "06_汇总去重",
    "unit_conversion": "07_数值与单位",
    "rates_and_denominators": "07_数值与单位",
    "temporal_version": "09_历史与生效版本",
    "entity_deduplication": "02_同义与别名",
    "evidence_sufficiency": "16_无答案与证据不足",
    "conflict_resolution": "10_冲突判断",
    "structured_output": "14_长文与干扰",
    "dialogue_state": "15_多轮检索更新",
}

FORMAT_DIM = {          # 跨格式取证：涉及多格式文档
    "csv": "13_跨格式取证",
    "xlsx": "12_表格与跨表",
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_bank() -> tuple[list[dict], dict[str, dict]]:
    qs = [json.loads(l) for l in (BANK / "question_bank.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ans = {json.loads(l)["qid"]: json.loads(l)
           for l in (BANK / "answer_bank.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    # 合并补建题（20261101_augment_v1）：同一套字段结构，直接并入。
    aq = ROOT / "research" / "retrieval_evaluation_20261101" / "bank" / "augment_questions.jsonl"
    aa = ROOT / "research" / "retrieval_evaluation_20261101" / "bank" / "augment_answers.jsonl"
    if aq.exists() and aa.exists():
        for line in aq.read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                q.setdefault("benchmark_version", "20261101_augment_v1")
                qs.append(q)
        for line in aa.read_text(encoding="utf-8").splitlines():
            if line.strip():
                a = json.loads(line)
                ans[a["qid"]] = a
    return qs, ans


def load_formats() -> dict[str, str]:
    """doc_id → 文件格式（从语料 manifest 取，避免靠 doc_id 后缀猜）。"""
    out = {}
    manifest = ROOT / "enterprise_rag" / "corpus" / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        out[m["doc_id"]] = m.get("format", "")
    return out


FORMATS = load_formats()


def doc_format(doc_id: str) -> str:
    if doc_id in FORMATS:
        return FORMATS[doc_id]
    return doc_id.rsplit(".", 1)[-1] if "." in doc_id else ""


def assign(q: dict, ans: dict) -> str:
    """为单题分配 16 维度之一（主维度，按确定性规则）。"""
    t = q["question_type"]
    dim = TYPE_DIM.get(t)
    if dim is None:
        dim = "01_精确定位"

    hops = q.get("multi_hop_depth", 1)
    if dim == "04_两跳检索" and hops >= 3:
        dim = "05_三跳以上"

    # 跨格式：答案依据横跨 ≥3 种文件格式
    fmts = {doc_format(e.get("doc_id", "")) for e in ans.get("evidence", [])}
    fmts.discard("")
    if len(fmts) >= 3:
        dim = "13_跨格式取证"
    elif fmts & {"csv", "xlsx"}:
        dim = "12_表格与跨表"

    # 证据不足 / 不可判定：答案含明确否定或不可判定表述
    a = ans.get("answer", "")
    if any(w in a for w in ("不能判断", "不能断定", "无法判断", "无法确定",
                            "文档未", "未记载", "无法从", "不可判定", "未披露")):
        dim = "16_无答案与证据不足"

    # 长文与干扰：依据条数多（需在长材料中定位）
    if len(ans.get("evidence", [])) >= 5 and dim in ("01_精确定位", "05_三跳以上"):
        dim = "14_长文与干扰"

    # 同义与别名：题干含旧称/简称/口语线索
    qtext = q["question"]
    if any(w in qtext for w in ("简称", "旧称", "又称", "俗称", "别名", "口语", "即指", "以下简称")):
        dim = "02_同义与别名"

    # 模糊需求定位：题干不含具体文档名/章节，只描述需求
    if any(w in qtext for w in ("哪份", "哪些资料", "哪个文件", "哪里能找到", "找哪")):
        dim = "03_模糊需求定位"

    # 多轮：题干自带对话历史
    if q.get("evaluation_track") == "context_reasoning" and q["question_type"] == "dialogue_state":
        dim = "15_多轮检索更新"

    return dim


def main() -> int:
    qs, ans = load_bank()
    rows = []
    for q in qs:
        a = ans.get(q["qid"], {})
        # 补建题自带 dimension（人工按维度定义判定），优先采用；
        # 原 1126 题无此标签，按确定性规则推导。
        dim = q.get("dimension") or assign(q, a)
        rows.append({
            **q,
            "qid": q["qid"],
            "dimension": dim,
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "multi_hop_depth": q.get("multi_hop_depth", 1),
            "evaluation_track": q.get("evaluation_track", ""),
            "expected_doc_ids": q.get("expected_doc_ids", []),
            "source_formats": sorted({doc_format(e.get("doc_id", "")) for e in a.get("evidence", []) if doc_format(e.get("doc_id", ""))}),
            "answerability": "needs_clarification" if "不能判断" in a.get("answer", "") or "澄清" in a.get("answer", "") else "answerable",
            "dimension_review": "provisional_not_capability_validation",
        })

    OUT.mkdir(parents=True, exist_ok=True)
    tagged = OUT / "question_bank_dimensioned.jsonl"
    tagged.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # 同时把补建题的原始题面合并进 dimensioned 产物，供执行器按维度抽样时取到题干。

    counts = Counter(r["dimension"] for r in rows)
    print("=== 16 维度分布 ===")
    for d in DIMENSIONS:
        print(f"  {d}: {counts.get(d, 0)}")
    print(f"  未分配: {sum(1 for r in rows if r['dimension'] not in DIMENSIONS)}")

    # ---------------- 抽样：共同核心 400（每维度 25） ----------------
    # 只抽 domain_closed_book（真实需要检索的业务题）。
    # context_reasoning 的题干自带全部资料、不需要检索工具，
    # 混进检索测评会虚高成绩（方案第 5 节：必须通过工具取得事实）。
    rng = random.Random(20261101)
    by_dim = defaultdict(list)
    for r in rows:
        if r.get("evaluation_track") == "context_reasoning":
            continue
        by_dim[r["dimension"]].append(r)
    core = []
    shortfalls = []
    for d in DIMENSIONS:
        pool = sorted(by_dim.get(d, []), key=lambda r: r["qid"])
        rng.shuffle(pool)
        take = pool[:25]
        if len(take) < 25:
            shortfalls.append((d, len(take)))
        core.extend(take)
    core_qids = sorted(r["qid"] for r in core)
    core_sha = sha(json.dumps(core_qids))

    (OUT / "core400.json").write_text(json.dumps({
        "seed": 20261101, "per_dimension": 25, "dimensions": DIMENSIONS,
        "count": len(core_qids), "qids": core_qids, "qids_sha256": core_sha,
        "shortfalls": [{"dimension": d, "available": n} for d, n in shortfalls],
        "note": "共同核心题单，四模型主比较用；抽样只读题目标签，不读模型答案。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 抽样：Q4 追加 400（每维度再 25） ----------------
    used = set(core_qids)
    extra = []
    extra_shortfalls = []
    for d in DIMENSIONS:
        pool = sorted([r for r in by_dim.get(d, [])
                       if r["qid"] not in used
                       and r.get("evaluation_track") != "context_reasoning"],
                      key=lambda r: r["qid"])
        rng.shuffle(pool)
        take = pool[:25]
        if len(take) < 25:
            extra_shortfalls.append({"dimension": d, "available": len(take), "missing": 25 - len(take)})
        extra.extend(take)
        used.update(r["qid"] for r in take)
    extra_qids = sorted(r["qid"] for r in extra)
    (OUT / "q4_extra400.json").write_text(json.dumps({
        "seed": 20261101, "per_dimension": 25, "dimensions": DIMENSIONS,
        "count": len(extra_qids), "qids": extra_qids,
        "qids_sha256": sha(json.dumps(extra_qids)),
        "shortfalls": extra_shortfalls,
        "target_count": 400,
        "ready": len(extra_qids) == 400,
        "note": "Q4 相对共同400的追加题；与 core400 不重叠。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\ncore400: {len(core_qids)} 题  sha={core_sha[:16]}…")
    if shortfalls:
        print("维度配额不足（不足 25 题）:")
        for d, n in shortfalls:
            print(f"   {d}: {n}")
    print(f"q4_extra400: {len(extra_qids)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
