"""从当前语料与图谱动态生成多跳/版本/反例评测题（S28）。

题目在运行时由当前索引与图谱推导，不写入固定题库；`--dry-run` 只输出数量。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_OUT = ROOT / "audit_artifacts" / "generated_cases"
_DEPT_LIST = re.compile(r"设\s*\d+\s*个一级部门[：:]([^。]+)。")


def _doc_tokens(index, doc_id: str) -> set[str]:
    from enterprise_rag.evidence import MODEL_RE, UNIT_RE

    text = "\n".join(chunk.get("text", "") for chunk in index.by_doc_ord.get(doc_id, []))
    tokens = set(MODEL_RE.findall(text)) | set(UNIT_RE.findall(text))
    for match in _DEPT_LIST.finditer(text):
        tokens.update(item.strip() for item in re.split(r"[、，,]", match.group(1)) if item.strip())
    return {token for token in tokens if len(token) >= 2 and not re.fullmatch(r"20\d{2}", token)}


def _dept_names(index, doc_id: str) -> set[str]:
    text = "\n".join(chunk.get("text", "") for chunk in index.by_doc_ord.get(doc_id, []))
    names: set[str] = set()
    for match in _DEPT_LIST.finditer(text):
        names.update(item.strip() for item in re.split(r"[、，,]", match.group(1)) if item.strip())
    return names


def _version_keywords(index, older: dict, newer: dict) -> list[str]:
    """每版至少一个只在该版出现的词，避免“答案无关但引用正确”通过。"""
    old_tokens = _doc_tokens(index, older["doc_id"])
    new_tokens = _doc_tokens(index, newer["doc_id"])
    old_depts = _dept_names(index, older["doc_id"])
    new_depts = _dept_names(index, newer["doc_id"])
    dept_tokens = old_depts | new_depts
    old_only = sorted(old_tokens - new_tokens, key=len, reverse=True)
    new_only = sorted(new_tokens - old_tokens, key=len, reverse=True)
    if dept_tokens:
        old_only = [token for token in old_only if token in dept_tokens] or old_only
        new_only = [token for token in new_only if token in dept_tokens] or new_only
    return ([old_only[0]] if old_only else []) + ([new_only[0]] if new_only else [])


def _version_cases(index, rng: random.Random, limit: int) -> list[dict]:
    cases = []
    for family, docs in sorted(index.family_docs.items()):
        if len(docs) < 2:
            continue
        docs = sorted(docs, key=lambda item: (item.get("effective_date") or "", item.get("version") or ""))
        older, newer = docs[0], docs[-1]
        question = (
            f"《{older['title']}》（{older['version']}）与《{newer['title']}》"
            f"（{newer['version']}）相比，主要变化有哪些？"
        )
        doc_ids = [older["doc_id"], newer["doc_id"]]
        keywords = _version_keywords(index, older, newer)
        if len(keywords) < 2:
            continue
        cases.append({
            "id": f"gen_version_{len(cases)+1:03d}",
            "category": "version_comparison",
            "question": question,
            "expected_doc_ids": doc_ids,
            "answer_keywords": keywords,
            "expect_refusal": False,
            "generated_from": {"family": family, "doc_ids": doc_ids},
        })
    rng.shuffle(cases)
    return cases[:limit]


def _multi_hop_cases(store, rng: random.Random, limit: int) -> list[dict]:
    paths = []
    persons = [entity for entity in store.entities.values() if entity.get("type") == "person"]
    for person in persons:
        # manages 的方向是“上级 -> 下属”；问“上级的上级”必须沿 incoming 边回溯。
        for key in store.incoming.get(person["id"], []):
            first = store.relations.get(key)
            if not first or first.get("relation") != "manages":
                continue
            manager_id = first["source"]
            for second_key in store.incoming.get(manager_id, []):
                second = store.relations.get(second_key)
                if not second or second.get("relation") != "manages":
                    continue
                target = store.entities.get(second["source"])
                if not target or target.get("type") != "person":
                    continue
                manager = store.entities.get(manager_id, {})
                paths.append({
                    "root": person,
                    "middle": manager,
                    "target": target,
                    "chunk_ids": [first.get("chunk_id", ""), second.get("chunk_id", "")],
                    "doc_ids": [first.get("doc_id", ""), second.get("doc_id", "")],
                })
    rng.shuffle(paths)
    cases = []
    for path in paths[:limit]:
        root_name = path["root"].get("name", "")
        target_name = path["target"].get("name", "")
        if not root_name or not target_name:
            continue
        cases.append({
            "id": f"gen_multihop_{len(cases)+1:03d}",
            "category": "multi_hop_relation",
            "question": f"{root_name}的上级的上级是谁？",
            "expected_doc_ids": [doc for doc in dict.fromkeys(path["doc_ids"]) if doc],
            "answer_keywords": [target_name],
            "expect_refusal": False,
            "generated_from": {"chunk_ids": path["chunk_ids"],
                               "path": f"{root_name} -> {path['middle'].get('name', '')} -> {target_name}"},
        })
    return cases


def _counterexample_cases(rng: random.Random, limit: int) -> list[dict]:
    unknowns = [
        "星河重工集团", "蓝海精密制造有限公司", "云岭机器人研究院",
        "北辰自动化有限公司", "远山智能装备有限公司",
    ]
    rng.shuffle(unknowns)
    cases = []
    for name in unknowns[:limit]:
        cases.append({
            "id": f"gen_counter_{len(cases)+1:03d}",
            "category": "counterexample",
            "question": f"{name}的注册资本和总部地址是什么？",
            "expected_doc_ids": [],
            "answer_keywords": [],
            "expect_refusal": True,
            "generated_from": {"unknown_entity": name},
        })
    return cases


def generate_cases(index=None, store=None, seed: int = 20261022, version_limit: int = 20,
                   hop_limit: int = 20, counter_limit: int = 10) -> list[dict]:
    if index is None or store is None:
        from enterprise_rag.graph.retrieve import load_graph
        from enterprise_rag.store import load_index

        index = index or load_index()
        store = store or load_graph()
    rng = random.Random(seed)
    cases = []
    cases.extend(_version_cases(index, rng, version_limit))
    cases.extend(_multi_hop_cases(store, rng, hop_limit))
    cases.extend(_counterexample_cases(rng, counter_limit))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = generate_cases(seed=args.seed, version_limit=args.limit,
                           hop_limit=args.limit, counter_limit=max(1, args.limit // 4))
    cases = cases[:args.limit]
    summary = {
        "seed": args.seed,
        "count": len(cases),
        "by_category": {},
        "dry_run": args.dry_run,
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    }
    for case in cases:
        summary["by_category"][case["category"]] = summary["by_category"].get(case["category"], 0) + 1
    if not args.dry_run:
        _OUT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (_OUT / f"generated_{stamp}.jsonl").write_text(
            "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
