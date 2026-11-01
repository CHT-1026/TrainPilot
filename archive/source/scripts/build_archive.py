# -*- coding: utf-8 -*-
"""构建归档目录：source/ + questions/ + answers/ 三部分。

在 code_release/ 下生成一个自包含的归档目录，结构：

    code_release/archive/
      README.md                 归档总说明
      source/                   完整源代码
        app/ enterprise_rag/ scripts/ tests/ sql/ eval/ local_runtime/
        main.py requirements*.txt *.cmd 启动本地服务.py
        corpus/                 平台实际使用的知识语料（137 份）
      questions/                测评题库
        question_bank.jsonl         1239 道题（含题干、题型、难度、维度）
        question_bank_dimensioned.jsonl  同上 + 16 维度标签
        dimensions.json              维度定义与配额
        frozen/core400.json          冻结的共同 400 题题单
        frozen/q4_extra400.json      本地模型追加题单
      answers/                 与题目严格一一对应的答案
        answer_bank.jsonl            1239 份答案（含金标准、证据、期望文档）
        grading_rules.md             判分规则说明
      evaluation/               测评产出（成绩与轨迹）
        deliverables/                成绩报告、结论建议、题单
        runs/                        A/B 各 400 题完整轨迹
        bank/                        冻结题单、人名白名单
        判分器校准记录.md
        执行记录_v1.md
        清理前清单.md

题目与答案的对应关系由 **qid** 保证：两边的 qid 集合严格相等，
一一对应，缺一即报错退出。

用法：
    ./.venv312/Scripts/python.exe scripts/build_archive.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(r"C:/Users/34157/Desktop/vllmproject/澜图智能企业智能体平台")
REL = ROOT / "code_release"
ARCH = REL / "archive"

SRC = ROOT / "eval_bench" / "versions" / "20261030_v2"
EVAL = ROOT / "research" / "retrieval_evaluation_20261101"

# 源代码：目录 -> 是否整体复制（排除缓存与运行产物）
CODE_DIRS = ["app", "enterprise_rag", "scripts", "tests", "sql", "eval", "local_runtime"]
CODE_FILES = ["main.py", "requirements.txt", "requirements-local.txt",
              "启动本地服务.py", "停止服务.cmd", "启动服务.cmd",
              "运行评测.cmd", "重新导入语料.cmd"]

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                ".git", "node_modules", "dist", "build", ".egg-info"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".log", ".tmp"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    """复制目录，跳过缓存与临时产物。返回 (文件数, 字节数)。"""
    n = 0
    size = 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        if EXCLUDE_DIRS & set(f.parts):
            continue
        if f.suffix.lower() in EXCLUDE_SUFFIX:
            continue
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        n += 1
        size += f.stat().st_size
    return n, size


def load_jsonl(p: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["qid"]] = r
    return out


def main() -> int:
    if ARCH.exists():
        print(f"[warn] {ARCH} 已存在，先移除以保证干净重建")
        shutil.rmtree(ARCH)

    # ---------------- source/ ----------------
    print("== 构建 source/ ==")
    src_root = ARCH / "source"
    src_root.mkdir(parents=True)
    total_n = total_size = 0
    for d in CODE_DIRS:
        s = ROOT / d
        if not s.exists():
            print(f"  [skip] {d} 不存在")
            continue
        n, size = copy_tree(s, src_root / d)
        total_n += n
        total_size += size
        print(f"  {d:<18} {n:>4} 文件  {size/1024/1024:>8.1f} MB")
    for f in CODE_FILES:
        s = ROOT / f
        if not s.exists():
            continue
        shutil.copy2(s, src_root / f)
        total_n += 1
        total_size += s.stat().st_size
    # 语料：平台检索链路的输入，题目答案的正确性依赖它
    corpus_src = ROOT / "enterprise_rag" / "corpus"
    if corpus_src.exists():
        n, size = copy_tree(corpus_src, src_root / "corpus")
        total_n += n
        total_size += size
        print(f"  {'corpus':<18} {n:>4} 文件  {size/1024/1024:>8.1f} MB")
    print(f"  source/ 合计：{total_n} 文件 / {total_size/1024/1024:.1f} MB")

    # ---------------- questions/ + answers/ ----------------
    print("== 构建 questions/ 与 answers/ ==")
    qdir = ARCH / "questions"
    adir = ARCH / "answers"
    qdir.mkdir(parents=True)
    adir.mkdir(parents=True)

    q_main = load_jsonl(SRC / "question_bank.jsonl")
    a_main = load_jsonl(SRC / "answer_bank.jsonl")
    q_dim = load_jsonl(SRC / "question_bank_dimensioned.jsonl")
    q_aug = load_jsonl(EVAL / "bank" / "augment_questions.jsonl")
    a_aug = load_jsonl(EVAL / "bank" / "augment_answers.jsonl")

    # 合并：以主库为底，dimensioned 补维度标签与补建题题干
    questions: dict[str, dict] = {}
    for qid, r in q_main.items():
        questions[qid] = dict(r)
    for qid, r in q_dim.items():
        if qid in questions:
            if r.get("dimension"):
                questions[qid]["dimension"] = r["dimension"]
            if r.get("question"):
                questions[qid]["question"] = r["question"]
        else:
            questions[qid] = dict(r)
    for qid, r in q_aug.items():
        questions[qid] = dict(r)

    answers: dict[str, dict] = {}
    for qid, r in a_main.items():
        answers[qid] = dict(r)
    for qid, r in a_aug.items():
        answers[qid] = dict(r)

    # ---- 一致性校验：qid 必须严格一一对应 ----
    only_q = sorted(set(questions) - set(answers))
    only_a = sorted(set(answers) - set(questions))
    if only_q or only_a:
        print(f"[FAIL] 题目与答案 qid 不一致：")
        if only_q:
            print(f"  有题无答 {len(only_q)}: {only_q[:10]}")
        if only_a:
            print(f"  有答无题 {len(only_a)}: {only_a[:10]}")
        return 1
    print(f"  qid 一致性校验通过：{len(questions)} 题 ↔ {len(answers)} 答")

    # 逐题答案字段完整性
    missing_field = []
    for qid, a in answers.items():
        if not a.get("answer"):
            missing_field.append(qid)
    if missing_field:
        print(f"[FAIL] {len(missing_field)} 道题缺答案内容：{missing_field[:10]}")
        return 1
    print("  答案内容完整性校验通过")

    # 写出（按 qid 排序，保证可重复与可 diff）
    def dump(rows: dict[str, dict], path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for qid in sorted(rows):
                f.write(json.dumps(rows[qid], ensure_ascii=False) + "\n")

    dump(questions, qdir / "question_bank.jsonl")
    dump(answers, adir / "answer_bank.jsonl")
    # 带维度标签的题库（评测直接消费的文件）
    dump(q_dim, qdir / "question_bank_dimensioned.jsonl")

    # 维度定义
    dims: dict[str, int] = {}
    for q in questions.values():
        d = q.get("dimension")
        if d:
            dims[d] = dims.get(d, 0) + 1
    (qdir / "dimensions.json").write_text(json.dumps({
        "note": "16 个检索能力维度。题库按维度分层抽样，等配额宏平均。",
        "dimensions": [{"code": k, "name": k.split("_", 1)[1] if "_" in k else k,
                        "count": v} for k, v in sorted(dims.items())],
        "total_questions": len(questions),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  维度分布：{len(dims)} 个维度，共 {len(questions)} 题")

    # 冻结题单
    frozen = qdir / "frozen"
    frozen.mkdir(parents=True)
    for name in ("core400.json", "q4_extra400.json"):
        src = EVAL / "bank" / name
        if src.exists():
            shutil.copy2(src, frozen / name)
            print(f"  冻结题单：{name}")

    # ---------------- evaluation/ ----------------
    print("== 构建 evaluation/ ==")
    edir = ARCH / "evaluation"
    edir.mkdir(parents=True)
    for sub in ("deliverables", "runs", "bank"):
        s = EVAL / sub
        if s.exists():
            n, size = copy_tree(s, edir / sub)
            print(f"  {sub:<14} {n:>4} 文件  {size/1024/1024:>8.1f} MB")
    for f in ("判分器校准记录.md", "执行记录_v1.md", "清理前清单.md",
              "人工抽审清单.md", "完整检测方案.md"):
        s = EVAL / f
        if s.exists():
            shutil.copy2(s, edir / f)

    # ---------------- manifest ----------------
    manifest = {
        "archive_version": "1.0",
        "generated_at": "2026-11-01",
        "structure": {
            "source/": "完整源代码与平台实际使用的知识语料",
            "questions/": "测评题库（1239 题，含 16 维度标签与冻结题单）",
            "answers/": "与题目一一对应的答案（金标准 + 证据 + 期望文档）",
            "evaluation/": "本轮测评产出：成绩报告、运行轨迹、判分器校准记录",
        },
        "question_answer_correspondence": {
            "key": "qid",
            "question_count": len(questions),
            "answer_count": len(answers),
            "strictly_one_to_one": True,
            "verification": "构建时逐 qid 比对，缺一即中止",
        },
        "files": [],
    }
    for f in sorted(ARCH.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.json":
            manifest["files"].append({
                "path": str(f.relative_to(ARCH)).replace("\\", "/"),
                "bytes": f.stat().st_size,
                "sha256": sha256(f),
            })
    (ARCH / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  MANIFEST.json：{len(manifest['files'])} 个文件的 sha256 清单")

    print()
    print(f"归档完成 → {ARCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
