# -*- coding: utf-8 -*-
"""导出测评交付物：题单清单、来源对照表、缺口表。

方案第 10 节要求的验收交付之一：题目与来源对照表、固定题单、
资料可达/依赖缺口表。本脚本从题库与 manifest 直接生成，不手工誊抄。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:/Users/34157/Desktop/vllmproject/澜图智能企业智能体平台")
PLAN = ROOT / "research" / "retrieval_evaluation_20261101"
BANK = ROOT / "eval_bench" / "versions" / "20261030_v2"
OUT = PLAN / "deliverables"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 语料 manifest：来源对照表 ----
    manifest = load_jsonl(ROOT / "enterprise_rag" / "corpus" / "manifest.jsonl")
    fmt_count = Counter(m.get("format") for m in manifest)
    cat_count = Counter(m.get("category") for m in manifest)
    fam = defaultdict(list)
    for m in manifest:
        fam[m.get("doc_family")].append(m)

    lines = ["# 资料来源对照表", "",
             f"语料来源：`enterprise_rag/corpus/docs/`，共 {len(manifest)} 篇。"
             f"格式分布：{dict(fmt_count)}。类别分布：{dict(cat_count)}。", "",
             "## 一、同族多版本（版本消解与生效期判定的依据）", ""]
    for family, items in sorted(fam.items()):
        if len(items) > 1:
            lines.append(f"### 文档族 `{family}`（{len(items)} 个版本）")
            lines.append("")
            lines.append("| doc_id | 版本 | 生效日期 | 状态 | 标题 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for m in sorted(items, key=lambda x: str(x.get("effective_date", ""))):
                lines.append(f"| `{m['doc_id']}` | {m.get('version', '')} | "
                             f"{m.get('effective_date', '')} | {m.get('status', '')} | {m.get('title', '')} |")
            lines.append("")

    lines += ["## 二、按类别的文档清单", "",
              "| 类别 | 篇数 | doc_id |", "| --- | ---: | --- |"]
    bycat = defaultdict(list)
    for m in manifest:
        bycat[m.get("category")].append(m["doc_id"])
    for c in sorted(bycat):
        lines.append(f"| {c} | {len(bycat[c])} | {' '.join('`' + d + '`' for d in sorted(bycat[c]))} |")

    (OUT / "来源对照表.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 固定题单 ----
    core = json.loads((PLAN / "bank" / "core400.json").read_text(encoding="utf-8"))
    extra = json.loads((PLAN / "bank" / "q4_extra400.json").read_text(encoding="utf-8"))

    dim_rows = {r["qid"]: r for r in load_jsonl(BANK / "question_bank_dimensioned.jsonl")}
    for r in load_jsonl(PLAN / "bank" / "augment_questions.jsonl"):
        dim_rows[r["qid"]] = r

    qrows = []
    for r in load_jsonl(BANK / "question_bank.jsonl"):
        qrows.append(r)
    for r in load_jsonl(PLAN / "bank" / "augment_questions.jsonl"):
        qrows.append(r)
    qmap = {r["qid"]: r for r in qrows}

    for name, spec in (("题单_core400", core), ("题单_q4_extra400", extra)):
        out = [f"# {name}（固定题单）", "",
               f"- 种子：{spec.get('seed')}",
               f"- 题数：{spec.get('count')}",
               f"- qids SHA256：`{spec.get('qids_sha256')}`",
               f"- 说明：{spec.get('note', '')}", "",
               "| # | qid | 维度 | 难度 | 跳数 | 期望文档数 | 题型 |",
               "| ---: | --- | --- | --- | ---: | ---: | --- |"]
        for i, qid in enumerate(spec["qids"], 1):
            r = qmap.get(qid, {})
            dim = dim_rows.get(qid, {}).get("dimension", r.get("dimension", ""))
            out.append(f"| {i} | `{qid}` | {dim} | {r.get('difficulty', '')} | "
                       f"{r.get('multi_hop_depth', '')} | "
                       f"{len(r.get('expected_doc_ids', []))} | {r.get('question_type', '')} |")
        (OUT / f"{name}.md").write_text("\n".join(out), encoding="utf-8")

    # ---- 资料可达 / 依赖缺口表 ----
    gap = ["# 资料可达与依赖缺口表", "",
           "按方案第 4 节要求：未接通的来源必须明确标为未实现，不能混入主题单计为模型能力失败。", "",
           "| 来源套件 | 状态 | 说明 | 对测评的影响 |", "| --- | --- | --- | --- |",
           "| 企业知识文档（md/pdf/docx/xlsx/csv） | **已接入** | 136 篇快照，"
           "泄漏检查 0；工具支持检索/分页读取/列表 | 主测全部题目基于此套件 |",
           "| 源码检索（search_code 等） | **未纳入本轮题单** | 方案要求作为 source_type 分层，"
           "需真实代码任务；本轮 core400 全部为业务文档题 | 源码检索能力未被本轮成绩覆盖 |",
           "| 公开配置查询 | **未纳入本轮题单** | 同上 | 同上 |",
           "| 数据库 SQL | **not_implemented** | 方案已列为扩展来源套件；未接通前不冻结相关题 |",
           "| 第三方 API | **not_implemented** | 同上 |",
           "| 外部网页检索 | **not_implemented** | 方案要求用固定页面快照与实时联网轮分开，不混合 |",
           "| 平台 HTTP/流式入口 | **not_run** | 检索层依赖 MySQL；沙箱禁止启动 wsl.exe → "
           "MySQL/Redis 不可用 → 平台无法启动 | 方案第 6 阶段未执行，未用直连结果冒充 |",
           "| MCP 文档搜索 | **部分可用** | 现有实现仅 MD 字面匹配与短片段返回 |",
           "| OCR 扫描件 | **not_implemented** | 语料中无扫描件，未用可提取 PDF 冒充 OCR 测试 |",
           "", "## 已实现但未纳入成绩的连接", "",
           "| 能力 | 状态 | 说明 |", "| --- | --- | --- |",
           "| 本地 Q4_K_M / Q6_K 推理服务 | 已跑通但**未按方案执行** | "
           "16GB 显存下 27B 模型单轮 >400 秒，8 轮/600 秒预算不可达；无任何本地成绩 |",
           "| 故障注入/权限套件 | 未建 | 依赖平台服务 |",
           ""]
    (OUT / "资料可达与依赖缺口表.md").write_text("\n".join(gap), encoding="utf-8")

    print(f"已导出到 {OUT}:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
