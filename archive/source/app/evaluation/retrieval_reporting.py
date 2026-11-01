"""Explicit runs, immutable source answers and frozen scoring denominators."""
import json
from pathlib import Path

from app.evaluation.retrieval_corpus import read_jsonl, write_json
from app.evaluation.retrieval_grading import JUDGE_VERSION, grade


def run_rows(run: Path) -> tuple[list[str], dict[str, dict]]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    expected = [c["qid"] for c in manifest["contract"]["cases"]] if "contract" in manifest else manifest["cases"]
    if len(expected) != len(set(expected)):
        raise ValueError(f"Duplicate frozen qids: {run.name}")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "traces").glob("*.json"))]
    ids = [r["qid"] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(expected):
        raise ValueError(f"Trace identity mismatch: {run.name}")
    return expected, {r["qid"]: r for r in rows}


def report_runs(runs: list[Path], gold_path: Path, output: Path) -> dict:
    # No score-based run selection or exclusion; even all-error runs are retained.
    gold_rows = read_jsonl(gold_path)
    golds = {r["qid"]: r for r in gold_rows}
    if len(golds) != len(gold_rows):
        raise ValueError("Duplicate gold qids")
    output.mkdir(parents=True, exist_ok=False)
    summaries, comparisons = [], {}
    for run in runs:
        expected, traces = run_rows(run)
        scores = []
        for qid in expected:
            trace = traces.get(qid)
            verdict = grade(trace, golds[qid]) if trace else {"correct": None, "success": False,
                "evidence_supported": False, "review_status": "not_run", "reasons": ["missing_trace"]}
            scores.append({"qid": qid, "run": run.name, "status": trace["status"] if trace else "not_run", **verdict})
        write_json(output / f"{run.name}.scores.json", scores)
        summaries.append({"run": run.name, "planned": len(expected), "recorded": len(traces),
            "answered": sum(t["status"] == "answered" for t in traces.values()),
            "success": sum(s["success"] for s in scores),
            "pending_review": sum(s["correct"] is None and s["review_status"] != "not_run" for s in scores),
            "not_run": sum(s["review_status"] == "not_run" for s in scores),
            "verified_failures": sum(s["correct"] is False or s["correct"] is True and not s["success"] for s in scores),
            "status_counts": {s: sum(t["status"] == s for t in traces.values()) for s in sorted({t["status"] for t in traces.values()})}})
        comparisons[run.name] = {s["qid"]: s for s in scores}
    common = set.intersection(*(set(s) for s in comparisons.values()))
    result = {"runs": summaries, "common_qids": sorted(common), "scoring": JUDGE_VERSION,
              "warning": "Pending semantic reviews are not graded failures or verified successes. Original traces unchanged."}
    write_json(output / "summary.json", result)
    lines = ["# 检索复测报告", "", "分母来自显式选择的运行题单；不按作答比例删除运行。原回答不改写。", "",
             "| 运行 | 计划任务 | 记录 | 已作答 | 已核验通过 | 已核验失败 | 待语义复核 | 未运行 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        lines.append(f"| {s['run']} | {s['planned']} | {s['recorded']} | {s['answered']} | {s['success']} | {s['verified_failures']} | {s['pending_review']} | {s['not_run']} |")
    lines += ["", "待复核答案不计为确定错误；未完成复核前不发布最终正确率。已核验通过数/计划数只是通过率下界。",
              "引用须逐段验证来源、定位和原文；所有必要证据组均覆盖，不采用50%文档覆盖阈值。",
              "缺少任务级rubric的历史开放答案全部待复核，旧启发式分数不作为正式结论。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
