"""Regrade only saved original answers after rejecting a false-positive join.

The product pipeline is never invoked. Original executions and source hashes are
preserved; the only change is requiring an explicit relational conclusion in R07.
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reasoning import markdown

out = Path(__file__).parent
target = out / "baseline.json"
report = json.loads(target.read_text(encoding="utf-8"))
for suffix in ["json", "md"]:
    original = out / ("baseline_initial_grading." + suffix)
    if not original.exists():
        original.write_bytes((out / ("baseline." + suffix)).read_bytes())
fixtures = {c["id"]: c for c in [json.loads(x) for x in (out / "cases.jsonl").read_text(encoding="utf-8").splitlines()]}
for case in report["cases"]:
    if case["id"] != "R07_person_department":
        continue
    fixture = fixtures[case["id"]]
    case["expected"] = fixture["expected"]
    case["pattern_hits"] = {p: bool(re.search(p, case["answer"], re.I)) for p in fixture["answer_patterns"]}
    case["answer_correct"] = all(case["pattern_hits"].values()) and not case["refused"]
    case["strict_passed"] = case["answer_correct"] and case["all_evidence_recalled"] and case["all_evidence_cited"]
    if not case["strict_passed"]:
        case["failure_reasons"] = ["仅并列引用项目负责人和员工部门原文，没有显式形成项目负责人所属部门结论"]
report["fixtures_sha256_at_execution"] = report.get("fixtures_sha256_at_execution", report["fixtures_sha256"])
report["fixtures_sha256"] = hashlib.sha256((out / "cases.jsonl").read_bytes()).hexdigest()
report["grading_note"] = "R07原关键词检查错误放过并列照抄；现仅使用保存的实际回答重新评分，未运行改后业务代码。原始分数与记录见baseline_initial_grading.json/md。"
report["grading_updated_at"] = datetime.now().astimezone().isoformat()
for metric in ["strict_passed", "answer_correct"]:
    report["metrics"][metric] = sum(c[metric] for c in report["cases"])
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(out / "baseline.md").write_text(markdown(report) + "\n\n评分修订说明：" + report["grading_note"], encoding="utf-8")
print(json.dumps(report["metrics"], ensure_ascii=False))
