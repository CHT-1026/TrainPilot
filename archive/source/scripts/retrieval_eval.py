"""Native-tool retrieval evaluation: public generation and private offline scoring."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.retrieval_corpus import build_snapshot, read_jsonl
from app.evaluation.retrieval_reporting import report_runs
from app.evaluation.retrieval_runtime import DEFAULT_BUDGET, RUNTIME_CONFIG, execute, model_config


def main():
    ap = argparse.ArgumentParser()
    commands = ap.add_subparsers(dest="action", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--corpus", type=Path, default=ROOT / "enterprise_rag/corpus/docs")
    snapshot.add_argument("--workspace", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", choices=["A", "B", "C", "D"], required=True)
    run.add_argument("--cases", type=Path, required=True, help="Public JSONL: qid and question/turns only")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--transport", choices=["mcp", "in_process"], default="mcp")
    run.add_argument("--resume", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--runs", nargs="+", type=Path, required=True)
    report.add_argument("--gold", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.action == "snapshot":
        print(json.dumps(build_snapshot(args.corpus, args.workspace), ensure_ascii=False))
    elif args.action == "run":
        if not RUNTIME_CONFIG['execution_enabled']:
            raise ValueError('Retrieval evaluation execution is disabled')
        cases = read_jsonl(args.cases)
        ids = [c["qid"] for c in cases]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Cases must have unique qids and cannot be empty")
        if any(set(c) - {"qid", "question", "turns"} for c in cases):
            raise ValueError("Only public prompt fields may enter generation")
        asyncio.run(execute(cases, model_config(args.model), args.workspace, args.run_dir, DEFAULT_BUDGET, args.transport, args.resume))
    else:
        print(json.dumps(report_runs(args.runs, args.gold, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
