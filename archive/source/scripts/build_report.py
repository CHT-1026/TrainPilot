"""Offline report: explicit runs, no score-based selection or source mutation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation.retrieval_reporting import report_runs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(report_runs(a.runs, a.gold, a.output), ensure_ascii=False))
