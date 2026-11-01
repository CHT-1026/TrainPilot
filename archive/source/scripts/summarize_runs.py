"""Read a generated summary without re-selecting or re-grading runs."""
import argparse
import json
from pathlib import Path
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    a = p.parse_args()
    print(json.dumps(json.loads(a.summary.read_text(encoding="utf-8")),ensure_ascii=False,indent=2))
