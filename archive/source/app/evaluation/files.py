"""Shared JSONL I/O only; no evaluator or provider dependencies."""
import hashlib
import json
from pathlib import Path

from app.evaluation.config import EvaluationConfig


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_rows(rows: list[dict], key: str) -> dict[str, dict]:
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'{key}存在重复记录')
    return indexed


def run_directory(config: EvaluationConfig, run_id: str) -> Path:
    root = config.path(config.runs_dir)
    directory = (root/run_id).resolve()
    if directory.parent != root:
        raise ValueError('run_id必须为runs目录下的单个运行名称')
    return directory
