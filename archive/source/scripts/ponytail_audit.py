"""Ponytail-style repository audit: over-defensive patterns and intentional debt.

This is a report generator, not an auto-rewriter. Security red lines (ACL,
embedding integrity, budget, trust boundaries) are reported separately and are
not candidates for deletion.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("app", "enterprise_rag")
RED_LINE_FILES = {
    "enterprise_rag/scope.py",
    "enterprise_rag/budget.py",
    "enterprise_rag/release.py",
    "enterprise_rag/embedding.py",
    "app/core/backpressure.py",
    "app/domain/user/models.py",
}
PATTERNS = {
    "broad_except": re.compile(r"except\s+Exception"),
    "silent_except": re.compile(r"except\s+Exception[^\n]*:\s*\n\s*(pass|continue|return\s+None)"),
    "todo": re.compile(r"\b(TODO|FIXME|XXX)\b"),
    "shortcut": re.compile(r"#\s*ponytail:shortcut"),
    "manual_db_close": re.compile(r"get_sync_db_connection\(\)"),
}


def _files():
    for folder in SCAN_DIRS:
        for path in (ROOT / folder).rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path


def _function_lengths(path: Path) -> list[tuple[str, int, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
            rows.append((node.name, node.end_lineno - node.lineno + 1, node.lineno))
    return rows


def main() -> None:
    counts = Counter()
    files: list[dict] = []
    long_functions: list[dict] = []
    for path in _files():
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        row = {"file": relative, "red_line": relative in RED_LINE_FILES}
        for name, pattern in PATTERNS.items():
            found = len(pattern.findall(text))
            row[name] = found
            counts[name] += found
        files.append(row)
        for function, length, line in _function_lengths(path):
            if length > 100:
                long_functions.append({"file": relative, "function": function,
                                       "lines": length, "line": line})

    files.sort(key=lambda item: (-item["broad_except"], item["file"]))
    long_functions.sort(key=lambda item: -item["lines"])
    report = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "counts": dict(counts),
        "files": files,
        "long_functions": long_functions[:30],
        "red_lines": sorted(RED_LINE_FILES),
        "note": "broad_except 在边界层用于统一错误映射；安全红线文件不计入删除候选。",
    }
    out = ROOT / "audit_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ponytail_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Ponytail 全库瘦身审计", "",
        f"- 生成时间：{report['generated_at']}",
        f"- 宽泛异常捕获：{counts['broad_except']} 处",
        f"- 静默异常：{counts['silent_except']} 处",
        f"- TODO/FIXME：{counts['todo']} 处",
        f"- 已登记 shortcut：{counts['shortcut']} 处",
        f"- 超 100 行函数：{len(long_functions)} 个", "",
        "## 宽泛异常分布（按文件）", "",
        "| 文件 | broad_except | silent_except | 安全红线 |",
        "| --- | --- | --- | --- |",
    ]
    for row in files[:25]:
        if row["broad_except"] or row["silent_except"]:
            lines.append(f"| {row['file']} | {row['broad_except']} | {row['silent_except']} | "
                         f"{'是' if row['red_line'] else '否'} |")
    lines += ["", "## 长函数（>100 行）", "", "| 文件 | 函数 | 行数 |", "| --- | --- | --- |"]
    for item in long_functions[:20]:
        lines.append(f"| {item['file']} | {item['function']} | {item['lines']} |")
    lines += ["", "## 处理原则", "",
              "1. 先 grep 全部调用方，再改公共入口；不在每个调用点加守卫。",
              "2. 边界层统一映射用户错误；内部只抛领域异常，不重复 log+raise。",
              "3. ACL、向量完整性、预算、路径和信任边界是安全红线，不删除。",
              "4. 为性能/模型兼容保留的捷径用 `# ponytail:shortcut` 登记。"]
    (out / "ponytail_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"counts": dict(counts), "long_functions": len(long_functions),
                      "report": str(out / "ponytail_audit.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
