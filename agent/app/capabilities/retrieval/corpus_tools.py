"""Public corpus tools, from lantu-platform 61bf13d archive/source/app/evaluation/retrieval_corpus.py.

Copied unchanged except this provenance note. No question or answer bank access.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import jsonschema


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_blocks(path: Path):
    """Yield original source locators in document order; parsing failures stop snapshot creation."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                yield f"line:{i}", line
    elif suffix == ".pdf":
        from pypdf import PdfReader
        for i, page in enumerate(PdfReader(path).pages, 1):
            text = page.extract_text() or ""
            if not text.strip():
                raise ValueError(f"PDF page needs OCR: {path.name} page {i}")
            yield f"page:{i}", text
    elif suffix == ".docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        for i, block in enumerate(Document(path).iter_inner_content(), 1):
            if isinstance(block, Paragraph) and block.text.strip():
                yield f"block:{i}:paragraph", block.text
            elif isinstance(block, Table):
                for j, row in enumerate(block.rows, 1):
                    yield f"block:{i}:row:{j}", "\t".join(c.text for c in row.cells)
    elif suffix == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, data_only=True, read_only=True)
        for sheet in book:
            for row in sheet:
                cells = [f"{c.coordinate}={c.value}" for c in row if c.value is not None]
                if cells:
                    yield f"sheet:{sheet.title}:row:{row[0].row}", " | ".join(cells)
        book.close()
    elif suffix == ".csv":
        import csv
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for i, row in enumerate(csv.reader(stream), 1):
                yield f"row:{i}", " | ".join(row)
    else:
        raise ValueError(f"Unsupported source format: {path.name}")


def build_snapshot(corpus: Path, workspace: Path) -> dict:
    # Create a fresh directory: existing snapshots and their hashes remain immutable.
    workspace.mkdir(parents=True, exist_ok=False)
    docs, segments, ids = [], [], set()
    originals = workspace / "originals"
    import shutil
    for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
        doc_id = path.stem
        if doc_id in ids:
            raise ValueError(f"Duplicate document id: {doc_id}")
        ids.add(doc_id)
        rel = path.relative_to(corpus)
        target = originals / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        blocks = list(source_blocks(path))
        if not blocks:
            raise ValueError(f"Empty source: {path.name}")
        docs.append({"doc_id": doc_id, "path": rel.as_posix(), "format": path.suffix[1:],
                     "title": blocks[0][1][:120], "sha256": digest(path.read_bytes())})
        for locator, text in blocks:
            # Long pages keep their source page and an explicit character offset.
            for offset in range(0, len(text), 1000):
                segments.append({"doc_id": doc_id, "locator": f"{locator}@{offset}",
                                 "text": text[offset:offset + 1000]})
    for name, rows in [("docs.jsonl", docs), ("segments.jsonl", segments)]:
        (workspace / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {"documents": len(docs), "segments": len(segments),
                "files": {n: digest((workspace / n).read_bytes()) for n in ["docs.jsonl", "segments.jsonl"]},
                "source_files": {d["path"]: d["sha256"] for d in docs}}
    write_json(workspace / "manifest.json", manifest)
    return manifest


def terms(text: str) -> list[str]:
    import re
    text = text.lower()
    out = re.findall(r"[a-z0-9_.%+-]+", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            out.append(run)
    return out


SCHEMAS = {
    "search": {"type": "object", "properties": {"query": {"type": "string", "minLength": 1},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["query"], "additionalProperties": False},
    "read": {"type": "object", "properties": {"doc_id": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 60}},
        "required": ["doc_id"], "additionalProperties": False},
    "list": {"type": "object", "properties": {"keyword": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0}}, "additionalProperties": False},
}
DESCRIPTIONS = {
    "search": "按词法BM25检索公开资料原文。返回稳定doc_id、locator与完整片段；top_k默认5，上限10。可改写查询或read补证。",
    "read": "读取doc_id原文。offset为文档内片段序号，limit为片段数(默认40，上限60)。默认可读常见短文的正文；next_offset非空表示尚有后文。",
    "list": "列出公开来源的doc_id、标题、格式。keyword可选；offset分页，每页20项。目录元数据不是事实证据。",
}
TOOL_SPECS = [{"type": "function", "function": {"name": n, "description": DESCRIPTIONS[n], "parameters": s}}
              for n, s in SCHEMAS.items()]


class CorpusTools:
    def __init__(self, workspace: Path):
        manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["files"].items():
            if digest((workspace / name).read_bytes()) != expected:
                raise ValueError(f"Snapshot changed: {name}")
        self.docs = {d["doc_id"]: d for d in read_jsonl(workspace / "docs.jsonl")}
        self.segments = read_jsonl(workspace / "segments.jsonl")
        self.tf = [Counter(terms(self.docs[s["doc_id"]]["title"] + " " + s["text"])) for s in self.segments]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        df = Counter(term for tf in self.tf for term in tf)
        import math
        self.idf = {term: math.log(1 + (len(self.tf) - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def run(self, name: str, args: dict) -> dict:
        # A single external tool boundary: reject bad arguments, never coerce to unrelated defaults.
        if name not in SCHEMAS:
            return {"error": "unknown_tool", "name": name}
        try:
            jsonschema.validate(args, SCHEMAS[name])
        except jsonschema.ValidationError as exc:
            return {"error": "invalid_arguments", "detail": exc.message}
        if name == "list":
            keyword, offset = args.get("keyword", "").lower(), int(args.get("offset", 0))
            docs = [d for d in self.docs.values() if keyword in (d["doc_id"] + d["title"]).lower()]
            return {"documents": docs[offset:offset + 20], "next_offset": offset + 20 if offset + 20 < len(docs) else None}
        if name == "read":
            doc_id = args["doc_id"]
            if doc_id not in self.docs:
                return {"error": "unknown_document", "doc_id": doc_id}
            rows = [s for s in self.segments if s["doc_id"] == doc_id]
            start, size = int(args.get("offset", 0)), int(args.get("limit", 40))
            return {"segments": rows[start:start + size], "next_offset": start + size if start + size < len(rows) else None,
                    "total_segments": len(rows)}
        query, scores = Counter(terms(args["query"])), []
        for i, tf in enumerate(self.tf):
            score = sum(self.idf[t] * count * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * self.lengths[i] / self.avg_length))
                        for t, count in query.items() if t in tf)
            if score:
                scores.append((score, i))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return {"hits": [{**self.segments[i], "offset": sum(s["doc_id"] == self.segments[i]["doc_id"] for s in self.segments[:i]),
                          "rank": rank, "score": round(score, 5)}
                         for rank, (score, i) in enumerate(scores[:int(args.get("top_k", 5))], 1)]}
