"""结构感知切分：按章节 + 段落打包，表格按表头重复切分。"""
import re

from app.capabilities.retrieval.tokenizer import tokenize

MAX_CHARS = 480
MIN_CHARS = 110
OVERLAP = 70
TABLE_ROWS_PER_CHUNK = 10

_SENTENCE = re.compile(r"(?<=[。！？\n])")


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    return parts or [text.strip()]


def _pack_text(text: str) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if len(sentence) > MAX_CHARS:
            # 超长句按字符窗口硬切，保留少量重叠。
            start = 0
            while start < len(sentence):
                end = min(start + MAX_CHARS, len(sentence))
                chunks.append(sentence[start:end])
                if end >= len(sentence):
                    break
                start = max(end - OVERLAP, start + 1)
            buffer = ""
            continue
        if not buffer:
            buffer = sentence
        elif len(buffer) + len(sentence) + 1 <= MAX_CHARS:
            buffer += "\n" + sentence
        else:
            chunks.append(buffer)
            tail = buffer[-OVERLAP:] if len(buffer) > OVERLAP else ""
            buffer = f"{tail}\n{sentence}".strip()
    if buffer:
        chunks.append(buffer)
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHARS:
        chunks[-2] = f"{chunks[-2]}\n{chunks[-1]}"
        chunks.pop()
    return chunks


def _split_table(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return [text]
    header = lines[0]
    body = [line for line in lines[1:] if not re.match(r"^\|\s*-{3,}", line.replace(" ", ""))]
    if not body:
        return [header]
    chunks = []
    for start in range(0, len(body), TABLE_ROWS_PER_CHUNK):
        group = body[start : start + TABLE_ROWS_PER_CHUNK]
        label = ""
        if len(body) > TABLE_ROWS_PER_CHUNK:
            label = f"（第 {start + 1}-{start + len(group)} 行）"
        chunks.append(f"{header}\n" + "\n".join(group) + (f"\n{label}" if label else ""))
    return chunks


def chunk_document(document: dict, blocks: list[dict]) -> list[dict]:
    title = document["title"]
    chunks: list[dict] = []
    ord_index = 0
    for block in blocks:
        section = block.get("section") or "正文"
        chunk_type = block.get("type", "text")
        pieces = _split_table(block["text"]) if chunk_type == "table" else _pack_text(block["text"])
        for piece in pieces:
            if not piece.strip():
                continue
            suffix = "（表格）" if chunk_type == "table" else ""
            text = f"{title} · {section}{suffix}\n{piece.strip()}"
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}#c{ord_index:03d}",
                    "doc_id": document["doc_id"],
                    "section": section,
                    "chunk_type": chunk_type,
                    "ord": ord_index,
                    "text": text,
                    "token_count": len(tokenize(text)),
                    "page": block.get("page"),
                    "paragraph": block.get("paragraph"),
                    "anchor": block.get("anchor", ""),
                    "extractor_version": block.get("extractor_version", ""),
                    "ocr": bool(block.get("ocr", False)),
                    "parent_id": document["doc_id"],
                }
            )
            ord_index += 1
    for position, chunk in enumerate(chunks):
        chunk["prev_id"] = chunks[position - 1]["chunk_id"] if position else ""
        chunk["next_id"] = chunks[position + 1]["chunk_id"] if position + 1 < len(chunks) else ""
    return chunks
