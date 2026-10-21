"""按标题/章节切分制度文档，并控制块大小与重叠。"""
import re

from app.capabilities.retrieval.tokenizer import tokenize

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_CN_HEADING = re.compile(r"^(第[一二三四五六七八九十百零]+[章节条]|[一二三四五六七八九十百零]+、)\s*(.*)$")

MAX_CHARS = 420
MIN_CHARS = 120
OVERLAP = 60


def _detect_heading(line: str) -> str | None:
    match = _MD_HEADING.match(line)
    if match:
        return match.group(1).strip()
    match = _CN_HEADING.match(line)
    if match:
        prefix, rest = match.group(1), match.group(2).strip()
        return f"{prefix}{rest}".strip()
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = ""
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((current_title or "正文", body))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = _detect_heading(line)
        if heading:
            flush()
            buffer = []
            current_title = heading
        else:
            buffer.append(line)
    flush()
    return sections


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return pieces


def _pack(body: str) -> list[str]:
    paragraphs = [line.strip() for line in body.splitlines() if line.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        for piece in _split_long(paragraph):
            if not buffer:
                buffer = piece
            elif len(buffer) + len(piece) + 1 <= MAX_CHARS:
                buffer += "\n" + piece
            else:
                chunks.append(buffer)
                buffer = piece
    if buffer:
        chunks.append(buffer)
    # 过短的尾块并入上一块，避免“半句话”块。
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHARS:
        merged = chunks[-2] + "\n" + chunks[-1]
        chunks = chunks[:-2] + [_split_long(merged)[0]]
    return chunks


def chunk_document(document: dict) -> list[dict]:
    title = document["title"]
    chunks = []
    index = 0
    for section, body in split_sections(document["text"]):
        for piece in _pack(body):
            prefixed = f"{title} · {section}\n{piece}"
            chunks.append(
                {
                    "chunk_id": f"{document['doc_id']}#c{index:03d}",
                    "doc_id": document["doc_id"],
                    "section": section,
                    "ord": index,
                    "text": prefixed,
                    "token_count": len(tokenize(prefixed)),
                }
            )
            index += 1
    return chunks
