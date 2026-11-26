"""多格式文档读取：Markdown/TXT、DOCX、PDF、CSV、XLSX。

统一输出结构：
    {"format": "md", "blocks": [{"section": "章节", "text": "...", "type": "text|table"}]}
表格统一渲染为管道文本，保留表头以便切分后仍可独立理解。
"""
import csv
import re
from pathlib import Path

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_CN_HEADING = re.compile(r"^(第[一二三四五六七八九十百零]+[章节条]|[一二三四五六七八九十百零]+、)\s*(.*)$")
_META_PREFIXES = (
    "所属部门：",
    "作者：",
    "创建时间：",
    "更新时间：",
    "版本号：",
    "文档类型：",
    "文档族：",
    "生效日期：",
)


def _is_meta_line(line: str) -> bool:
    """元数据由 manifest/数据库承载，不进入正文索引，避免时间/部门词污染检索。"""
    return any(line.startswith(prefix) for prefix in _META_PREFIXES)


def _is_heading(line: str) -> str | None:
    match = _MD_HEADING.match(line)
    if match:
        return match.group(1).strip()
    match = _CN_HEADING.match(line)
    if match:
        return f"{match.group(1)}{match.group(2).strip()}"
    return None


def _markdown_blocks(text: str, fallback: str) -> list[dict]:
    blocks: list[dict] = []
    section = fallback
    buffer: list[str] = []
    table: list[str] = []

    def flush_text():
        nonlocal buffer
        body = "\n".join(line for line in buffer if line.strip()).strip()
        if body:
            blocks.append({"section": section, "text": body, "type": "text"})
        buffer = []

    def flush_table():
        nonlocal table
        rows = [line for line in table if line.strip()]
        if rows:
            blocks.append({"section": section, "text": "\n".join(rows), "type": "table"})
        table = []

    for raw in text.replace("\r\n", "\n").splitlines():
        line = raw.rstrip()
        if _is_meta_line(line.strip()):
            continue
        if line.strip().startswith("|") and line.strip().endswith("|"):
            flush_text()
            table.append(line.strip())
            continue
        if table:
            flush_table()
        heading = _is_heading(line.strip())
        if heading:
            flush_text()
            section = heading
            continue
        if line.strip():
            buffer.append(line.strip())
    flush_text()
    flush_table()
    return blocks


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _docx_blocks(path: Path) -> list[dict]:
    import docx

    document = docx.Document(str(path))
    blocks: list[dict] = []
    section = "正文"
    buffer: list[str] = []

    def flush():
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            blocks.append({"section": section, "text": body, "type": "text"})
        buffer = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text or _is_meta_line(text):
            continue
        style = (paragraph.style.name or "").lower()
        if "heading" in style or "标题" in style:
            flush()
            section = text
            continue
        buffer.append(text)
    flush()

    for index, table in enumerate(document.tables, 1):
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            blocks.append(
                {"section": f"表格{index}", "text": "\n".join(rows), "type": "table"}
            )
    return blocks


def _pdf_blocks(path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[dict] = []
    for page_number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        page_blocks = _markdown_blocks(text, "正文")
        for block in page_blocks:
            block["page"] = page_number
        blocks.extend(page_blocks)
    return blocks



def _table_text(headers: list[str], rows: list[list[str]]) -> str:
    rendered = ["| " + " | ".join(str(item) for item in headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        rendered.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(rendered)


def _csv_blocks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return []
    headers, body = rows[0], rows[1:]
    return [
        {
            "section": path.stem,
            "text": _table_text(headers, body),
            "type": "table",
        }
    ]


def _xlsx_blocks(path: Path) -> list[dict]:
    import openpyxl

    workbook = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    blocks = []
    for sheet in workbook.worksheets:
        if sheet.title in {"文档信息", "说明"}:
            continue
        rows = []
        for row in sheet.iter_rows(values_only=True):
            values = ["" if cell is None else str(cell) for cell in row]
            if any(value.strip() for value in values):
                rows.append(values)
        if len(rows) < 2:
            continue
        headers = rows[0]
        width = len(headers)
        body = [row[:width] + [""] * max(0, width - len(row)) for row in rows[1:]]
        blocks.append(
            {"section": sheet.title, "text": _table_text(headers, body), "type": "table"}
        )
    workbook.close()
    return blocks


EXTRACTOR_VERSION = "2026.10-page-anchor"


def _annotate_blocks(blocks: list[dict], fmt: str) -> list[dict]:
    """为每个块保留可回到原页/原段落的定位信息。"""
    for index, block in enumerate(blocks):
        page = block.get("page")
        block["paragraph"] = index
        block["anchor"] = f"{fmt}:p{page if page is not None else 0}#b{index}"
        block["extractor_version"] = EXTRACTOR_VERSION
        block.setdefault("ocr", False)
    return blocks


def load_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        blocks = _docx_blocks(path)
    elif suffix == ".pdf":
        blocks = _pdf_blocks(path)
    elif suffix == ".csv":
        blocks = _csv_blocks(path)
    elif suffix == ".xlsx":
        blocks = _xlsx_blocks(path)
    else:
        blocks = _markdown_blocks(_read_text(path), "正文")
    return {"format": suffix.lstrip("."), "blocks": _annotate_blocks(blocks, suffix.lstrip("."))}
