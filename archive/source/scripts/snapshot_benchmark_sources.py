"""Read primary documents in order, with reproducible physical locators. No model I/O."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from openpyxl import load_workbook
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'enterprise_rag/corpus'
OUT = ROOT / 'research/benchmark_audit_20261030'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(path):
    section = ''
    segments = []
    for line, text in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if text.startswith('#'):
            section = text.lstrip('#').strip()
        segments.append({'locator': f'line:{line}', 'section': section, 'text': text})
    return segments, {}


def word(path):
    doc = Document(path)
    segments, paragraph, table, section = [], 0, 0, ''
    for block in doc.element.body.iterchildren():
        if block.tag == qn('w:p'):
            paragraph += 1
            p = Paragraph(block, doc)
            if p.style.name.startswith('Heading') or p.style.name == 'Title':
                section = p.text
            segments.append({'locator': f'paragraph:{paragraph}', 'section': section, 'text': p.text})
        elif block.tag == qn('w:tbl'):
            table += 1
            for row, cells in enumerate(Table(block, doc).rows, 1):
                segments.append({'locator': f'table:{table}/row:{row}', 'section': section,
                                 'text': ' | '.join(c.text for c in cells.cells)})
    revisions = len(list(doc.element.iter(qn('w:ins')))) + len(list(doc.element.iter(qn('w:del'))))
    return segments, {'paragraphs': paragraph, 'tables': table, 'tracked_change_elements': revisions}


def pdf(path):
    reader = PdfReader(path)
    return ([{'locator': f'physical_page:{i}', 'section': '', 'text': page.extract_text()}
             for i, page in enumerate(reader.pages, 1)], {'physical_pages': len(reader.pages)})


def spreadsheet(path):
    formulas = load_workbook(path, read_only=True, data_only=False)
    values = load_workbook(path, read_only=True, data_only=True)
    segments, calculation_cells = [], []
    for ws in formulas:
        value_sheet = values[ws.title]
        for cells in ws.iter_rows():
            nonempty = [cell for cell in cells if cell.value is not None]
            if not nonempty:
                continue
            texts = []
            for cell in nonempty:
                value = cell.value
                if cell.data_type == 'f':
                    cached = value_sheet[cell.coordinate].value
                    calculation_cells.append({'sheet': ws.title, 'cell': cell.coordinate,
                                              'formula': value, 'cached_value': cached})
                    value = cached
                texts.append(f'{cell.coordinate}={value}')
            segments.append({'locator': f'sheet:{ws.title}/range:{nonempty[0].coordinate}:{nonempty[-1].coordinate}',
                             'section': ws.title, 'text': ' | '.join(texts),
                             'values': [cell.value for cell in cells]})
    formulas.close()
    values.close()
    return segments, {'formula_cells': calculation_cells}


def comma_separated(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.reader(source))
    return ([{'locator': f'row:{i}', 'section': 'CSV', 'text': ' | '.join(row), 'values': row}
             for i, row in enumerate(rows, 1)], {'header': rows[0]})


READERS = {'.md': markdown, '.docx': word, '.pdf': pdf, '.xlsx': spreadsheet, '.csv': comma_separated}


def main():
    records = []
    for meta in map(json.loads, (CORPUS/'manifest.jsonl').read_text(encoding='utf-8').splitlines()):
        path = CORPUS/meta['path']
        segments, properties = READERS[path.suffix.lower()](path)
        records.append({'doc_id': meta['doc_id'], 'title': meta['title'], 'category': meta['category'],
                        'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                        'manifest_checksum_matches': sha(path) == meta['checksum'],
                        'metadata': meta, 'properties': properties, 'segments': segments})
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'sources.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False, default=str)+'\n' for r in records), encoding='utf-8')
    packets = OUT/'source_packets'
    packets.mkdir(exist_ok=True)
    for category in sorted({r['category'] for r in records}):
        text = []
        for doc in records:
            if doc['category'] != category:
                continue
            text.append(f"\n## {doc['doc_id']} | {doc['title']} | {doc['path']}\n")
            text.append(json.dumps({k: doc['metadata'][k] for k in ('created_at', 'updated_at', 'effective_date', 'version', 'status')}, ensure_ascii=False))
            text.extend(f"[{s['locator']}] {s['text']}" for s in doc['segments'] if s['text'])
        (packets/f'{category}.md').write_text('\n'.join(text), encoding='utf-8')
    receipt = {'created_at': datetime.now(timezone.utc).isoformat(), 'documents': len(records),
               'formats': dict(Counter(Path(r['path']).suffix for r in records)),
               'manifest_mismatches': [r['doc_id'] for r in records if not r['manifest_checksum_matches']],
               'source_snapshot_sha256': sha(target),
               'original_bank_sha256': {name: sha(ROOT/'eval_bench'/name) for name in
                                        ('question_bank.jsonl', 'answer_bank.jsonl', 'grading_rules.jsonl')},
               'model_requests': 0}
    (OUT/'source_snapshot_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
