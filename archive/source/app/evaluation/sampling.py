"""Run-bound question selection; no answers or model calls."""
import json
from pathlib import Path

from app.evaluation.files import digest


def selected_question_ids(directory: Path, manifest: dict, label: str) -> list[str]:
    entry = manifest.get('sampling', {}).get(label)
    if entry is None:
        return list(manifest['question_type_by_id'])
    path = directory/entry['file']
    if digest(path) != entry['sha256']:
        raise ValueError('抽样清单哈希不一致，不能改变本轮测试范围')
    qids = json.loads(path.read_text(encoding='utf-8'))['qids']
    if (len(qids) != entry['total'] or len(set(qids)) != len(qids)
            or not set(qids) <= manifest['question_type_by_id'].keys()):
        raise ValueError('抽样清单数量、唯一性或题库归属不符')
    return qids
