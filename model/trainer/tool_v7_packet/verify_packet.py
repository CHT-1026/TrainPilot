"""Verify shipped immutable code, data and initial V6 adapter."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'packet_manifest.json').read_text(encoding='utf-8'))
bad=[name for name,digest in manifest.items()
     if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
if bad:
    raise SystemExit('PACKET_HASH_MISMATCH: '+repr(bad))
print('PACKET_OK',len(manifest),'files')
