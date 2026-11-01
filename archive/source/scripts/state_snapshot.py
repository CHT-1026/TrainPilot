"""生成版本化项目状态制品：代码、配置、语料、索引、图谱、提示词与发布快照。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe scripts\\state_snapshot.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

OUT = ROOT / "audit_artifacts" / "state"


def _cmd(*args: str) -> str:
    try:
        return subprocess.run(list(args), cwd=ROOT, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    from enterprise_rag import config
    from enterprise_rag.graph.retrieve import load_graph
    from enterprise_rag.prompts import PROMPT_VERSIONS
    from enterprise_rag.release import active_release, manifest_checksum
    from enterprise_rag.store import index_stats

    manifest = [json.loads(line) for line in config.MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    try:
        stats = index_stats()
    except Exception as exc:
        stats = {"error": type(exc).__name__}
    graph = load_graph()
    state = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "git_head": _cmd("git", "rev-parse", "HEAD"),
        "git_status": _cmd("git", "status", "--short").splitlines(),
        "config": {
            "model": config.LLM_MODEL,
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": config.EMBEDDING_DIM,
            "namespace": config.RAG_NAMESPACE,
            "top_k": config.TOP_K,
            "min_score": config.MIN_SCORE,
            "feature_flags": {
                "multi_representation": config.MULTI_REPRESENTATION,
                "neighbor_expansion": config.NEIGHBOR_EXPANSION,
                "context_compression": config.CONTEXT_COMPRESSION,
                "cross_encoder": config.CROSS_ENCODER_ENABLED,
                "hyde": config.HYDE_ENABLED,
            },
        },
        "corpus": {
            "manifest_documents": len(manifest),
            "manifest_checksum": manifest_checksum(),
            "formats": sorted({item.get("format") for item in manifest if item.get("format")}),
            "visibilities": sorted({item.get("visibility") for item in manifest if item.get("visibility")}),
        },
        "index": stats,
        "graph": graph.stats(),
        "release": active_release(),
        "prompt_versions": PROMPT_VERSIONS,
        "notes": [
            "本文件由脚本生成，不包含任何凭据值",
            "配置以当前进程环境与 .env.local 为准；旧报告中的模型/向量数字不能当作现状",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT / f"state_{stamp}.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = OUT / "latest.json"
    latest.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "documents": len(manifest),
                      "chunks": stats.get("chunks"), "graph_entities": len(graph.entities),
                      "graph_relations": len(graph.relations)}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
