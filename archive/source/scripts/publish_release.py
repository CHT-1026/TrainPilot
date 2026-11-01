"""发布企业 RAG 快照：绑定语料校验、向量索引版本、图谱版本与 embedding 配置。

用法（backend 目录）：
    .venv312\\Scripts\\python.exe scripts\\publish_release.py
    .venv312\\Scripts\\python.exe scripts\\publish_release.py --verify-only
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env.local", override=False)
load_dotenv(ROOT / ".env", override=False)

from enterprise_rag import config  # noqa: E402
from enterprise_rag.graph.retrieve import load_graph  # noqa: E402
from enterprise_rag.release import (  # noqa: E402
    active_release,
    manifest_checksum,
    prepare_release,
    activate_release,
    verify_release,
)
from enterprise_rag.store import index_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default=config.RAG_NAMESPACE)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    stats = index_stats(args.namespace)
    graph_stats = load_graph().stats()
    checksum = manifest_checksum()
    current = active_release(args.namespace)
    if args.verify_only:
        issues = verify_release(current, index_stats=stats, graph_stats=graph_stats,
                                corpus_checksum=checksum)
        print(json.dumps({"release": current, "issues": issues}, ensure_ascii=False, indent=2))
        sys.exit(1 if issues else 0)

    candidate = prepare_release(
        args.namespace,
        index_version=stats.get("index_version", ""),
        graph_version=graph_stats.get("version", ""),
        corpus_checksum=checksum,
        index_documents=stats.get("documents"),
        index_chunks=stats.get("chunks"),
        graph_entities=graph_stats.get("entities"),
        graph_relations=graph_stats.get("relations"),
    )
    issues = verify_release(candidate, index_stats=stats, graph_stats=graph_stats,
                            corpus_checksum=checksum)
    if issues:
        print(json.dumps({"candidate": candidate, "issues": issues}, ensure_ascii=False, indent=2))
        sys.exit(1)
    release = activate_release(candidate, index_stats=stats, graph_stats=graph_stats,
                               corpus_checksum=checksum)
    print(json.dumps({"release": release, "issues": issues}, ensure_ascii=False, indent=2))
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
