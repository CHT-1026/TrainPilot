"""生产索引的只读工具：每次调用重新检查当前用户权限和版本，禁止模型指定身份。"""
import jsonschema

from agent.app.capabilities.retrieval.corpus_tools import SCHEMAS
from agent.app.capabilities.retrieval.index import BM25Index, DEFAULT_NAMESPACE, get_index
from agent.app.capabilities.retrieval.search import chunk_allowed
from agent.enterprise_rag import config
from agent.enterprise_rag.scope import EvidenceScope
from agent.enterprise_rag.store import load_index


class IndexTools:
    def __init__(self, user, *, as_of=None):
        self.user, self.as_of = user, as_of

    def _segments(self):
        index = load_index(config.RAG_NAMESPACE)
        scope = EvidenceScope.from_request(self.user, as_of=self.as_of,
                                          strict_effective_date=bool(self.as_of), comparison=True)
        # Keep historical versions available for model comparison; never bypass ACL or an explicit as_of.
        chunks = scope.filter(index.chunks)
        oa = get_index(namespace=DEFAULT_NAMESPACE)
        chunks += [c for c in oa.chunks if chunk_allowed(c, self.user)
                   and (not self.as_of or not c.get("effective_date") or str(c["effective_date"]) <= self.as_of)]
        segments = []
        for chunk in chunks:
            namespace = chunk.get("namespace") or DEFAULT_NAMESPACE
            doc_id = f"{namespace}:{chunk['doc_id']}"
            text = chunk.get("text") or ""
            for offset in range(0, len(text), 1000):
                locator = f"chunk:{chunk['chunk_id']}@{offset}"
                segments.append({"doc_id": doc_id, "title": chunk.get("title", ""),
                                 "locator": locator, "text": text[offset:offset + 1000]})
        return segments

    def run(self, name, arguments):
        if name not in SCHEMAS:
            return {"error": "unknown_tool", "name": name}
        try:
            jsonschema.validate(arguments, SCHEMAS[name])
        except jsonschema.ValidationError as error:
            return {"error": "invalid_arguments", "detail": error.message}
        segments = self._segments()
        if name == "search":
            index = BM25Index([{**s, "chunk_id": str(i)} for i, s in enumerate(segments)])
            hits = index.search(arguments["query"], top_k=arguments.get("top_k", 5))
            return {"hits": [{k: v for k, v in hit.items() if k != "chunk_id"} for hit in hits]}
        if name == "read":
            rows = [s for s in segments if s["doc_id"] == arguments["doc_id"]]
            if not rows:
                return {"error": "unknown_or_inaccessible_document"}
            start, size = arguments.get("offset", 0), arguments.get("limit", 40)
            return {"segments": rows[start:start + size], "total_segments": len(rows),
                    "next_offset": start + size if start + size < len(rows) else None}
        documents = {s["doc_id"]: {"doc_id": s["doc_id"], "title": s["title"]} for s in segments}
        keyword, offset = arguments.get("keyword", "").lower(), arguments.get("offset", 0)
        rows = [d for d in documents.values() if keyword in (d["doc_id"] + d["title"]).lower()]
        return {"documents": rows[offset:offset + 20],
                "next_offset": offset + 20 if offset + 20 < len(rows) else None}

    def validate_citations(self, citations):
        current = {(s["doc_id"], s["locator"]): s["text"] for s in self._segments()}
        for citation in citations:
            text = current.get((citation["doc_id"], citation["locator"]), "")
            if " ".join(citation["quote"].split()) not in " ".join(text.split()):
                raise ValueError("evidence_no_longer_accessible")
