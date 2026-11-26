"""评测隔离层：文件索引及SQLite替代存储，不替换模型、检索排序或业务规则。"""
from contextlib import contextmanager, ExitStack
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_chunks(output):
    from agent.enterprise_rag.loaders import load_document
    from agent.enterprise_rag.chunking import chunk_document
    from agent.app.capabilities.retrieval.parser import parse_document
    from agent.app.capabilities.retrieval.chunker import chunk_document as oa_chunk
    records = [json.loads(line) for line in (ROOT / "enterprise_rag/corpus/manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    enterprise, oa = [], []
    for r in records:
        path = ROOT / "enterprise_rag/corpus" / r["path"]
        loaded = load_document(path)
        chunks = chunk_document({"doc_id": r["doc_id"], "title": r["title"]}, loaded["blocks"])
        metadata = {"title": r["title"], "namespace": r.get("namespace", "lantu"),
                    "version": r["version"], "effective_date": r.get("effective_date"),
                    "doc_updated_at": r["updated_at"], "doc_created_at": r["created_at"],
                    "visibility": r.get("visibility", "all"), "department": r.get("department"),
                    "author": r["author"], "doc_family": r.get("doc_family"), "format": r.get("format"),
                    "category": r.get("category"), "source_path": str(path), "source_hash": sha(path)}
        enterprise.extend({**c, **metadata} for c in chunks)
    for path in sorted((ROOT / "data/oa_docs").iterdir()):
        if path.suffix.lower() not in {".txt", ".md", ".docx", ".pdf"}:
            continue
        document = parse_document(path)
        oa.extend({**c, "title": document["title"], "version": document["version"],
                   "namespace": "default", "visibility": "all", "department": None,
                   "source_path": str(path), "source_hash": sha(path)} for c in oa_chunk(document))
    save(output / "chunks.json", {"enterprise": enterprise, "oa": oa})
    return enterprise, oa


@contextmanager
def file_indexes(output, *, build=False):
    from agent.enterprise_rag.store import RagIndex
    from agent.app.capabilities.retrieval.index import BM25Index
    from agent.enterprise_rag.graph import build as graph_build, retrieve as graph_retrieve
    from agent.enterprise_rag.graph.store import GraphStore
    from agent.enterprise_rag import release
    data = read(output / "chunks.json")
    index = RagIndex(data["enterprise"])
    index.index_version = sha(output / "chunks.json")
    oa = BM25Index(data["oa"])
    graph_dir = output / "graph"
    graph_dir.mkdir(exist_ok=True)
    with ExitStack() as stack:
        stack.enter_context(patch("agent.enterprise_rag.retrieval.load_index", return_value=index))
        stack.enter_context(patch("agent.enterprise_rag.store.load_index", return_value=index))
        stack.enter_context(patch.object(graph_build, "load_index", return_value=index))
        stack.enter_context(patch.object(graph_build, "GRAPH_PATH", graph_dir / "graph.json"))
        stack.enter_context(patch.object(graph_build, "META_PATH", graph_dir / "graph_meta.json"))
        stack.enter_context(patch.object(graph_build, "VERSIONS_DIR", graph_dir / "versions"))
        stack.enter_context(patch.object(graph_retrieve, "GRAPH_PATH", graph_dir / "graph.json"))
        stack.enter_context(patch("agent.app.capabilities.retrieval.search.get_index", return_value=oa))
        stack.enter_context(patch.object(release, "CURRENT_DIR", output / "releases/current"))
        stack.enter_context(patch.object(release, "RELEASE_DIR", output / "releases"))
        if build:
            summary = graph_build.build_graph(force=True)
            graph = GraphStore.load(graph_dir / "graph.json")
            stats = graph.stats()
            candidate = release.prepare_release(
                index_version=index.index_version, graph_version=graph.version,
                corpus_checksum=sha(output / "chunks.json"),
                index_documents=len({c["doc_id"] for c in index.chunks}), index_chunks=len(index.chunks),
                graph_entities=stats["entities"], graph_relations=stats["relations"])
            # 测试快照落盘；跳过Redis缓存写入，权威文件路径仍使用原发布校验。
            with patch("agent.app.database.connection.get_redis_client", side_effect=RuntimeError("eval_no_redis")):
                release.activate_release(candidate, index_stats={"version": index.index_version,
                    "documents": len({c["doc_id"] for c in index.chunks}), "chunks": len(index.chunks)},
                    graph_stats=stats, corpus_checksum=sha(output / "chunks.json"))
            save(output / "index_summary.json", {"enterprise_documents": len({c["doc_id"] for c in index.chunks}),
                "enterprise_chunks": len(index.chunks), "oa_documents": len({c["doc_id"] for c in data["oa"]}),
                "oa_chunks": len(data["oa"]), "graph": summary,
                "embedding": "production default hash n-gram 384; not pretrained semantic embeddings",
                "storage": "JSON/file-backed indexes replacing MySQL loading only"})
        yield index, oa


SCHEMA = """
CREATE TABLE users(username TEXT PRIMARY KEY,name TEXT);
CREATE TABLE leave_applications(
 id TEXT PRIMARY KEY, employee_id TEXT, employee_name TEXT, employee_role TEXT, manager_id TEXT,
 department TEXT, leave_type TEXT, leave_days INTEGER, start_date TEXT, end_date TEXT, reason TEXT,
 status TEXT DEFAULT 'pending', manager_approval TEXT DEFAULT 'pending', hr_approval TEXT DEFAULT 'pending',
 approval_step TEXT DEFAULT 'manager', version INTEGER DEFAULT 0, idempotency_key TEXT,
 comments TEXT DEFAULT '', submitted_at TEXT, cancelled_at TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE approval_audit(id INTEGER PRIMARY KEY AUTOINCREMENT, application_id TEXT,action TEXT,
 actor_id TEXT,actor_role TEXT,from_status TEXT,to_status TEXT,comment TEXT,trace_id TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE idempotency_keys(idempotency_key TEXT PRIMARY KEY,user_id TEXT,endpoint TEXT,
 response_json TEXT, expires_at TEXT);
"""


class SQLiteSandbox:
    """串行SQLite适配原repository SQL；不验证MySQL锁/方言/并发行为。"""
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = RLock()
        self.fail_next_transaction = False

    @contextmanager
    def connection(self):
        with self.lock:
            yield self

    @contextmanager
    def transaction(self):
        with self.lock:
            if self.fail_next_transaction:
                self.fail_next_transaction = False
                raise sqlite3.OperationalError("isolated_test_injected_connection_failure")
            self.db.execute("BEGIN")
            try:
                yield self
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def cursor(self):
        return SandboxCursor(self.db.cursor())

    def rows(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM leave_applications ORDER BY id")]

    def audit(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM approval_audit ORDER BY id")]

    def seed(self, app_id, *, owner="eval_employee", start="2027-03-01", end="2027-03-02"):
        from agent.app.domain.leave import repository
        repository.insert_application(self, app_id=app_id, employee_id=owner, employee_name="隔离测试员工",
            employee_role="employee", manager_id="eval_manager", department="技术部", leave_type="事假",
            leave_days=2, start_date=date.fromisoformat(start), end_date=date.fromisoformat(end),
            reason="既有测试记录", idempotency_key=None)
        self.db.execute("INSERT OR IGNORE INTO users VALUES (?,?)", (owner,"隔离测试员工"))


class SandboxCursor:
    def __init__(self, cursor): self.inner = cursor
    def __enter__(self): return self
    def __exit__(self, *args): self.inner.close()
    @property
    def rowcount(self): return self.inner.rowcount

    def execute(self, sql, params=()):
        sql = sql.replace("%s", "?").replace(" FOR UPDATE", "")
        sql = sql.replace("DATE_ADD(NOW(), INTERVAL ? HOUR)", "datetime('now', '+' || ? || ' hours')")
        sql = sql.replace("NOW()", "CURRENT_TIMESTAMP")
        if "ON DUPLICATE KEY UPDATE" in sql:
            assert "INSERT INTO idempotency_keys" in sql
            sql = sql.split("ON DUPLICATE KEY UPDATE")[0] + " ON CONFLICT(idempotency_key) DO UPDATE SET response_json=excluded.response_json, expires_at=excluded.expires_at"
        params = tuple(p.isoformat() if isinstance(p, (date, datetime)) else p for p in params)
        self.inner.execute(sql, params)

    def fetchone(self):
        row = self.inner.fetchone()
        if row is None: return None
        data = dict(row)
        for key in ("start_date", "end_date"):
            if data.get(key): data[key] = date.fromisoformat(data[key])
        return data

    def fetchall(self):
        rows = []
        while (row := self.fetchone()) is not None: rows.append(row)
        return rows
