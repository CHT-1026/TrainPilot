"""Regression contracts for access invalidation, release activation and cancelled work."""
import asyncio
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.domain.user.models import UserContext
from app.capabilities.retrieval import index as oa_index, kb_admin
from app.core.backpressure import inflight
from enterprise_rag import api, config, release, store
from enterprise_rag.scope import EvidenceScope

ADMIN = UserContext(user_id="revision-admin", name="test", role="admin", department="test")
EMP = UserContext(user_id="revision-employee", name="test", role="employee", department="test")


class NamespaceAndRevisionTest(unittest.TestCase):
    def test_admin_cannot_cross_explicit_namespace(self):
        self.assertFalse(EvidenceScope(ADMIN, namespace="lantu").allows({"namespace": "other", "visibility": "all"}))

    def test_directory_filters_namespace_and_returns_it(self):
        conn = MagicMock(); cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [{"id": "x", "namespace": "lantu", "chunks": 2}]
        with patch.object(kb_admin, "connection") as connect:
            connect.return_value.__enter__.return_value = conn
            rows = kb_admin.list_documents("lantu")
        sql, args = cursor.execute.call_args.args
        self.assertIn("WHERE d.namespace = %s", sql)
        self.assertEqual(args, ("lantu",)); self.assertEqual(rows[0]["namespace"], "lantu")

    def test_update_cannot_modify_document_in_another_namespace(self):
        conn = MagicMock(); cursor = conn.cursor.return_value.__enter__.return_value
        with patch.object(kb_admin, "transaction") as txn, patch.object(kb_admin, "_get_document", return_value=None) as read:
            txn.return_value.__enter__.return_value = conn
            self.assertIsNone(kb_admin.update_document("x", {"visibility": "admin"}, "lantu"))
        sql, args = cursor.execute.call_args.args
        self.assertIn("AND namespace = %s", sql); self.assertEqual(args[-2:], ("x", "lantu"))
        read.assert_called_once_with("x", "lantu")

    def test_oa_index_reloads_acl_even_when_redis_revision_is_unchanged(self):
        old = {"chunk_id":"d#1", "doc_id":"d", "text":"制度", "visibility":"all"}
        revoked = {**old, "visibility":"admin"}
        with patch.dict(oa_index._indexes, {}, clear=True), patch.dict(oa_index._index_versions, {}, clear=True), \
             patch.dict(oa_index._access_versions, {}, clear=True), \
             patch.object(oa_index, "_current_version", return_value="same"), \
             patch.object(oa_index, "access_revision", side_effect=["a", "b"]), \
             patch.object(oa_index, "_load_chunks", side_effect=[[old], [revoked]]):
            first=oa_index.get_index("lantu"); second=oa_index.get_index("lantu")
        self.assertIsNot(first, second); self.assertEqual(second.chunks[0]["visibility"], "admin")

    def test_enterprise_index_reloads_acl_without_notification(self):
        first, second = SimpleNamespace(chunks=[],family_docs={}), SimpleNamespace(chunks=[],family_docs={})
        with patch.dict(store._indexes, {}, clear=True), patch.dict(store._versions, {}, clear=True), \
             patch("app.capabilities.retrieval.revision.access_revision", side_effect=["a","b"]), \
             patch.object(store, "_current_version", return_value="same"), \
             patch.object(store, "_load_chunks", return_value=[]), \
             patch.object(store, "RagIndex", side_effect=[first, second]):
            a=store.load_index("lantu"); b=store.load_index("lantu")
        self.assertIsNot(a,b); self.assertEqual(b.access_revision,"b")

    def test_access_revision_failure_does_not_serve_cached_acl(self):
        with patch.dict(store._indexes, {"lantu":SimpleNamespace(chunks=[])}, clear=True), \
             patch("app.capabilities.retrieval.revision.access_revision", side_effect=RuntimeError("db down")):
            with self.assertRaises(RuntimeError): store.load_index("lantu")


class ReleaseActivationTest(unittest.TestCase):
    def candidate(self):
        return release.prepare_release("test", index_version="i1", graph_version="g1", corpus_checksum="c1",
                                       index_documents=2, index_chunks=3, graph_entities=4, graph_relations=5)

    def test_prepare_does_not_activate(self):
        with patch.object(release,"_write_release") as write:
            self.candidate()
        write.assert_not_called()

    def test_invalid_candidate_preserves_current_pointer(self):
        with patch.object(release,"_write_release") as write:
            with self.assertRaises(ValueError):
                release.activate_release(self.candidate(),index_stats={"index_version":"i2"},graph_stats={"version":"g1"},corpus_checksum="c1")
        write.assert_not_called()

    def test_matching_actual_resources_activate_once(self):
        with patch.object(release,"_write_release",side_effect=lambda ns,doc:doc) as write:
            candidate=self.candidate()
            result=release.activate_release(candidate,index_stats={"index_version":"i1","documents":2,"chunks":3},graph_stats={"version":"g1","entities":4,"relations":5},corpus_checksum="c1")
        self.assertEqual(result,candidate); write.assert_called_once_with("test",candidate)

    def test_verify_rejects_unpublished_empty_binding_and_changed_checksum(self):
        self.assertIn("STATUS_INVALID", {i["code"] for i in release.verify_release({"status":"unpublished"})})
        candidate=self.candidate();candidate["index_version"]=""
        codes={i["code"] for i in release.verify_release(candidate,index_stats={"index_version":"i1"},corpus_checksum="new")}
        self.assertIn("INDEX_VERSION_MISMATCH",codes);self.assertIn("CORPUS_CHECKSUM_MISMATCH",codes)


class CancelledCapacityTest(unittest.IsolatedAsyncioTestCase):
    async def test_queued_executor_cancel_keeps_slot_until_real_work_finishes(self):
        gate=threading.Event(); entered=threading.Event(); seen=[]
        executor=ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(executor)
        blocker=executor.submit(lambda: (entered.set(),gate.wait(3)))
        self.assertTrue(entered.wait(1))
        def fake_query(*args,**kwargs):
            seen.append(kwargs["cancel_check"]()); return {"answer":"stopped"}
        try:
            with patch.object(api,"run_query",side_effect=fake_query):
                task=asyncio.create_task(api.rag_query(api.RagQueryRequest(question="test"),EMP))
                await asyncio.sleep(.02)
                self.assertEqual(inflight("rag"),1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):await task
                self.assertEqual(inflight("rag"),1)
                gate.set()
                for _ in range(100):
                    if inflight("rag")==0:break
                    await asyncio.sleep(.01)
                self.assertEqual(inflight("rag"),0);self.assertEqual(seen,[True])
        finally:
            gate.set();blocker.result(3);executor.shutdown(wait=True)


class ReadinessTest(unittest.IsolatedAsyncioTestCase):
    async def test_default_kb_ready_does_not_hide_enterprise_dependency_failure(self):
        from main import ready
        conn=MagicMock()
        with patch("main.get_sync_db_connection",return_value=conn), \
             patch("main.get_redis_client"), \
             patch("app.capabilities.retrieval.index.index_stats",return_value={"namespace":"default","chunks":75}), \
             patch("enterprise_rag.store.index_stats",side_effect=RuntimeError("index unavailable")):
            response=await ready()
        self.assertEqual(response.status_code,503)
        self.assertIn("enterprise_rag",json.loads(response.body)["checks"])

    async def test_matching_enterprise_resources_are_ready(self):
        from main import ready
        doc={"status":"active","namespace":"lantu","snapshot_id":"s1","index_version":"i1","graph_version":"g1", "corpus_checksum":"c1","embedding_model":config.EMBEDDING_MODEL,"embedding_dim":config.EMBEDDING_DIM}
        with patch("main.get_sync_db_connection",return_value=MagicMock()), patch("main.get_redis_client"), \
             patch("app.capabilities.retrieval.index.index_stats",return_value={"namespace":"default","chunks":75}), \
             patch("enterprise_rag.store.index_stats",return_value={"namespace":"lantu","chunks":616,"index_version":"i1"}), \
             patch("enterprise_rag.graph.retrieve.load_graph",return_value=SimpleNamespace(stats=lambda:{"version":"g1"})), \
             patch("enterprise_rag.release.active_release",return_value=doc), \
             patch("enterprise_rag.release.manifest_checksum",return_value="c1"):
            response=await ready()
        self.assertEqual(response.status_code,200)


if __name__=="__main__":unittest.main()
