"""Regression contracts for access invalidation, release activation and cancelled work."""
import asyncio
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.domain.user.models import UserContext
from app.capabilities.retrieval import index as oa_index, kb_admin
from app.core.backpressure import inflight
from enterprise_rag import api, config, release, store
from enterprise_rag.scope import EvidenceScope

ADMIN = UserContext(user_id="revision-admin", name="test", role="admin", department="test")
EMP = UserContext(user_id="revision-employee", name="test", role="employee", department="test")


class NamespaceAndRevisionTest(unittest.TestCase):
    def test_statistics_keep_the_resource_handle_version_during_refresh(self):
        oa = oa_index.BM25Index([]); oa.index_version = "old"
        enterprise = store.RagIndex([]); enterprise.index_version = "old"
        with patch.object(oa_index, "get_index", return_value=oa), \
             patch.dict(oa_index._index_versions, {"test":"new"}), \
             patch.object(store, "load_index", return_value=enterprise), \
             patch.dict(store._versions, {"test":"new"}):
            self.assertEqual(oa_index.index_stats("test")["version"], "old")
            self.assertEqual(store.index_stats("test")["index_version"], "old")

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
            result=release.activate_release(candidate,index_stats={"namespace":"test","index_version":"i1","documents":2,"chunks":3},graph_stats={"namespace":"test","version":"g1","entities":4,"relations":5},corpus_checksum="c1")
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
             patch("enterprise_rag.graph.retrieve.load_graph",return_value=SimpleNamespace(stats=lambda:{"namespace":"lantu","version":"g1"})), \
             patch("enterprise_rag.release.active_release",return_value=doc), \
             patch("enterprise_rag.release.manifest_checksum",return_value="c1"):
            response=await ready()
        self.assertEqual(response.status_code,200)

    async def test_dependency_errors_do_not_expose_raw_details_anonymously(self):
        from main import ready
        secret="MOCK_PASSWORD_EXPOSED"
        with patch("main.get_sync_db_connection",side_effect=RuntimeError(secret)), \
             patch("main.get_redis_client",side_effect=RuntimeError(secret)), \
             patch("app.capabilities.retrieval.index.index_stats",side_effect=RuntimeError(secret)), \
             patch("enterprise_rag.store.index_stats",side_effect=RuntimeError(secret)):
            response=await ready()
        self.assertEqual(response.status_code,503)
        self.assertNotIn(secret,response.body.decode())


class StartupCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_graph_initialization_failure_still_closes_dependencies(self):
        import main
        conn=MagicMock()
        with patch.object(main,"get_sync_db_connection",return_value=conn), \
             patch.object(main,"init_graphs",new=AsyncMock(side_effect=RuntimeError("init failed"))), \
             patch("app.jobs.rag_jobs.start_job_worker") as start, \
             patch("app.jobs.rag_jobs.shutdown_job_worker") as shutdown, \
             patch.object(main,"close_async_redis",new=AsyncMock()) as close_redis, \
             patch.object(main,"close_all_connections") as close_db:
            with self.assertRaisesRegex(RuntimeError,"init failed"):
                async with main.lifespan(main.app): pass
        conn.close.assert_called_once(); start.assert_not_called()
        shutdown.assert_called_once();close_redis.assert_awaited_once();close_db.assert_called_once()

    async def test_worker_shutdown_failure_does_not_skip_other_cleanup(self):
        import main
        with patch.object(main,"get_sync_db_connection",return_value=MagicMock()), \
             patch.object(main,"init_graphs",new=AsyncMock()), \
             patch("app.jobs.rag_jobs.start_job_worker"), \
             patch("app.jobs.rag_jobs.shutdown_job_worker",side_effect=RuntimeError("pool failed")), \
             patch.object(main,"close_async_redis",new=AsyncMock()) as close_redis, \
             patch.object(main,"close_all_connections") as close_db:
            with self.assertRaisesRegex(RuntimeError,"pool failed"):
                async with main.lifespan(main.app): pass
        close_redis.assert_awaited_once();close_db.assert_called_once()

    async def test_async_redis_cleanup_failure_still_closes_mysql(self):
        import main
        with patch.object(main,"get_sync_db_connection",return_value=MagicMock()), \
             patch.object(main,"init_graphs",new=AsyncMock()), \
             patch("app.jobs.rag_jobs.start_job_worker"),patch("app.jobs.rag_jobs.shutdown_job_worker"), \
             patch.object(main,"close_async_redis",new=AsyncMock(side_effect=RuntimeError("redis close failed"))), \
             patch.object(main,"close_all_connections") as close_db:
            with self.assertRaisesRegex(RuntimeError,"redis close failed"):
                async with main.lifespan(main.app): pass
        close_db.assert_called_once()


class IndexAndAdmissionFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_executor_rejection_releases_unstarted_admission_once(self):
        from app.core.backpressure import OverloadedError
        loop=asyncio.get_running_loop()
        with patch.object(loop,"run_in_executor",side_effect=RuntimeError("executor closed")), \
             patch.object(api,"run_query") as query:
            with self.assertRaises(OverloadedError):
                await api.rag_query(api.RagQueryRequest(question="test"),EMP)
        query.assert_not_called(); self.assertEqual(inflight("rag"),0)

    async def test_lost_revision_store_does_not_become_empty_version(self):
        from app.capabilities.retrieval.revision import IndexRevisionUnavailable
        for module in (store,oa_index):
            with patch.object(module,"get_redis_client",side_effect=RuntimeError("redis down")):
                with self.assertRaises(IndexRevisionUnavailable):module._current_version("lantu")

    async def test_published_index_with_missing_version_is_not_loaded(self):
        from app.capabilities.retrieval.revision import IndexRevisionUnavailable
        with patch("app.capabilities.retrieval.revision.access_revision",return_value="acl"), \
             patch.object(store,"_current_version",return_value=""), \
             patch("enterprise_rag.release.active_release",return_value={"status":"active","index_version":"published"}), \
             patch.object(store,"_load_chunks") as chunks:
            with self.assertRaises(IndexRevisionUnavailable):store.load_index("lantu")
        chunks.assert_not_called()

    async def test_activation_rejects_wrong_namespace_or_missing_resources(self):
        candidate={"status":"active","namespace":"lantu","index_version":"i1","graph_version":"g1","corpus_checksum":"c"}
        for stats,graph in [({"namespace":"other","index_version":"i1"},{"version":"g1"}),({}, {})]:
            with patch.object(release,"_write_release") as write:
                with self.assertRaises(ValueError):release.activate_release(candidate,index_stats=stats,graph_stats=graph,corpus_checksum="c")
            write.assert_not_called()

    async def test_release_namespace_cannot_escape_its_directory(self):
        with self.assertRaises(ValueError):release.release_path("../../outside")

    async def test_metadata_dependency_failure_maps_to_index_unavailable(self):
        from app.capabilities.retrieval.revision import access_revision, IndexRevisionUnavailable
        with patch("app.capabilities.retrieval.revision.connection",side_effect=RuntimeError("MOCK_SECRET")):
            with self.assertRaises(IndexRevisionUnavailable) as caught:access_revision("lantu")
        self.assertIsInstance(caught.exception.__cause__,RuntimeError)
        self.assertNotIn("MOCK_SECRET",str(caught.exception))

    async def test_index_dependency_error_has_registered_safe_503_handler(self):
        import main
        from app.capabilities.retrieval.revision import IndexRevisionUnavailable
        handler=main.app.exception_handlers[IndexRevisionUnavailable]
        response=await handler(None,IndexRevisionUnavailable("MOCK_SECRET"))
        self.assertEqual(response.status_code,503)
        self.assertNotIn("MOCK_SECRET",response.body.decode())


if __name__=="__main__":unittest.main()
