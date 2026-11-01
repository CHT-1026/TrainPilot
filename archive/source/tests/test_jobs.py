"""Task contract tests: in-memory storage, no Redis connection or key deletion."""
import json
import socket
import unittest
from unittest.mock import patch

from redis.exceptions import ConnectionError

from app.core.backpressure import OverloadedError
from app.domain.user.models import UserContext
from app.jobs import rag_jobs as jobs
from enterprise_rag.budget import RequestCancelled
from tests.job_fakes import FakePool, FakeRedis

USER = UserContext(user_id="job_u1", name="任务用户", role="employee", department="产品部")
OTHER = UserContext(user_id="job_u2", name="他人", role="employee", department="销售部")
ADMIN = UserContext(user_id="job_admin", name="管理员", role="admin")


class JobTestCase(unittest.TestCase):
    def setUp(self):
        self.client, self.pool = FakeRedis(), FakePool()
        self.enterContext(patch.object(jobs, "_redis", return_value=self.client))
        self.enterContext(patch.object(jobs, "_POOL", self.pool))
        self.enterContext(patch.object(jobs, "_ACCEPTING", True))
        self.enterContext(patch.object(jobs, "_FUTURES", {}))
        self.enterContext(patch.object(jobs, "_ADMITTING", set()))
        self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")))

    def data(self, job_id):
        return json.loads(self.client.hget(jobs._key(job_id), "payload"))

    def write(self, job_id, data):
        self.client.hset(jobs._key(job_id), "payload", json.dumps(data))


class RagJobTest(JobTestCase):
    def test_job_completes_and_is_owner_scoped(self):
        with patch("enterprise_rag.pipeline.query", return_value={"answer": "ok", "citations": []}):
            job = jobs.create_job(USER, "公司总部在哪里")
            self.assertEqual(jobs.get_job(USER, job.job_id).status, "queued")
            self.pool.run_one()
        finished = jobs.get_job(USER, job.job_id)
        self.assertEqual(finished.status, "succeeded")
        self.assertEqual(finished.result["answer"], "ok")
        self.assertIsNone(jobs.get_job(OTHER, job.job_id))
        self.assertEqual(jobs.get_job(ADMIN, job.job_id).owner, USER.user_id)
        self.assertEqual(len(jobs.list_jobs(USER)), 1)
        self.assertEqual(jobs.list_jobs(OTHER), [])

    def test_cold_load_and_worker_use_persisted_parameters(self):
        with patch("enterprise_rag.pipeline.query", return_value={"answer": "cold"}) as query:
            job = jobs.create_job(USER, "冷加载任务", use_llm=True, top_k=17)
            restored = jobs.get_job(USER, job.job_id)
            self.assertEqual((restored.owner, restored.use_llm, restored.top_k), (USER.user_id, True, 17))
            # Executor receives identity only; execution options come from authoritative payload.
            self.assertEqual(self.pool.calls[0][1], (job.job_id, USER))
            self.pool.run_one()
            self.assertEqual(query.call_args.kwargs["top_k"], 17)
            self.assertTrue(query.call_args.kwargs["use_llm"])
            rerun = jobs.create_job(USER, restored.question, use_llm=restored.use_llm, top_k=restored.top_k)
            self.pool.run_one()
        self.assertNotEqual(job.job_id, rerun.job_id)
        self.assertEqual(jobs.get_job(USER, rerun.job_id).result, {"answer": "cold"})

    def test_queued_cancel_is_terminal_without_waiting_for_executor(self):
        job = jobs.create_job(USER, "排队取消")
        result = jobs.cancel_job(USER, job.job_id)
        self.assertEqual(result.status, "cancelled")
        with patch("enterprise_rag.pipeline.query") as query:
            self.pool.run_one()
        query.assert_not_called()
        with patch.object(jobs, "MAX_PENDING_JOBS", 1):
            self.assertEqual(jobs.create_job(USER, "空出的容量").status, "queued")

    def test_running_cancel_wins_over_result_commit(self):
        def query(*args, cancel_check, **kwargs):
            self.assertFalse(cancel_check())
            self.assertEqual(jobs.cancel_job(USER, job.job_id).status, "cancelling")
            self.assertTrue(cancel_check())
            return {"answer": "late"}
        job = jobs.create_job(USER, "运行取消")
        with patch("enterprise_rag.pipeline.query", side_effect=query):
            self.pool.run_one()
        finished = jobs.get_job(USER, job.job_id)
        self.assertEqual(finished.status, "cancelled")
        self.assertIsNone(finished.result)
        stale = jobs._finish_job(job.job_id, "succeeded", {"answer": "stale"}, "", token="old")
        self.assertEqual(stale.status, "cancelled")
        self.assertIsNone(stale.result)

    def test_pipeline_cancel_exception_is_terminal(self):
        job = jobs.create_job(USER, "取消异常")
        with patch("enterprise_rag.pipeline.query", side_effect=RequestCancelled("cancelled")):
            self.pool.run_one()
        self.assertEqual(jobs.get_job(USER, job.job_id).status, "cancelled")

    def test_failed_query_is_observable_and_can_be_rerun(self):
        job = jobs.create_job(USER, "fake failure")
        with patch("enterprise_rag.pipeline.query", side_effect=ValueError("private failure text")):
            with self.assertLogs(jobs.logger, level="ERROR"):
                self.pool.run_one()
        finished = jobs.get_job(USER, job.job_id)
        self.assertEqual((finished.status, finished.error, finished.result), ("failed", "ValueError", None))
        with patch("enterprise_rag.pipeline.query", return_value={"answer": "recovered"}):
            rerun = jobs.create_job(USER, finished.question)
            self.pool.run_one()
        self.assertEqual(jobs.get_job(USER, rerun.job_id).status, "succeeded")

    def test_storage_failure_is_explicit_and_creation_is_atomic(self):
        self.client.execute_error = ConnectionError("down")
        with self.assertRaises(jobs.JobStorageError):
            jobs.create_job(USER, "写入失败")
        self.assertEqual(self.client.hashes, {})
        self.assertEqual(self.client.sets, {})
        self.assertEqual(self.pool.calls, [])
        with patch.object(self.client, "pipeline", side_effect=ConnectionError("down")):
            with self.assertRaises(jobs.JobStorageError):
                jobs.get_job(USER, "missing")
        with patch.object(jobs, "_redis", return_value=None):
            with self.assertRaises(jobs.JobStorageError):
                jobs.list_jobs(USER)

    def test_expired_worker_lease_is_failed_and_never_reclaimed(self):
        stale = jobs.Job("job_stale", USER.user_id, "崩溃遗留", status="running",
                         lease="old-worker", lease_expires_at=1)
        jobs._persist(stale)
        loaded = jobs.get_job(USER, stale.job_id)
        self.assertEqual((loaded.status, loaded.error, loaded.result), ("failed", "worker_lost", None))
        self.assertIsNone(jobs._mark_running(stale.job_id))
        self.assertFalse(jobs._renew_lease(stale.job_id, "old-worker"))
        self.assertEqual(jobs._finish_job(stale.job_id, "succeeded", {"late": True}, "", token="old-worker").status,
                         "failed")

    def test_enqueue_failure_is_terminal_and_service_unavailable(self):
        self.pool.error = RuntimeError("pool closed")
        with self.assertRaises(OverloadedError):
            jobs.create_job(USER, "入队失败")
        result = jobs.list_jobs(USER)
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].status, result[0].error), ("failed", "enqueue_failed"))

    def test_owner_index_cannot_expose_another_users_job(self):
        job = jobs.create_job(USER, "owner")
        self.client.sadd(jobs._owner_key(OTHER.user_id), job.job_id)
        self.assertEqual(jobs.list_jobs(OTHER), [])
        self.assertEqual(self.client.smembers(jobs._owner_key(OTHER.user_id)), set())

    def test_duplicate_identity_cannot_overwrite_existing_record(self):
        job = jobs.create_job(USER, "original")
        with self.assertRaises(jobs.JobStorageError):
            jobs._persist(jobs.Job(job.job_id, OTHER.user_id, "replacement"))
        self.assertEqual(jobs.get_job(USER, job.job_id).question, "original")


if __name__ == "__main__":
    unittest.main()
