"""Independent-worker races, fencing, migration, cancellation and shutdown."""
import importlib.util
import json
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

from redis.exceptions import ConnectionError, WatchError

from app.core.backpressure import OverloadedError
from app.jobs import rag_jobs as jobs
from app.routes import jobs as routes
from tests.test_jobs import JobTestCase, OTHER, USER


class RaceTest(JobTestCase):
    def competing_commits(self):
        barrier = threading.Barrier(2)
        seen = set()
        lock = threading.Lock()

        def hook(pipe):
            with lock:
                identity = threading.get_ident()
                first = identity not in seen
                seen.add(identity)
            if first:
                barrier.wait(timeout=3)
        self.client.before_execute = hook

    def test_independent_workers_claim_queued_only_once(self):
        job = jobs.create_job(USER, "once")
        self.competing_commits()
        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda _: jobs._mark_running(job.job_id), range(2)))
        winners = [value for value in claims if value is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(self.data(job.job_id)["lease"], winners[0].lease)
        self.client.before_execute = None
        with patch("enterprise_rag.pipeline.query") as query:
            jobs._run(job.job_id, USER)
        query.assert_not_called()

    def test_same_owner_capacity_is_atomic_across_process_local_pools(self):
        # Separate modules have separate locks/pools, sharing only authoritative Redis.
        name = "rag_job_test_second_process"
        spec = importlib.util.spec_from_file_location(name, Path(jobs.__file__))
        second = importlib.util.module_from_spec(spec)
        sys.modules[name] = second
        spec.loader.exec_module(second)
        self.addCleanup(sys.modules.pop, name, None)
        self.addCleanup(second._POOL.shutdown, wait=True)
        self.enterContext(patch.object(second, "_POOL", self.pool))
        self.enterContext(patch.object(second, "_redis", return_value=self.client))
        self.enterContext(patch.object(second, "MAX_PENDING_JOBS", 1))
        self.enterContext(patch.object(jobs, "MAX_PENDING_JOBS", 1))
        self.competing_commits()

        def create(module):
            try:
                return module.create_job(USER, "concurrent capacity").status
            except OverloadedError:
                return "overloaded"
        with ThreadPoolExecutor(max_workers=2) as executor:
            result = list(executor.map(create, [jobs, second]))
        self.assertCountEqual(result, ["queued", "overloaded"])
        self.assertEqual(len(self.client.hashes), 1)
        self.assertEqual(len(self.pool.calls), 1)

    def test_fencing_rejects_wrong_token_and_accepts_current_owner(self):
        job = jobs.create_job(USER, "fence")
        running = jobs._mark_running(job.job_id)
        data = self.data(job.job_id)
        data["lease"] = "new-execution"
        self.write(job.job_id, data)
        self.assertTrue(jobs._cancel_requested(running))
        self.assertFalse(jobs._renew_lease(job.job_id, running.lease))
        for token in (running.lease, ""):
            stale = jobs._finish_job(job.job_id, "succeeded", {"stale": True}, "", token=token)
            self.assertEqual(stale.status, "running")
            self.assertIsNone(stale.result)
        good = jobs._finish_job(job.job_id, "succeeded", {"current": True}, "", token="new-execution")
        self.assertEqual((good.status, good.result), ("succeeded", {"current": True}))

    def test_cancel_race_immediately_before_finish_exec_is_not_lost(self):
        job = jobs.create_job(USER, "commit race")
        running = jobs._mark_running(job.job_id)
        fired = []

        def hook(pipe):
            if not fired:
                fired.append(True)
                self.assertEqual(jobs.cancel_job(USER, job.job_id).status, "cancelling")
        self.client.before_execute = hook
        finished = jobs._finish_job(job.job_id, "succeeded", {"late": True}, "", token=running.lease)
        self.assertEqual((finished.status, finished.result), ("cancelled", None))

    def test_cancelling_lease_renews_until_real_worker_finishes(self):
        job = jobs.create_job(USER, "keep cancelling")
        running = jobs._mark_running(job.job_id)
        jobs.cancel_job(USER, job.job_id)
        before = jobs.get_job(USER, job.job_id).lease_expires_at
        with patch.object(jobs.time, "time", return_value=before - 1):
            self.assertTrue(jobs._renew_lease(job.job_id, running.lease))
        current = jobs.get_job(USER, job.job_id)
        self.assertEqual(current.status, "cancelling")
        self.assertGreater(current.lease_expires_at, before)

    def test_renewal_cannot_resurrect_expired_lease(self):
        job = jobs.create_job(USER, "expire")
        running = jobs._mark_running(job.job_id)
        with patch.object(jobs.time, "time", return_value=running.lease_expires_at + 1):
            self.assertFalse(jobs._renew_lease(job.job_id, running.lease))
        loaded = jobs.get_job(USER, job.job_id)
        self.assertEqual((loaded.status, loaded.error), ("failed", "worker_lost"))

    def test_queue_wait_deadline_is_distinct_from_lost_running_worker(self):
        job = jobs.create_job(USER, "queue wait expired")
        deadline = jobs.get_job(USER, job.job_id).lease_expires_at
        with patch.object(jobs.time, "time", return_value=deadline + 1):
            loaded = jobs.get_job(USER, job.job_id)
            with patch("enterprise_rag.pipeline.query") as query:
                self.pool.run_one()
            query.assert_not_called()
        self.assertEqual((loaded.status, loaded.error), ("failed", "queue_timeout"))

    def test_lost_storage_is_not_interpreted_as_no_cancel(self):
        job = jobs.create_job(USER, "authority lost")
        running = jobs._mark_running(job.job_id)
        with patch.object(self.client, "pipeline", side_effect=ConnectionError("down")):
            with self.assertRaises(jobs.JobStorageError):
                jobs._cancel_requested(running)
        self.client.delete(jobs._key(job.job_id))
        self.assertTrue(jobs._cancel_requested(running))

    def test_heartbeat_failure_sets_local_stop_signal(self):
        class OneTick:
            calls = 0
            def wait(self, seconds):
                self.calls += 1
                return self.calls > 1
        lost = threading.Event()
        with patch.object(jobs, "_renew_lease", side_effect=jobs.JobStorageError("down")):
            with self.assertLogs(jobs.logger, level="WARNING"):
                jobs._keep_lease("missing", "token", OneTick(), lost)
        self.assertTrue(lost.is_set())

    def test_watch_contention_is_bounded_and_explicit(self):
        job = jobs.create_job(USER, "contention")
        self.client.execute_error = WatchError("permanent conflict")
        with self.assertRaises(jobs.JobStorageError):
            jobs._mark_running(job.job_id)
        self.assertEqual(self.data(job.job_id)["status"], "queued")

    def test_local_physical_queue_is_bounded_even_after_repeated_cancel(self):
        with patch.object(jobs, "MAX_LOCAL_JOBS", 2):
            first = jobs.create_job(USER, "cancelled wrapper 1")
            self.assertEqual(jobs.cancel_job(USER, first.job_id).status, "cancelled")
            second = jobs.create_job(OTHER, "cancelled wrapper 2")
            self.assertEqual(jobs.cancel_job(OTHER, second.job_id).status, "cancelled")
            with self.assertRaises(OverloadedError):
                jobs.create_job(USER, "must not grow executor queue")
            self.assertEqual(len(self.pool.calls), 2)
            with patch("enterprise_rag.pipeline.query") as query:
                self.pool.run_one()
            query.assert_not_called()
            self.assertEqual(jobs.create_job(USER, "after real drain").status, "queued")

    def test_shutdown_waits_for_in_progress_admission_and_compensates_record(self):
        entered, allow_commit, stopped = threading.Event(), threading.Event(), threading.Event()
        actual_pool = ThreadPoolExecutor(max_workers=1)
        self.enterContext(patch.object(jobs, "_POOL", actual_pool))
        self.addCleanup(actual_pool.shutdown, wait=True, cancel_futures=True)
        self.addCleanup(allow_commit.set)
        fired, outcome = [], []

        def hook(pipe):
            if not fired:
                fired.append(True)
                entered.set()
                self.assertTrue(allow_commit.wait(timeout=3))
        self.client.before_execute = hook

        def create():
            try:
                jobs.create_job(USER, "admission in progress")
                outcome.append("accepted")
            except OverloadedError:
                outcome.append("overloaded")
        def stop():
            jobs.shutdown_job_worker()
            stopped.set()
        creator = threading.Thread(target=create)
        closer = threading.Thread(target=stop)
        with patch("enterprise_rag.pipeline.query") as query:
            creator.start()
            self.assertTrue(entered.wait(timeout=1))
            closer.start()
            try:
                deadline = time.monotonic() + 1
                while jobs._ACCEPTING and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertFalse(stopped.is_set())
            finally:
                allow_commit.set()
                creator.join(timeout=3)
                closer.join(timeout=3)
            query.assert_not_called()
        self.assertTrue(stopped.is_set())
        self.assertEqual(outcome, ["overloaded"])
        self.assertEqual(jobs._ADMITTING, set())
        self.assertEqual([(job.status, job.error) for job in jobs.list_jobs(USER)], [("failed", "enqueue_failed")])

    def test_shutdown_stops_admission_cancels_queue_and_waits_for_running(self):
        started, release, shutdown_done = threading.Event(), threading.Event(), threading.Event()
        actual_pool = ThreadPoolExecutor(max_workers=1)
        self.enterContext(patch.object(jobs, "_POOL", actual_pool))
        self.addCleanup(actual_pool.shutdown, wait=True, cancel_futures=True)
        self.addCleanup(release.set)

        def query(*args, **kwargs):
            started.set()
            release.wait(timeout=3)
            return {"answer": "settled"}
        with patch("enterprise_rag.pipeline.query", side_effect=query):
            active = jobs.create_job(USER, "running")
            self.assertTrue(started.wait(timeout=1))
            queued = jobs.create_job(USER, "queued")
            def shutdown():
                jobs.shutdown_job_worker()
                shutdown_done.set()
            thread = threading.Thread(target=shutdown)
            thread.start()
            try:
                deadline = time.monotonic() + 1
                while jobs._ACCEPTING and time.monotonic() < deadline:
                    time.sleep(0.005)
                with self.assertRaises(OverloadedError):
                    jobs.create_job(USER, "must reject")
                self.assertFalse(shutdown_done.is_set())
                self.assertEqual(jobs.get_job(USER, active.job_id).status, "running")
                deadline = time.monotonic() + 1
                while jobs.get_job(USER, queued.job_id).status == "queued" and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(jobs.get_job(USER, queued.job_id).error, "worker_stopped")
            finally:
                release.set()
                thread.join(timeout=3)
            self.assertTrue(shutdown_done.is_set())
        self.assertEqual(jobs.get_job(USER, active.job_id).status, "succeeded")
        self.assertEqual(jobs._FUTURES, {})

    def test_worker_can_restart_after_clean_lifespan_shutdown(self):
        pool = ThreadPoolExecutor(max_workers=1)
        self.enterContext(patch.object(jobs, "_POOL", pool))
        jobs.shutdown_job_worker()
        self.assertFalse(jobs._ACCEPTING)
        jobs.start_job_worker()
        replacement = jobs._POOL
        self.addCleanup(replacement.shutdown, wait=True, cancel_futures=True)
        self.assertIsNot(replacement, pool)
        self.assertTrue(jobs._ACCEPTING)
        with patch("enterprise_rag.pipeline.query", return_value={"answer": "fresh"}):
            job = jobs.create_job(USER, "fresh lifespan")
            jobs.shutdown_job_worker()
        # Shutdown may cancel it while still queued, or let the already-started job settle.
        self.assertIn(jobs.get_job(USER, job.job_id).status, {"succeeded", "failed"})
        self.assertEqual(jobs._FUTURES, {})


class LegacyRecordTest(JobTestCase):
    def test_legacy_owner_migration_is_cas_and_preserves_concurrent_cancel(self):
        job = jobs.Job("legacy", USER.user_id, "old", lease_expires_at=time.time() + 60)
        data = job.to_dict()
        data["lease_expires_at"] = job.lease_expires_at
        self.client.hset(jobs._key(job.job_id), mapping={"payload": json.dumps(data), "owner": USER.user_id.encode()})
        fired = []

        def hook(pipe):
            if not fired:
                fired.append(True)
                jobs.cancel_job(USER, job.job_id)
        self.client.before_execute = hook
        loaded = jobs.get_job(USER, job.job_id)
        self.assertEqual((loaded.status, loaded.owner), ("cancelled", USER.user_id))
        self.assertEqual(self.data(job.job_id)["schema_version"], 1)

    def test_legacy_active_without_lease_is_failed_not_resumed(self):
        for status in ("queued", "running", "cancelling"):
            job = jobs.Job("old_" + status, USER.user_id, "old", status=status)
            self.client.hset(jobs._key(job.job_id), mapping={"payload": json.dumps(job.to_dict()), "owner": USER.user_id})
            loaded = jobs.get_job(USER, job.job_id)
            self.assertEqual((loaded.status, loaded.error), ("failed", "worker_lost"))
            self.assertIsNone(jobs._mark_running(job.job_id))

    def test_corrupt_or_future_records_raise_storage_error(self):
        valid = jobs._record(jobs.Job("bad", USER.user_id, "old", status="succeeded"))
        payloads = ["{broken", "[]", json.dumps({**valid, "schema_version": 999}),
                    json.dumps({**valid, "job_id": "different"}),
                    json.dumps({**valid, "status": "invented"}),
                    json.dumps({**valid, "lease_expires_at": "bad"}),
                    json.dumps({**valid, "cancel_requested": "false"}),
                    json.dumps({**valid, "use_llm": "false"}),
                    json.dumps({**valid, "top_k": True}),
                    json.dumps({**valid, "owner": ""}),
                    json.dumps({**valid, "result": ["bad"]})]
        for raw in payloads:
            with self.subTest(raw=raw):
                self.client.hset(jobs._key("bad"), mapping={"payload": raw})
                with self.assertRaises(jobs.JobStorageError):
                    jobs.get_job(USER, "bad")


class RouteTest(unittest.TestCase):
    def resolve(self, coroutine):
        # AsyncMock completes immediately; no event loop, client socket or live server.
        with self.assertRaises(StopIteration) as result:
            coroutine.send(None)
        return result.exception.value

    def test_routes_offload_sync_storage_and_keep_response_contract(self):
        job = jobs.Job("route", USER.user_id, "question")
        with patch.object(routes.asyncio, "to_thread", new_callable=AsyncMock, return_value=job) as offload:
            response = self.resolve(routes.create_rag_job(routes.RagJobRequest(question="question", use_llm=True, top_k=7), USER))
            self.assertEqual(response["data"]["status"], "queued")
            offload.assert_awaited_once_with(routes.create_job, USER, "question", use_llm=True, top_k=7)
        with patch.object(routes.asyncio, "to_thread", new_callable=AsyncMock, return_value=[job]) as offload:
            self.assertEqual(len(self.resolve(routes.jobs(USER))["data"]["items"]), 1)
            offload.assert_awaited_once_with(routes.list_jobs, USER)
        with patch.object(routes.asyncio, "to_thread", new_callable=AsyncMock, return_value=job) as offload:
            self.assertEqual(self.resolve(routes.job_detail("route", USER))["data"]["job_id"], "route")
            offload.assert_awaited_once_with(routes.get_job, USER, "route")
        with patch.object(routes.asyncio, "to_thread", new_callable=AsyncMock, return_value=job) as offload:
            self.resolve(routes.job_cancel("route", USER))
            offload.assert_awaited_once_with(routes.cancel_job, USER, "route")


if __name__ == "__main__":
    unittest.main()
