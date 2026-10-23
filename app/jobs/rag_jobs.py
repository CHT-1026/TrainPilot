"""Redis 权威任务记录：原子领取、租约 fencing、取消与终态结算。

每 owner 的待执行上限跨进程生效；线程池执行数仍是每进程的限制。
失联任务仅标记失败并允许手动新建重跑，不自动重复可能收费的模型调用。
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from redis.exceptions import RedisError, WatchError

from app.core.backpressure import OverloadedError
from app.domain.user.models import UserContext
from enterprise_rag.budget import RequestCancelled

logger = logging.getLogger(__name__)

TERMINAL = {"succeeded", "failed", "cancelled"}
_ACTIVE = {"running", "cancelling"}
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-job")
_POOL_LOCK = threading.RLock()
_POOL_CHANGED = threading.Condition(_POOL_LOCK)
_ACCEPTING = True
_FUTURES: dict[str, Future] = {}
_ADMITTING: set[str] = set()
_TTL_SECONDS = 24 * 60 * 60
_SCHEMA_VERSION = 1
MAX_PENDING_JOBS = 64
MAX_LOCAL_JOBS = 128
_LEASE_SECONDS = 60
_QUEUE_LEASE_SECONDS = 10 * 60
_CAS_ATTEMPTS = 16


class JobStorageError(RuntimeError):
    """任务权威存储不可用或记录无法解码。"""


@dataclass
class Job:
    job_id: str
    owner: str
    question: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_requested: bool = False
    result: dict | None = None
    error: str = ""
    lease: str = ""
    lease_expires_at: float = 0.0
    use_llm: bool = False
    top_k: int | None = None

    def to_dict(self, include_result: bool = True) -> dict:
        payload = {
            "job_id": self.job_id, "question": self.question, "status": self.status,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "cancel_requested": self.cancel_requested, "error": self.error,
            "use_llm": self.use_llm, "top_k": self.top_k,
        }
        if include_result:
            payload["result"] = self.result
        return payload


def _key(job_id: str) -> str:
    return f"rag_job:{job_id}"


def _owner_key(owner: str) -> str:
    return f"rag_jobs:{owner}"


def _redis():
    try:
        from app.database.connection import get_redis_client

        return get_redis_client()
    except Exception:
        return None


def _client():
    client = _redis()
    if client is None:
        raise JobStorageError("Redis 不可用")
    return client


def _from_record(data: dict) -> Job:
    return Job(**{name: data[name] for name in Job.__dataclass_fields__})


def _record(job: Job) -> dict:
    return {**job.to_dict(), "owner": job.owner, "schema_version": _SCHEMA_VERSION,
            "lease": job.lease, "lease_expires_at": job.lease_expires_at}


def _decode(raw, job_id: str, reader) -> dict:
    """Normalize legacy payloads under the caller's WATCH; never rewrite outside CAS."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("job_id") != job_id:
            raise ValueError("invalid job identity")
        version = data.get("schema_version", 0)
        if type(version) is not int or not 0 <= version <= _SCHEMA_VERSION:
            raise ValueError("unsupported job schema")
        owner = data.get("owner") or reader.hget(_key(job_id), "owner")
        if isinstance(owner, bytes):
            owner = owner.decode("utf-8")
        if not isinstance(owner, str) or not owner:
            raise ValueError("missing owner")
        data["owner"] = owner
        for name in ("question", "error", "lease"):
            if not isinstance(data.get(name, ""), str):
                raise ValueError(f"invalid {name}")
        if data.get("status", "queued") not in TERMINAL | _ACTIVE | {"queued"}:
            raise ValueError("invalid status")
        for name in ("cancel_requested", "use_llm"):
            if type(data.get(name, False)) is not bool:
                raise ValueError(f"invalid {name}")
        top_k = data.get("top_k")
        if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 50):
            raise ValueError("invalid top_k")
        if data.get("result") is not None and not isinstance(data["result"], dict):
            raise ValueError("invalid result")
        for name in ("created_at", "updated_at", "lease_expires_at"):
            value = data.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {name}")
        job = Job(job_id=job_id, owner=owner, question=data.get("question", ""),
                  status=data.get("status", "queued"),
                  created_at=data.get("created_at", 0.0), updated_at=data.get("updated_at", 0.0),
                  cancel_requested=data.get("cancel_requested", False), result=data.get("result"),
                  error=data.get("error", ""), lease=data.get("lease", ""),
                  lease_expires_at=float(data.get("lease_expires_at", 0.0)),
                  use_llm=data.get("use_llm", False), top_k=top_k)
        return _record(job)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise JobStorageError(f"任务 {job_id} 记录损坏或版本不兼容") from exc


def _expire_lease(data: dict):
    if data["status"] in TERMINAL:
        return None
    if data["lease_expires_at"] > time.time() and (data["status"] == "queued" or data["lease"]):
        return None
    # Old active records without a lease cannot be safely resumed or charged again.
    reason = "queue_timeout" if data["status"] == "queued" and data["lease_expires_at"] > 0 else "worker_lost"
    data.update(status="failed", result=None, lease="", lease_expires_at=0.0,
                error=reason, updated_at=time.time())
    return data


def _write_record(pipe, data: dict) -> None:
    key = _key(data["job_id"])
    pipe.hset(key, mapping={"payload": json.dumps(data, ensure_ascii=False), "owner": data["owner"]})
    pipe.expire(key, _TTL_SECONDS)
    pipe.sadd(_owner_key(data["owner"]), data["job_id"])
    pipe.expire(_owner_key(data["owner"]), _TTL_SECONDS)


def _persist(job: Job, *, enforce_capacity: bool = False) -> None:
    """Create once; owner index + payload + TTL and capacity reservation commit together."""
    client = _client()
    if job.lease_expires_at <= 0 and job.status == "queued":
        job.lease_expires_at = time.time() + _QUEUE_LEASE_SECONDS
    key, owner_key = _key(job.job_id), _owner_key(job.owner)
    try:
        with client.pipeline() as pipe:
            for _ in range(_CAS_ATTEMPTS):
                try:
                    pipe.watch(key, owner_key)
                    if pipe.hget(key, "payload") is not None:
                        raise JobStorageError("任务 ID 已存在")
                    if enforce_capacity:
                        ids = pipe.smembers(owner_key) or set()
                        ids = [value.decode() if isinstance(value, bytes) else value for value in ids]
                        if ids:
                            pipe.watch(*[_key(value) for value in ids])
                        pending = 0
                        for job_id in ids:
                            raw = pipe.hget(_key(job_id), "payload")
                            if raw:
                                data = _decode(raw, job_id, pipe)
                                if data["owner"] == job.owner:
                                    _expire_lease(data)
                                    pending += data["status"] not in TERMINAL
                        if pending >= MAX_PENDING_JOBS:
                            raise OverloadedError("后台任务排队已满，请稍后重试")
                    pipe.multi()
                    _write_record(pipe, _record(job))
                    pipe.execute()
                    return
                except WatchError:
                    continue
    except RedisError as exc:
        raise JobStorageError("任务写入失败") from exc
    raise JobStorageError("任务创建冲突，请稍后重试")


def _transition(job_id: str, change) -> Job | None:
    client, key = _client(), _key(job_id)
    try:
        with client.pipeline() as pipe:
            for _ in range(_CAS_ATTEMPTS):
                try:
                    pipe.watch(key)
                    raw = pipe.hget(key, "payload")
                    if raw is None:
                        return None
                    data = _decode(raw, job_id, pipe)
                    original = json.loads(raw)
                    _expire_lease(data)
                    change(data)
                    if data == original:
                        return _from_record(data)
                    pipe.multi()
                    _write_record(pipe, data)
                    pipe.execute()
                    return _from_record(data)
                except WatchError:
                    continue
    except RedisError as exc:
        raise JobStorageError("任务状态更新失败") from exc
    raise JobStorageError("任务状态冲突，请稍后重试")


def _load(job_id: str) -> Job | None:
    return _transition(job_id, lambda data: None)


def _mark_running(job_id: str) -> Job | None:
    token = uuid.uuid4().hex

    def change(data: dict):
        if data["status"] != "queued":
            return
        if data["cancel_requested"]:
            data.update(status="cancelled", result=None, lease="", lease_expires_at=0.0,
                        updated_at=time.time())
        else:
            data.update(status="running", lease=token,
                        lease_expires_at=time.time() + _LEASE_SECONDS, updated_at=time.time())

    job = _transition(job_id, change)
    return job if job is not None and job.status == "running" and job.lease == token else None


def _renew_lease(job_id: str, token: str) -> bool:
    def change(data: dict):
        if data["status"] in _ACTIVE and data["lease"] == token:
            data.update(lease_expires_at=time.time() + _LEASE_SECONDS, updated_at=time.time())

    job = _transition(job_id, change)
    return job is not None and job.status in _ACTIVE and job.lease == token


def _keep_lease(job_id: str, token: str, stop: threading.Event, lost: threading.Event) -> None:
    while not stop.wait(_LEASE_SECONDS / 3):
        try:
            if _renew_lease(job_id, token):
                continue
        except JobStorageError:
            logger.warning("任务续租失败 %s", job_id, exc_info=True)
        # Stop new work when authority cannot be verified; never reacquire an expired lease.
        lost.set()
        return


def _finish_job(job_id: str, status: str, result: dict | None, error: str,
                *, token: str = "") -> Job | None:
    if status not in TERMINAL:
        raise ValueError("任务结算必须为终态")

    def change(data: dict):
        if data["status"] in TERMINAL:
            return
        if data["status"] in _ACTIVE:
            if not token or data["lease"] != token:
                return
        elif token or status != "failed":
            return
        cancelled = data["cancel_requested"] or data["status"] == "cancelling" or status == "cancelled"
        data.update(status="cancelled" if cancelled else status,
                    cancel_requested=cancelled, result=None if cancelled else result,
                    error="" if cancelled else error, lease="", lease_expires_at=0.0,
                    updated_at=time.time())

    return _transition(job_id, change)


def _cancel_record(job_id: str) -> Job | None:
    def change(data: dict):
        if data["status"] in TERMINAL:
            return
        data.update(cancel_requested=True, updated_at=time.time())
        if data["status"] == "queued":
            data.update(status="cancelled", result=None, lease="", lease_expires_at=0.0)
        else:
            data["status"] = "cancelling"

    return _transition(job_id, change)


def _cancel_requested(job: Job) -> bool:
    current = _load(job.job_id)
    return (current is None or current.status not in _ACTIVE or current.lease != job.lease
            or current.cancel_requested)


def _run(job_id: str, user: UserContext) -> None:
    stop, lost = threading.Event(), threading.Event()
    job = None
    heartbeat = None
    try:
        job = _mark_running(job_id)
        if job is None:
            return
        if job.owner != user.user_id:
            _finish_job(job_id, "failed", None, "owner_mismatch", token=job.lease)
            return
        heartbeat = threading.Thread(target=_keep_lease, args=(job_id, job.lease, stop, lost),
                                     daemon=True, name=f"rag-lease-{job_id[:8]}")
        heartbeat.start()
        from enterprise_rag.pipeline import query

        result = query(job.question, user=user, top_k=job.top_k, use_llm=job.use_llm,
                       cancel_check=lambda: lost.is_set() or _cancel_requested(job))
        payload = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        _finish_job(job_id, "succeeded", payload, "", token=job.lease)
    except RequestCancelled:
        if job is not None:
            try:
                _finish_job(job_id, "cancelled", None, "", token=job.lease)
            except JobStorageError:
                logger.error("任务取消结算存储不可用 %s", job_id, exc_info=True)
    except Exception as exc:
        if job is not None:
            try:
                _finish_job(job_id, "failed", None, type(exc).__name__, token=job.lease)
            except JobStorageError:
                logger.error("任务失败结算存储不可用 %s", job_id, exc_info=True)
        logger.error("后台任务失败 %s: %s", job_id, type(exc).__name__, exc_info=True)
    finally:
        stop.set()
        if heartbeat is not None:
            heartbeat.join()


def _forget_future(job_id: str, future: Future) -> None:
    with _POOL_LOCK:
        _FUTURES.pop(job_id, None)


def start_job_worker() -> None:
    """Start this process's executor; lifespan may call again after a prior shutdown."""
    global _POOL, _ACCEPTING
    with _POOL_LOCK:
        if not _ACCEPTING:
            _POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-job")
            _ACCEPTING = True


def shutdown_job_worker() -> None:
    """Stop admission, cancel local queued work, and wait for running workers to settle."""
    global _ACCEPTING
    with _POOL_CHANGED:
        _ACCEPTING = False
        while _ADMITTING:
            _POOL_CHANGED.wait()
        pool, pending = _POOL, list(_FUTURES.items())
    for job_id, future in pending:
        if future.cancel():
            try:
                _finish_job(job_id, "failed", None, "worker_stopped")
            except JobStorageError:
                logger.error("停机任务结算失败 %s；由租约到期标记失联", job_id, exc_info=True)
    pool.shutdown(wait=True, cancel_futures=True)


def create_job(user: UserContext, question: str, *, use_llm: bool = False,
               top_k: int | None = None) -> Job:
    question = (question or "").strip()
    if not question:
        raise ValueError("任务问题不能为空")
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 50):
        raise ValueError("top_k 必须为 1 到 50")
    job = Job(job_id=f"job_{uuid.uuid4().hex}", owner=user.user_id,
              question=question[:2000], use_llm=bool(use_llm), top_k=top_k)
    with _POOL_LOCK:
        if not _ACCEPTING:
            raise OverloadedError("任务执行器正在停止，请稍后重试")
        if len(_FUTURES) + len(_ADMITTING) >= MAX_LOCAL_JOBS:
            raise OverloadedError("本机任务队列已满，请稍后重试")
        _ADMITTING.add(job.job_id)
    try:
        _persist(job, enforce_capacity=True)
        try:
            with _POOL_LOCK:
                if not _ACCEPTING:
                    raise RuntimeError("executor stopping")
                future = _POOL.submit(_run, job.job_id, user)
                _FUTURES[job.job_id] = future
                future.add_done_callback(lambda value: _forget_future(job.job_id, value))
        except RuntimeError as exc:
            _finish_job(job.job_id, "failed", None, "enqueue_failed")
            raise OverloadedError("任务暂时无法入队，请稍后重试") from exc
        return job
    finally:
        with _POOL_CHANGED:
            _ADMITTING.discard(job.job_id)
            _POOL_CHANGED.notify_all()


def get_job(user: UserContext, job_id: str) -> Job | None:
    job = _load(job_id)
    return job if job is not None and (job.owner == user.user_id or user.is_admin) else None


def list_jobs(user: UserContext) -> list[Job]:
    client = _client()
    try:
        ids = client.smembers(_owner_key(user.user_id)) or set()
        jobs = []
        for job_id in ids:
            if isinstance(job_id, bytes):
                job_id = job_id.decode("utf-8")
            job = _load(job_id)
            if job is None or job.owner != user.user_id:
                client.srem(_owner_key(user.user_id), job_id)
                continue
            jobs.append(job)
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)
    except RedisError as exc:
        raise JobStorageError("任务索引读取失败") from exc


def cancel_job(user: UserContext, job_id: str) -> Job | None:
    job = get_job(user, job_id)
    if job is None:
        return None
    # Leave the lightweight wrapper queued until consumed: Future.cancel() does not
    # remove ThreadPoolExecutor's work item and would prematurely free local capacity.
    return _cancel_record(job_id)
