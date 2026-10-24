"""请求级预算、阶段计时与调用计数。

一次企业问答可能触发查询改写、最多 8 个检索变式（各自 embedding）、图扩展和
最多两次生成。这里把“允许调用什么、已调用多少次、还剩多少时间”变成显式状态，
避免 use_llm=false 或取消之后仍发生外部调用。
"""
from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

from enterprise_rag import config

_CURRENT_BUDGET: contextvars.ContextVar[RequestBudget | None] = contextvars.ContextVar(
    "rag_request_budget", default=None
)


class RequestCancelled(RuntimeError):
    """Cooperative cancellation requested by the caller at a stage boundary."""


def current_budget() -> "RequestBudget | None":
    return _CURRENT_BUDGET.get()


def model_usage_available(usage) -> bool:
    """An empty provider usage object is not a known zero-token response."""
    return (getattr(usage, "prompt_tokens", None) is not None
            and getattr(usage, "completion_tokens", None) is not None)


@contextmanager
def bind_budget(budget: "RequestBudget | None"):
    token = _CURRENT_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _CURRENT_BUDGET.reset(token)



@dataclass
class StageRecord:
    name: str
    elapsed_ms: float = 0.0
    calls: int = 0
    ok: bool = True
    detail: str = ""


@dataclass
class CallRecord:
    kind: str
    stage: str
    ok: bool = True
    detail: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    usage_available: bool = False


@dataclass
class RequestBudget:
    deadline_seconds: float = field(default_factory=lambda: config.REQUEST_TIMEOUT_SECONDS)
    offline: bool = False
    allow_model: bool = True
    max_model_calls: int = field(default_factory=lambda: config.MAX_MODEL_CALLS)
    max_embedding_calls: int = field(default_factory=lambda: config.MAX_EMBEDDING_CALLS)
    cancel_check: Callable[[], bool] | None = field(default=None, repr=False)

    started_at: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    model_calls: int = 0
    embedding_calls: int = 0
    stages: list[StageRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    stopped_reason: str = ""
    _stage_stack: list[StageRecord] = field(default_factory=list, repr=False)

    # ---- 时间 ---------------------------------------------------------
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - (time.monotonic() - self.started_at))

    def expired(self) -> bool:
        return self.remaining_seconds() <= 0

    # ---- 调用许可 -----------------------------------------------------
    def check_cancel(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            self.stop("cancelled")
            raise RequestCancelled("cancelled")

    def can_call_model(self) -> bool:
        self.check_cancel()
        with self._lock:
            return (
                not self.offline
                and self.allow_model
                and not self.expired()
                and self.model_calls < self.max_model_calls
                and not self.stopped_reason
            )

    def can_call_embedding(self) -> bool:
        self.check_cancel()
        with self._lock:
            return (
                not self.offline
                and not self.expired()
                and self.embedding_calls < self.max_embedding_calls
                and not self.stopped_reason
            )

    def charge_model(self, *, stage: str = "", detail: str = "", model: str = "",
                     prompt_tokens: int = 0, completion_tokens: int = 0,
                     total_tokens: int = 0, ok: bool = True) -> None:
        with self._lock:
            self.model_calls += 1
            self.calls.append(CallRecord(
                kind="model", stage=stage, ok=ok, detail=detail, model=model,
                prompt_tokens=prompt_tokens or 0,
                completion_tokens=completion_tokens or 0,
                total_tokens=total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
                usage_available=bool(prompt_tokens or completion_tokens or total_tokens),
            ))
            if self._stage_stack:
                self._stage_stack[-1].calls += 1
                self._stage_stack[-1].ok = self._stage_stack[-1].ok and ok

    def charge_embedding(self, *, stage: str = "", detail: str = "", ok: bool = True) -> None:
        with self._lock:
            self.embedding_calls += 1
            self.calls.append(CallRecord(kind="embedding", stage=stage, ok=ok, detail=detail))
            if self._stage_stack:
                self._stage_stack[-1].calls += 1
                self._stage_stack[-1].ok = self._stage_stack[-1].ok and ok

    def note_call_result(self, *, ok: bool = True, detail: str = "") -> None:
        """Update the most recent call attempt (used after a provider responds)."""
        with self._lock:
            if self.calls:
                self.calls[-1].ok = ok
                if detail:
                    self.calls[-1].detail = detail
            if self._stage_stack:
                self._stage_stack[-1].ok = self._stage_stack[-1].ok and ok

    def note_model_usage(self, *, prompt_tokens: int = 0, completion_tokens: int = 0,
                         total_tokens: int = 0, available: bool = True) -> None:
        """Attach token usage to the already-counted model attempt."""
        with self._lock:
            for call in reversed(self.calls):
                if call.kind == "model":
                    call.ok = True
                    call.prompt_tokens = prompt_tokens or 0
                    call.completion_tokens = completion_tokens or 0
                    call.total_tokens = total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)
                    call.usage_available = available
                    return

    def stop(self, reason: str) -> None:
        with self._lock:
            if not self.stopped_reason:
                self.stopped_reason = reason

    def stopped(self) -> bool:
        return bool(self.stopped_reason) or self.expired()

    # ---- 阶段 ---------------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        self.check_cancel()
        record = StageRecord(name=name)
        with self._lock:
            self.stages.append(record)
            self._stage_stack.append(record)
        started = time.monotonic()
        try:
            yield record
        except Exception:
            record.ok = False
            raise
        finally:
            record.elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            with self._lock:
                if self._stage_stack and self._stage_stack[-1] is record:
                    self._stage_stack.pop()

    # ---- 输出 ---------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for call in self.calls:
                usage["prompt_tokens"] += call.prompt_tokens
                usage["completion_tokens"] += call.completion_tokens
                usage["total_tokens"] += call.total_tokens
            return {
                "offline": self.offline,
                "allow_model": self.allow_model,
                "model_calls": self.model_calls,
                "embedding_calls": self.embedding_calls,
                "max_model_calls": self.max_model_calls,
                "max_embedding_calls": self.max_embedding_calls,
                "elapsed_ms": round(self.elapsed_ms(), 2),
                "remaining_ms": round(self.remaining_seconds() * 1000, 2),
                "stopped_reason": self.stopped_reason
                or ("deadline_exceeded" if self.expired() and self.calls else ""),
                "stages": [
                    {
                        "name": stage.name,
                        "elapsed_ms": stage.elapsed_ms,
                        "calls": stage.calls,
                        "ok": stage.ok,
                        "detail": stage.detail,
                    }
                    for stage in self.stages
                ],
                "calls": [
                    {
                        "kind": call.kind,
                        "stage": call.stage,
                        "ok": call.ok,
                        "detail": call.detail,
                        "model": call.model,
                        "usage_available": call.usage_available if call.kind == "model" else None,
                        "prompt_tokens": call.prompt_tokens,
                        "completion_tokens": call.completion_tokens,
                        "total_tokens": call.total_tokens,
                    }
                    for call in self.calls
                ],
                "usage": usage,
                "model_usage_complete": all(call.usage_available for call in self.calls if call.kind == "model"),
                "unknown_model_usage_calls": sum(not call.usage_available for call in self.calls if call.kind == "model"),
            }


def new_budget(use_llm: bool | None = None, offline: bool = False) -> RequestBudget:
    allow_model = config.USE_LLM if use_llm is None else bool(use_llm)
    return RequestBudget(offline=offline, allow_model=allow_model)
