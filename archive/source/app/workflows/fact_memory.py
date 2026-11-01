"""用户隔离、可纠错删除的事实记忆（S39）。

只保存用户明确要求记住的内容；不把模型回答或推断自动写成记忆。
Redis 是唯一权威存储：读写失败返回明确错误，不用本地副本复活已删除或过期内容。
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone, timedelta
from redis.exceptions import WatchError

from app.domain.user.models import UserContext

MAX_FACTS = 20
_TTL_SECONDS = 60 * 60 * 24 * 90
_TZ = timezone(timedelta(hours=8))
_REMEMBER = re.compile(r"^(?:请)?记住[：:，,\s]*(.+)$")


class MemoryUnavailable(RuntimeError):
    """权威记忆存储不可用；调用方必须与“成功删除/写入”区分。"""


def _key(user_id: str) -> str:
    return f"fact_memory:{user_id}"


def _client():
    from app.database.connection import get_redis_client

    return get_redis_client()


def _decode(raw: dict) -> dict[str, dict]:
    result = {}
    for value in raw.values():
        try:
            item = json.loads(value)
            if item.get("fact_id") and not item.get("deleted"):
                result[item["fact_id"]] = item
        except (TypeError, ValueError):
            continue
    return result


def _load(user_id: str) -> dict[str, dict]:
    try:
        return _decode(_client().hgetall(_key(user_id)))
    except Exception as exc:
        raise MemoryUnavailable(str(exc)) from exc


def _mutate(user_id: str, *, new_fact: dict | None = None,
            fact_id: str = "", text: str = "") -> dict | None:
    """容量淘汰、纠错和 TTL 同一事务提交；并发删除不会被旧副本复活。"""
    key = _key(user_id)
    try:
        client = _client()
        for _ in range(5):
            try:
                with client.pipeline() as pipe:
                    pipe.watch(key)
                    facts = _decode(pipe.hgetall(key))
                    if new_fact is not None:
                        fact = next((f for f in facts.values() if f.get("text") == new_fact["text"]), new_fact)
                    else:
                        if fact_id not in facts:
                            return None
                        fact = {**facts[fact_id], "text": text,
                                "updated_at": datetime.now(_TZ).isoformat()}
                    remaining = {**facts, fact["fact_id"]: fact}
                    candidates = [fid for fid in facts if fid != fact["fact_id"]]
                    evicted = sorted(candidates, key=lambda fid: (facts[fid].get("created_at", ""), fid))[:max(0, len(remaining) - MAX_FACTS)]
                    pipe.multi()
                    pipe.hset(key, fact["fact_id"], json.dumps(fact, ensure_ascii=False))
                    if evicted:
                        pipe.hdel(key, *evicted)
                    pipe.expire(key, _TTL_SECONDS)
                    pipe.execute()
                    return fact
            except WatchError:
                continue
        raise MemoryUnavailable("记忆并发更新冲突，请重试")
    except MemoryUnavailable:
        raise
    except Exception as exc:
        raise MemoryUnavailable(str(exc)) from exc


def remember_fact(user: UserContext, text: str, source: str = "user_explicit") -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("记忆内容不能为空")
    fact = {
        "fact_id": f"mem_{uuid.uuid4().hex[:10]}",
        "user_id": user.user_id,
        "text": text[:500],
        "source": source,
        "created_at": datetime.now(_TZ).isoformat(),
        "updated_at": datetime.now(_TZ).isoformat(),
        "deleted": False,
    }
    return _mutate(user.user_id, new_fact=fact)


def correct_fact(user: UserContext, fact_id: str, text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        raise ValueError("记忆内容不能为空")
    return _mutate(user.user_id, fact_id=fact_id, text=text[:500])


def forget_fact(user: UserContext, fact_id: str) -> bool:
    if fact_id not in _load(user.user_id):
        return False
    try:
        removed = int(_client().hdel(_key(user.user_id), fact_id))
    except Exception as exc:
        raise MemoryUnavailable(str(exc)) from exc
    return removed > 0


def list_facts(user: UserContext) -> list[dict]:
    facts = _load(user.user_id)
    return sorted(facts.values(), key=lambda item: item.get("created_at", ""))


def extract_explicit_fact(message: str) -> str:
    match = _REMEMBER.match((message or "").strip())
    return match.group(1).strip() if match else ""


def render_memory_block(facts: list[dict]) -> str:
    if not facts:
        return ""
    return "\n".join(f"- {fact['text']}" for fact in facts[:MAX_FACTS])
